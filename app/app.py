import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
from pathlib import Path
from datetime import datetime
import gc
import uuid
from functools import wraps
from itertools import islice
from threading import RLock

from audio_codes import AUDIO_END, END_OF_TURN, build_prompt, parse_output, split_codes

# Import required libraries for direct GGUF inference
try:
    from snac import SNAC
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please run Install in Pinokio to install the required dependencies.")
    IMPORTS_SUCCESSFUL = False

# === Konfiguration ===
# Model configurations
MODELS = {
    "english": {
        "repo_id": os.environ.get("ORPHEUS_REPO", "lex-au/Orpheus-3b-FT-Q8_0.gguf"),
        "filename": os.environ.get("ORPHEUS_FILENAME", "Orpheus-3b-FT-Q8_0.gguf"),
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    },
    "german": {
        "repo_id": "lex-au/Orpheus-3b-German-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-German-FT-Q8_0.gguf",
        "voices": ["Jana", "Thomas", "Max"]
    },
    "italian_spanish": {
        "repo_id": "lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf",
        "voices": ["Javi", "Sergio", "Maria", "Pietro", "Giulia", "Carlo"]
    },
    "french": {
        "repo_id": "lex-au/Orpheus-3b-French-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-French-FT-Q8_0.gguf",
        "voices": ["Pierre", "Amelie", "Marie"]
    },
    "korean": {
        "repo_id": "lex-au/Orpheus-3b-Korean-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-Korean-FT-Q8_0.gguf",
        "voices": ["유나", "준서"]
    },
    "chinese": {
        "repo_id": "lex-au/Orpheus-3b-Chinese-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-Chinese-FT-Q8_0.gguf",
        "voices": ["长乐", "白芷"]
    },
    "hindi": {
        "repo_id": "lex-au/Orpheus-3b-Hindi-FT-Q8_0.gguf",
        "filename": "Orpheus-3b-Hindi-FT-Q8_0.gguf",
        "voices": ["ऋतिका"]
    }
}

SNAC_MODEL_PATH = os.environ.get("SNAC_MODEL", "hubertsiuzdak/snac_24khz")

# Ausgabeverzeichnis für WAVs
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Current model selection
ORPHEUS_N_CTX = 4096
MODEL_LOCK = RLock()


def serialized_models(fn):
    """Protect native model state across loading and inference, including API calls."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with MODEL_LOCK:
            return fn(*args, **kwargs)
    return wrapped

# Global model storage
LOADED_MODELS = {
    "snac_model": None,
    "orpheus_model": None,
    "device": None,
    "current_model_type": None
}

@serialized_models
def load_models(model_type="english"):
    """Load Orpheus TTS and SNAC models"""
    if not IMPORTS_SUCCESSFUL:
        raise ImportError("Required libraries are not installed. Please install the required dependencies.")

    if model_type not in MODELS:
        raise ValueError(f"Unknown model: {model_type}")

    # Check if we need to reload models
    if (LOADED_MODELS["orpheus_model"] is not None and 
        LOADED_MODELS["current_model_type"] == model_type):
        return  # Same models already loaded
    if LOADED_MODELS["orpheus_model"] is not None:
        prev_model = LOADED_MODELS["orpheus_model"]
        try:
            if hasattr(prev_model, "close"):
                prev_model.close()
        except Exception as e:
            print(f"Warning: failed to close previous Orpheus model: {e}")
        finally:
            del prev_model
            LOADED_MODELS["orpheus_model"] = None
            LOADED_MODELS["current_model_type"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Loading Orpheus TTS models for {model_type}...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load SNAC model if not already loaded
        if LOADED_MODELS["snac_model"] is None:
            print("Loading SNAC model...")
            snac_model = SNAC.from_pretrained(SNAC_MODEL_PATH).eval()
            snac_model = snac_model.to(device)
            LOADED_MODELS["snac_model"] = snac_model

        # Get model configuration
        model_config = MODELS[model_type]
        
        # Download GGUF model file
        print(f"Downloading GGUF model from {model_config['repo_id']}/{model_config['filename']}...")
        model_path = hf_hub_download(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            cache_dir=str(APP_DIR / "models")
        )
        print(f"Model downloaded to: {model_path}")

        # Load GGUF model with llama-cpp-python
        print("Loading Orpheus GGUF model...")
        n_gpu_layers = -1 if device == "cuda" else 0  # Use all GPU layers if CUDA available
        
        if device == "cuda":
            print(f"Using GPU with {n_gpu_layers} layers")
        else:
            print("Using CPU")
            
        orpheus_model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            verbose=True,  # Enable verbose to see GPU info
            n_ctx=ORPHEUS_N_CTX,  # Context window
            n_threads=4,  # CPU threads
        )

        LOADED_MODELS.update({
            "orpheus_model": orpheus_model,
            "device": device,
            "current_model_type": model_type
        })

        print("Models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def redistribute_codes(code_list, snac_model):
    """Decode validated codes without allocating an autograd graph."""
    device = next(snac_model.parameters()).device
    codes = [
        torch.tensor(layer, dtype=torch.int64, device=device).unsqueeze(0)
        for layer in split_codes(code_list)
    ]
    with torch.inference_mode():
        audio = snac_model.decode(codes).squeeze().cpu().numpy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("The audio decoder returned empty or non-finite samples.")
    return audio


@serialized_models
def synthesize(text: str, voice: str, model_type: str, temperature: float, top_p: float, repetition_penalty: float, max_new_tokens: int):
    """Generate speech from text using Orpheus TTS"""
    if not text or not text.strip():
        return None, "Please enter text."

    try:
        if model_type not in MODELS:
            raise ValueError(f"Unknown model: {model_type}")
        if voice not in MODELS[model_type]["voices"]:
            voice = MODELS[model_type]["voices"][0]
        if not 0.1 <= temperature <= 1.5 or not 0.1 <= top_p <= 1.0:
            raise ValueError("Temperature or Top-p is outside the supported range.")
        if not 1.0 <= repetition_penalty <= 2.0:
            raise ValueError("Repetition Penalty must be between 1 and 2.")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) != max_new_tokens or not 100 <= max_new_tokens <= 3500:
            raise ValueError("Max New Tokens must be an integer between 100 and 3500.")
        max_new_tokens = int(max_new_tokens)
        # Load models if not already loaded
        load_models(model_type)
        
        snac_model = LOADED_MODELS["snac_model"]
        orpheus_model = LOADED_MODELS["orpheus_model"]
        prompt_ids = build_prompt(orpheus_model, text.strip(), voice)
        max_token_budget = orpheus_model.n_ctx() - len(prompt_ids)
        if max_token_budget < 7:
            return None, "Prompt is too long for the model context."
        safe_max_tokens = min(max_new_tokens, max_token_budget)

        # Keep generated IDs intact: decoding and re-encoding can lose special tokens.
        stream = orpheus_model.generate(
            prompt_ids,
            temp=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
        )
        generated_ids = []
        try:
            for token in islice(stream, safe_max_tokens):
                if token in (AUDIO_END, END_OF_TURN, orpheus_model.token_eos()):
                    break
                generated_ids.append(token)
        finally:
            stream.close()
        
        # Parse output and generate audio
        code_list = parse_output(generated_ids)
        audio_samples = redistribute_codes(code_list, snac_model)
        
        # Save to file
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_voice = "".join(c for c in voice if c.isalnum() or c in ("-","_"))
        unique_id = uuid.uuid4().hex[:8]
        out_wav = OUTPUT_DIR / f"orpheus_{model_type}_{safe_voice}_{ts}_{unique_id}.wav"
        
        # Save audio file
        sf.write(str(out_wav), audio_samples, 24000)
        
        return str(out_wav), "Done!"
        
    except Exception as e:
        print(f"Error in synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error during synthesis: {e}"

# Helper function to update voices based on model selection
def update_voices(model_type):
    return gr.Dropdown(choices=MODELS[model_type]["voices"], value=MODELS[model_type]["voices"][0])

# Gradio Interface
with gr.Blocks(title="Orpheus TTS – Multi-Model") as demo:
    gr.Markdown(
        """
        # Orpheus TTS – Multi-Model (GGUF)
        This UI supports multiple Orpheus TTS models with GGUF format for efficient inference.

        **Available Models:**
        - **English** (Q8_0 GGUF, ~3GB): Original Orpheus with voices: tara, leah, jess, leo, dan, mia, zac, zoe
        - **German** (Q8_0 GGUF, ~3GB): German fine-tuned model with voices: Jana (female, clear), Thomas (male, authoritative), Max (male, energetic)
        - **Italian/Spanish** (Q8_0 GGUF, ~3GB): Multi-language model with voices:
          - **Spanish**: Javi (male, warm), Sergio (male, professional), Maria (female, friendly)  
          - **Italian**: Pietro (male, passionate), Giulia (female, expressive), Carlo (male, refined)
        - **French** (Q8_0 GGUF, ~3GB): French fine-tuned model with voices: Pierre (male, sophisticated), Amelie (female, elegant), Marie (female, spirited)
        - **Korean** (Q8_0 GGUF, ~3GB): Korean fine-tuned model with voices: 유나 (female, melodic), 준서 (male, confident)
        - **Chinese** (Q8_0 GGUF, ~3GB): Mandarin fine-tuned model with voices: 长乐 (female, gentle), 白芷 (female, clear)
        - **Hindi** (Q8_0 GGUF, ~3GB): Hindi fine-tuned model with voice: ऋतिका (female, expressive)

        **All models use Q8_0 GGUF format:**
        - Good quality with consistent performance across languages
        - ~3GB file size each
        - Emotion tag support for all models
        - Higher memory usage but excellent expressiveness

        **Emotion Tags for All Models:**
        Use these tags in your text: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

        On first run, models will be automatically downloaded when selected.
        """
    )

    with gr.Row():
        model_selection = gr.Dropdown(
            choices=["english", "german", "italian_spanish", "french", "korean", "chinese", "hindi"], 
            value="english", 
            label="Model/Language"
        )
    
    with gr.Row():
        text = gr.Textbox(
            label="Text", 
            placeholder="Enter your text here… (All models support emotion tags like <laugh>, <sigh>, etc.)", 
            lines=4
        )
    
    with gr.Row():
        voice = gr.Dropdown(MODELS["english"]["voices"], value="tara", label="Voice")
    
    with gr.Row():
        temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, step=0.05, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, label="Top-p")
        repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.3, step=0.05, label="Repetition Penalty")
        max_new_tokens = gr.Slider(minimum=100, maximum=3500, value=2000, step=100, label="Max New Tokens")

    run_btn = gr.Button("Convert to Speech (Generate WAV)")
    out_audio = gr.Audio(label="Result (WAV)", type="filepath")
    out_status = gr.Markdown()

    # Update voices when model changes
    model_selection.change(
        fn=update_voices,
        inputs=[model_selection],
        outputs=[voice]
    )

    run_btn.click(
        fn=synthesize,
        inputs=[text, voice, model_selection, temperature, top_p, repetition_penalty, max_new_tokens],
        outputs=[out_audio, out_status],
        api_name="synthesize",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Gradio server port (default: next available port)")
    args = parser.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
