import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
import sys
from pathlib import Path
from datetime import datetime
import re

# Import required libraries for direct GGUF inference
try:
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download, login, hf_hub_download
    from llama_cpp import Llama
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install snac transformers huggingface_hub soundfile llama-cpp-python")
    IMPORTS_SUCCESSFUL = False

# === Konfiguration ===
# Orpheus model configuration
ORPHEUS_REPO_ID = os.environ.get("ORPHEUS_REPO", "unsloth/orpheus-3b-0.1-ft-GGUF")
ORPHEUS_FILENAME = os.environ.get("ORPHEUS_FILENAME", "orpheus-3b-0.1-ft-F16.gguf")
TOKENIZER_REPO_ID = os.environ.get("TOKENIZER_REPO", "unsloth/orpheus-3b-0.1-ft")  # For tokenizer
SNAC_MODEL_PATH = os.environ.get("SNAC_MODEL", "hubertsiuzdak/snac_24khz")

# Ausgabeverzeichnis für WAVs
OUTPUT_DIR = Path("outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verfügbare Stimmen
VOICES = [
    "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"
]

# Global model storage
LOADED_MODELS = {
    "snac_model": None,
    "orpheus_model": None,
    "tokenizer": None,
    "device": None
}

def load_models():
    """Load Orpheus TTS and SNAC models"""
    if not IMPORTS_SUCCESSFUL:
        raise ImportError("Required libraries are not installed. Please install the required dependencies.")

    global LOADED_MODELS

    if LOADED_MODELS["orpheus_model"] is not None:
        return  # Models already loaded

    print("Loading Orpheus TTS models...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load SNAC model
        print("Loading SNAC model...")
        snac_model = SNAC.from_pretrained(SNAC_MODEL_PATH).eval()
        snac_model = snac_model.to(device)

        # Download GGUF model file
        print(f"Downloading GGUF model from {ORPHEUS_REPO_ID}/{ORPHEUS_FILENAME}...")
        model_path = hf_hub_download(
            repo_id=ORPHEUS_REPO_ID,
            filename=ORPHEUS_FILENAME,
            cache_dir="./models"
        )
        print(f"Model downloaded to: {model_path}")

        # Load GGUF model with llama-cpp-python
        print("Loading Orpheus GGUF model...")
        n_gpu_layers = -1 if device == "cuda" else 0  # Use all GPU layers if CUDA available
        orpheus_model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_ctx=4096,  # Context window
            n_threads=4,  # CPU threads
        )

        # Load tokenizer from the original repo (non-GGUF)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO_ID)

        LOADED_MODELS = {
            "snac_model": snac_model,
            "orpheus_model": orpheus_model,
            "tokenizer": tokenizer,
            "device": device
        }

        print("Models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

def process_prompt(prompt, voice, tokenizer, device):
    """Process text prompt for the model"""
    prompt = f"{voice}: {prompt}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
    
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    attention_mask = torch.ones_like(modified_input_ids)
    
    return modified_input_ids.to(device), attention_mask.to(device)

def parse_output(generated_ids):
    """Parse output tokens to audio codes"""
    token_to_find = 128257
    token_to_remove = 128258
    
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
        
    return code_lists[0] if code_lists else []

def redistribute_codes(code_list, snac_model):
    """Redistribute codes for audio generation"""
    try:
        device = next(snac_model.parameters()).device
        
        if not code_list or len(code_list) < 7:
            print(f"Warning: code_list is too short: {len(code_list) if code_list else 0} elements")
            return np.zeros(24000, dtype=np.float32)
        
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        for i in range((len(code_list)+1)//7):
            if 7*i < len(code_list):
                layer_1.append(code_list[7*i])
            if 7*i+1 < len(code_list):
                layer_2.append(code_list[7*i+1]-4096)
            if 7*i+2 < len(code_list):
                layer_3.append(code_list[7*i+2]-(2*4096))
            if 7*i+3 < len(code_list):
                layer_3.append(code_list[7*i+3]-(3*4096))
            if 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+4]-(4*4096))
            if 7*i+5 < len(code_list):
                layer_3.append(code_list[7*i+5]-(5*4096))
            if 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+6]-(6*4096))
        
        codes = [
            torch.tensor(layer_1, device=device).unsqueeze(0),
            torch.tensor(layer_2, device=device).unsqueeze(0),
            torch.tensor(layer_3, device=device).unsqueeze(0)
        ]
        
        audio_hat = snac_model.decode(codes)
        audio_hat_squeezed = audio_hat.squeeze()
        
        return audio_hat_squeezed.detach().cpu().numpy()
        
    except Exception as e:
        print(f"Error in redistribute_codes: {e}")
        return np.zeros(24000, dtype=np.float32)

def synthesize(text: str, voice: str, temperature: float, top_p: float, repetition_penalty: float, max_new_tokens: int):
    """Generate speech from text using Orpheus TTS"""
    if not text or not text.strip():
        return None, "Please enter text."

    try:
        # Load models if not already loaded
        load_models()
        
        snac_model = LOADED_MODELS["snac_model"]
        orpheus_model = LOADED_MODELS["orpheus_model"]
        tokenizer = LOADED_MODELS["tokenizer"]
        device = LOADED_MODELS["device"]
        
        # Process the prompt for GGUF model
        prompt = f"{voice}: {text}"

        # Add special tokens manually
        start_token = tokenizer.decode([128259])  # Start of human
        end_tokens = tokenizer.decode([128009, 128260])  # End of text, End of human
        full_prompt = start_token + prompt + end_tokens

        # Generate tokens using llama-cpp-python
        output = orpheus_model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=[tokenizer.decode([128258])],  # Stop token
            echo=True  # Include prompt in output
        )

        # Extract generated text and convert to token IDs
        generated_text = output['choices'][0]['text']
        generated_ids = torch.tensor([tokenizer.encode(generated_text)], dtype=torch.int64)
        
        # Parse output and generate audio
        code_list = parse_output(generated_ids)
        audio_samples = redistribute_codes(code_list, snac_model)
        
        # Save to file
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_voice = "".join(c for c in voice if c.isalnum() or c in ("-","_"))
        out_wav = OUTPUT_DIR / f"orpheus_{safe_voice}_{ts}.wav"
        
        # Save audio file
        sf.write(str(out_wav), audio_samples, 24000)
        
        return str(out_wav), "Done!"
        
    except Exception as e:
        print(f"Error in synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error during synthesis: {e}"

# Gradio Interface
with gr.Blocks(title="Orpheus TTS – Standalone") as demo:
    gr.Markdown(
        """
        # Orpheus TTS – Standalone (GGUF)
        This UI uses Orpheus TTS with GGUF model format for efficient inference.

        **Models that will be downloaded:**
        - Orpheus TTS Model: unsloth/orpheus-3b-0.1-ft-GGUF (F16 GGUF format)
        - SNAC Audio Codec: hubertsiuzdak/snac_24khz (public)
        - Tokenizer: unsloth/orpheus-3b-0.1-ft (for text processing)

        On first run, models will be automatically downloaded (~1.5-2GB for GGUF).
        """
    )

    with gr.Row():
        text = gr.Textbox(label="Text", placeholder="Enter your text here…", lines=4)
    with gr.Row():
        voice = gr.Dropdown(VOICES, value="tara", label="Voice")
    with gr.Row():
        temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, step=0.05, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
        repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
        max_new_tokens = gr.Slider(minimum=100, maximum=4000, value=2700, step=100, label="Max New Tokens")

    run_btn = gr.Button("Convert to Speech (Generate WAV)")
    out_audio = gr.Audio(label="Result (WAV)", type="filepath")
    out_status = gr.Markdown()

    def _on_click(text, voice, temperature, top_p, repetition_penalty, max_new_tokens):
        wav_path, status = synthesize(text, voice, temperature, top_p, repetition_penalty, int(max_new_tokens))
        return wav_path, status

    run_btn.click(
        fn=_on_click,
        inputs=[text, voice, temperature, top_p, repetition_penalty, max_new_tokens],
        outputs=[out_audio, out_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False, share=False)
