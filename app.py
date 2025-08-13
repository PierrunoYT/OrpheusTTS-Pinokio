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
    from huggingface_hub import snapshot_download, login
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install snac transformers huggingface_hub soundfile")
    IMPORTS_SUCCESSFUL = False

# === Konfiguration ===
# Orpheus model path (Hugging Face model ID or local path)
DEFAULT_MODEL_PATH = os.environ.get("ORPHEUS_MODEL", "PierrunoYT/orpheus-3b-0.1-ft")
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
        
        # Load Orpheus model and tokenizer
        print("Loading Orpheus model and tokenizer...")
        orpheus_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_PATH, 
            torch_dtype=torch.bfloat16
        )
        orpheus_model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
        
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
        return None, "Bitte Text eingeben."

    try:
        # Load models if not already loaded
        load_models()
        
        snac_model = LOADED_MODELS["snac_model"]
        orpheus_model = LOADED_MODELS["orpheus_model"]
        tokenizer = LOADED_MODELS["tokenizer"]
        device = LOADED_MODELS["device"]
        
        # Process the prompt
        input_ids, attention_mask = process_prompt(text, voice, tokenizer, device)
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = orpheus_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        
        # Parse output and generate audio
        code_list = parse_output(generated_ids)
        audio_samples = redistribute_codes(code_list, snac_model)
        
        # Save to file
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_voice = "".join(c for c in voice if c.isalnum() or c in ("-","_"))
        out_wav = OUTPUT_DIR / f"orpheus_{safe_voice}_{ts}.wav"
        
        # Save audio file
        sf.write(str(out_wav), audio_samples, 24000)
        
        return str(out_wav), "Fertig!"
        
    except Exception as e:
        print(f"Error in synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Fehler bei der Synthese: {e}"

# Gradio Interface
with gr.Blocks(title="Orpheus TTS – Standalone (Direct GGUF)") as demo:
    gr.Markdown(
        """
        # Orpheus TTS – Standalone (Direct Model)
        Dieses UI nutzt Orpheus TTS direkt ohne LM Studio.

        **Modelle die heruntergeladen werden:**
        - Orpheus TTS Model: PierrunoYT/orpheus-3b-0.1-ft (public - keine Authentifizierung erforderlich)
        - SNAC Audio Codec: hubertsiuzdak/snac_24khz (public)
        
        Beim ersten Start werden die Modelle automatisch heruntergeladen (~3-6GB).
        """
    )

    with gr.Row():
        text = gr.Textbox(label="Text", placeholder="Gib hier deinen Text ein…", lines=4)
    with gr.Row():
        voice = gr.Dropdown(VOICES, value="tara", label="Stimme")
    with gr.Row():
        temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, step=0.05, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
        repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
        max_new_tokens = gr.Slider(minimum=100, maximum=4000, value=2700, step=100, label="Max New Tokens")

    run_btn = gr.Button("In Sprache umwandeln (WAV generieren)")
    out_audio = gr.Audio(label="Ergebnis (WAV)", type="filepath")
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
