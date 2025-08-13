# Orpheus TTS - Standalone Implementation

A standalone Text-to-Speech application using Orpheus TTS that works directly with Hugging Face models without requiring LM Studio or authentication.

## Features

- 🎙️ High-quality, natural-sounding speech synthesis
- 👥 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)
- 🌐 Web-based Gradio interface
- 🔧 Adjustable generation parameters (temperature, top-p, repetition penalty)
- 💾 Automatic audio file saving
- 🚀 Direct model inference without external dependencies

## Prerequisites

- Windows 10/11 (tested)
- Python 3.10-3.12
- Git
- Hugging Face account

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd OrpheusTTS
```

### Step 2: Create Python Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install torch numpy soundfile snac transformers huggingface_hub gradio
```

**Required packages:**
- `torch` - PyTorch for model inference
- `numpy` - Numerical operations
- `soundfile` - Audio file I/O
- `snac` - SNAC audio codec
- `transformers` - Hugging Face transformers library
- `huggingface_hub` - Hugging Face model hub
- `gradio` - Web interface

### Step 4: Run the Application

```bash
venv\Scripts\activate
python app_standalone.py
```

The application will start and display:
```
* Running on local URL:  http://127.0.0.1:7860
```

### Step 5: Use the Web Interface

1. Open your browser and go to http://127.0.0.1:7860
2. Enter text in the "Text" field
3. Select a voice from the dropdown
4. Adjust generation parameters if needed
5. Click **"In Sprache umwandeln (WAV generieren)"**
6. Wait for generation (first time will download models ~3-6GB)
7. Listen to the generated audio

## Model Information

### Orpheus TTS Model
- **Model ID:** `PierrunoYT/orpheus-3b-0.1-ft`
- **Type:** Public Hugging Face model (no authentication required)
- **Size:** ~6GB
- **License:** Apache 2.0
- **Voices:** 8 different voices available

### SNAC Audio Codec
- **Model ID:** `hubertsiuzdak/snac_24khz`
- **Type:** Public Hugging Face model
- **Size:** ~80MB
- **Sample Rate:** 24kHz

## Usage Tips

### Voice Selection
- **tara** - Best overall voice for general use (recommended)
- **leah** - Female voice, clear pronunciation
- **jess** - Female voice, expressive
- **leo** - Male voice, deep
- **dan** - Male voice, natural
- **mia** - Female voice, soft
- **zac** - Male voice, energetic
- **zoe** - Female voice, warm

### Generation Parameters
- **Temperature (0.1-1.5):** Controls randomness. Lower = more consistent, Higher = more varied
- **Top-p (0.1-1.0):** Controls diversity. Lower = more focused, Higher = more creative
- **Repetition Penalty (1.0-2.0):** Prevents repetition. Higher = less repetitive
- **Max New Tokens (100-4000):** Maximum length of generated audio

### Performance Notes
- **First run:** Models will be downloaded automatically (~3-6GB total)
- **GPU recommended:** CUDA-enabled GPU will significantly speed up generation
- **CPU fallback:** Works on CPU but slower generation times
- **Memory:** Requires ~8GB RAM minimum, 16GB recommended

## File Structure

```
OrpheusTTS/
├── app_standalone.py          # Main application
├── venv/                      # Python virtual environment
├── outputs/                   # Generated audio files
├── README.md                  # This file
└── orpheus-tts-local/        # Original repo (not needed for standalone)
```

## Troubleshooting

### Model Download Issues
```
Error: Connection timeout
```
**Solution:**
1. Check internet connection
2. Try again later (Hugging Face servers might be busy)
3. Use VPN if in restricted region

### Memory Issues
```
Error: CUDA out of memory
```
**Solution:**
1. Close other applications
2. Reduce max_new_tokens parameter
3. Use CPU instead of GPU

### Import Errors
```
ModuleNotFoundError: No module named 'snac'
```
**Solution:**
1. Activate virtual environment: `venv\Scripts\activate`
2. Install missing package: `pip install snac`

## Advanced Configuration

### Environment Variables
You can set these environment variables to customize default settings:

```bash
set ORPHEUS_MODEL=PierrunoYT/orpheus-3b-0.1-ft
set SNAC_MODEL=hubertsiuzdak/snac_24khz
```

### Custom Model Paths
Edit `app_standalone.py` to use local model paths:

```python
DEFAULT_MODEL_PATH = "path/to/your/local/model"
SNAC_MODEL_PATH = "path/to/your/snac/model"
```

## License

This project uses models with their respective licenses:
- Orpheus TTS: Apache 2.0 License
- SNAC Codec: Check model repository for license

## Credits

- **Orpheus TTS:** PierrunoYT (https://huggingface.co/PierrunoYT/orpheus-3b-0.1-ft)
- **SNAC Codec:** Hubert Siuzdak (https://huggingface.co/hubertsiuzdak/snac_24khz)
- **Implementation:** Based on ComfyUI-Orpheus-TTS by ShmuelRonen

## Support

If you encounter issues:
1. Check this README for troubleshooting steps
2. Ensure all dependencies are installed correctly
3. Check system requirements (RAM, disk space)

For model-specific issues, refer to the original model repositories on Hugging Face.
