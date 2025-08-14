# Orpheus TTS - Multi-Model Implementation (GGUF)

A standalone Text-to-Speech application supporting both English and German Orpheus TTS models with GGUF format for efficient inference, working directly with Hugging Face models without requiring LM Studio or authentication.

## Features

- 🎙️ High-quality, natural-sounding speech synthesis
- 🌍 **Multi-language support**: English, German, Italian, Spanish, and French models
- 👥 **Multiple voice options**: 
  - English: 8 voices (tara, leah, jess, leo, dan, mia, zac, zoe)
  - German: 3 specialized voices (Jana, Thomas, Max)
  - Italian/Spanish: 6 voices (Javi, Sergio, Maria, Pietro, Giulia, Carlo)
  - French: 3 voices (Pierre, Amelie, Marie)
- 🎭 **Emotion tags support** for German, Italian/Spanish, and French models
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

**Install PyTorch (Choose One):**

**For CPU-only:**
```bash
pip install torch torchvision torchaudio
```

**For GPU acceleration (recommended):**
```bash
# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install other dependencies:**
```bash
pip install numpy soundfile snac transformers huggingface_hub gradio
```

### Step 3b: Install llama-cpp-python (Choose One)

**For CPU-only (slower):**
```bash
pip install llama-cpp-python
```

**For GPU acceleration (recommended):**
```bash
# For CUDA 12.4
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/124

# For CUDA 12.1  
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/121

# For CUDA 11.8
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/118
```

**Note:** Check your CUDA version with `nvcc --version` or `nvidia-smi` to choose the right wheel.

**Required packages:**
- `torch` - PyTorch for model inference
- `numpy` - Numerical operations
- `soundfile` - Audio file I/O
- `snac` - SNAC audio codec
- `transformers` - Hugging Face transformers library
- `huggingface_hub` - Hugging Face model hub
- `gradio` - Web interface
- `llama-cpp-python` - GGUF model inference

### Step 4: Run the Application

```bash
venv\Scripts\activate
python app.py
```

**To enable GPU acceleration after installation:**
If you installed the CPU versions and want to switch to GPU:
```bash
venv\Scripts\activate

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Reinstall llama-cpp-python with CUDA support
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/124
```

The application will start and display:
```
* Running on local URL:  http://127.0.0.1:7860
```

### Step 5: Use the Web Interface

1. Open your browser and go to http://127.0.0.1:7860
2. **Select Model/Language**: Choose between "english", "german", "italian_spanish", or "french"
3. Enter text in the "Text" field (use emotion tags for German: `<laugh>`, `<sigh>`, etc.)
4. Select a voice from the dropdown (voices update based on selected model)
5. Adjust generation parameters if needed
6. Click **"Convert to Speech (Generate WAV)"**
7. Wait for generation (first time will download models when selected)
8. Listen to the generated audio

## Model Information

### English Orpheus TTS Model (GGUF)
- **Model ID:** `unsloth/orpheus-3b-0.1-ft-GGUF`
- **File:** `orpheus-3b-0.1-ft-F16.gguf`
- **Type:** Public Hugging Face GGUF model (no authentication required)
- **Quantization:** FP16 (16-bit floating point)
- **Size:** ~1.5GB (FP16 quantized)
- **License:** Apache 2.0
- **Language:** English
- **Voices:** 8 different voices available (tara, leah, jess, leo, dan, mia, zac, zoe)
- **Tokenizer:** `unsloth/orpheus-3b-0.1-ft` (for text processing)
- **Features:** High quality audio, faster loading, smaller file size

### German Orpheus TTS Model (GGUF)
- **Model ID:** `lex-au/Orpheus-3b-German-FT-Q8_0.gguf`
- **File:** `Orpheus-3b-German-FT-Q8_0.gguf`
- **Type:** Public Hugging Face GGUF model (no authentication required)
- **Quantization:** Q8_0 (8-bit quantized)
- **Size:** ~3GB (Q8_0 quantized)
- **License:** Apache 2.0
- **Language:** German
- **Voices:** 3 specialized German voices:
  - **Jana**: Female, German, clear
  - **Thomas**: Male, German, authoritative
  - **Max**: Male, German, energetic
- **Special Features:** Emotion tags support (`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`)
- **Audio Sample Rate:** 24kHz
- **Architecture:** Specialized token-to-audio sequence model (~3 billion parameters)
- **Features:** Good quality with emotion support, larger file size, higher memory usage

### Italian/Spanish Orpheus TTS Model (GGUF)
- **Model ID:** `lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf`
- **File:** `Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf`
- **Type:** Public Hugging Face GGUF model (no authentication required)
- **Quantization:** Q8_0 (8-bit quantized)
- **Size:** ~3GB (Q8_0 quantized)
- **License:** Apache 2.0
- **Languages:** Italian and Spanish
- **Voices:** 6 specialized voices:
  - **Spanish Voices:**
    - **Javi**: Male, Spanish, warm
    - **Sergio**: Male, Spanish, professional  
    - **Maria**: Female, Spanish, friendly
  - **Italian Voices:**
    - **Pietro**: Male, Italian, passionate
    - **Giulia**: Female, Italian, expressive
    - **Carlo**: Male, Italian, refined
- **Special Features:** Emotion tags support (`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`)
- **Audio Sample Rate:** 24kHz
- **Architecture:** Specialized token-to-audio sequence model (~3 billion parameters)
- **Features:** Good quality with emotion support, larger file size, higher memory usage

### French Orpheus TTS Model (GGUF)
- **Model ID:** `lex-au/Orpheus-3b-French-FT-Q8_0.gguf`
- **File:** `Orpheus-3b-French-FT-Q8_0.gguf`
- **Type:** Public Hugging Face GGUF model (no authentication required)
- **Quantization:** Q8_0 (8-bit quantized)
- **Size:** ~3GB (Q8_0 quantized)
- **License:** Apache 2.0
- **Language:** French
- **Voices:** 3 specialized French voices:
  - **Pierre**: Male, French, sophisticated
  - **Amelie**: Female, French, elegant
  - **Marie**: Female, French, spirited
- **Special Features:** Emotion tags support (`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`)
- **Audio Sample Rate:** 24kHz
- **Architecture:** Specialized token-to-audio sequence model (~3 billion parameters)
- **Features:** Good quality with emotion support, larger file size, higher memory usage

### SNAC Audio Codec
- **Model ID:** `hubertsiuzdak/snac_24khz`
- **Type:** Public Hugging Face model
- **Size:** ~80MB
- **Sample Rate:** 24kHz

## Usage Tips

### English Voice Selection
- **tara** - Best overall voice for general use (recommended)
- **leah** - Female voice, clear pronunciation
- **jess** - Female voice, expressive
- **leo** - Male voice, deep
- **dan** - Male voice, natural
- **mia** - Female voice, soft
- **zac** - Male voice, energetic
- **zoe** - Female voice, warm

### German Voice Selection
- **Jana** - Female, German, clear pronunciation (recommended for formal content)
- **Thomas** - Male, German, authoritative tone (great for announcements)
- **Max** - Male, German, energetic delivery (good for dynamic content)

### Italian/Spanish Voice Selection
- **Spanish Voices:**
  - **Javi** - Male, Spanish, warm tone (great for casual content)
  - **Sergio** - Male, Spanish, professional delivery (recommended for business)
  - **Maria** - Female, Spanish, friendly voice (good for conversational content)
- **Italian Voices:**
  - **Pietro** - Male, Italian, passionate delivery (great for expressive content)
  - **Giulia** - Female, Italian, expressive tone (recommended for dynamic content)
  - **Carlo** - Male, Italian, refined voice (perfect for formal presentations)

### French Voice Selection
- **Pierre** - Male, French, sophisticated tone (perfect for formal and elegant content)
- **Amelie** - Female, French, elegant voice (recommended for refined presentations)
- **Marie** - Female, French, spirited delivery (great for dynamic and lively content)

### Emotion Tags (German, Italian/Spanish & French)
Add expressiveness to speech by including these tags in your text:
- `<laugh>`, `<chuckle>` - For laughter sounds
- `<sigh>` - For sighing sounds  
- `<cough>`, `<sniffle>` - For subtle interruptions
- `<groan>`, `<yawn>`, `<gasp>` - For additional emotional expression

**Examples:** 
- **German:** `Jana: Hallo! <laugh> Das ist ein Test mit Emotionen. <sigh> Interessant, oder?`
- **Spanish:** `Javi: ¡Hola! <chuckle> Este es un ejemplo con emociones. <gasp> ¡Increíble!`
- **Italian:** `Pietro: Ciao! <laugh> Questo è un esempio molto interessante. <sigh> Fantastico!`
- **French:** `Pierre: Bonjour! <chuckle> Voici un exemple avec des émotions. <gasp> Magnifique!`

### Generation Parameters
- **Temperature (0.1-1.5):** Controls randomness. Lower = more consistent, Higher = more varied
- **Top-p (0.1-1.0):** Controls diversity. Lower = more focused, Higher = more creative
- **Repetition Penalty (1.0-2.0):** Prevents repetition. Higher = less repetitive
- **Max New Tokens (100-4000):** Maximum length of generated audio

### Performance Notes
- **First run:** Models will be downloaded automatically when selected:
  - English model: ~1.5GB (FP16 GGUF - higher quality, faster loading)
  - German model: ~3GB (Q8_0 GGUF - good quality with emotion tags)
  - Italian/Spanish model: ~3GB (Q8_0 GGUF - good quality with emotion tags)
  - French model: ~3GB (Q8_0 GGUF - good quality with emotion tags)
- **GPU recommended:** CUDA-enabled GPU will significantly speed up generation (especially for Q8_0 models)
- **CPU fallback:** GGUF format provides better CPU performance than standard models
- **Memory Requirements:**
  - English model (FP16): ~4GB RAM minimum, 6GB recommended
  - German/Italian/Spanish/French models (Q8_0): ~6GB RAM minimum, 8GB recommended
- **Quantization Comparison:**
  - **FP16**: Best quality, smaller size, faster loading, no emotion tags
  - **Q8_0**: Good quality, larger size, emotion tag support, higher memory usage

## File Structure

```
OrpheusTTS/
├── app.py                     # Main multi-model application
├── requirements.txt           # Python dependencies
├── venv/                      # Python virtual environment
├── outputs/                   # Generated audio files (English/German/Italian/Spanish/French)
├── models/                    # Downloaded model cache
├── README.md                  # This file
└── dione.json                 # Configuration file
```

## Troubleshooting

### GPU Not Being Used
If the app shows "Using device: cpu" instead of "Using device: cuda":

```bash
# Check if CUDA is available in PyTorch
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Check your llama-cpp-python installation
pip show llama-cpp-python

# Reinstall with CUDA support (replace 124 with your CUDA version)
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/124
```

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
set ORPHEUS_REPO=unsloth/orpheus-3b-0.1-ft-GGUF
set ORPHEUS_FILENAME=orpheus-3b-0.1-ft-F16.gguf
set TOKENIZER_REPO=unsloth/orpheus-3b-0.1-ft
set SNAC_MODEL=hubertsiuzdak/snac_24khz
```

### Custom Model Paths
Edit `app.py` to use local model paths:

```python
ORPHEUS_REPO_ID = "path/to/your/gguf/repo"
ORPHEUS_FILENAME = "your-model.gguf"
TOKENIZER_REPO_ID = "path/to/your/tokenizer/repo"
SNAC_MODEL_PATH = "path/to/your/snac/model"
```

## License

This project uses models with their respective licenses:
- Orpheus TTS: Apache 2.0 License
- SNAC Codec: Check model repository for license

## Credits

- **English Orpheus TTS GGUF:** Unsloth (https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF)
- **German Orpheus TTS GGUF:** lex-au (https://huggingface.co/lex-au/Orpheus-3b-German-FT-Q8_0.gguf)
- **Italian/Spanish Orpheus TTS GGUF:** lex-au (https://huggingface.co/lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf)
- **French Orpheus TTS GGUF:** lex-au (https://huggingface.co/lex-au/Orpheus-3b-French-FT-Q8_0.gguf)
- **Original Orpheus TTS:** Created by Canopy Labs, quantized versions available
- **SNAC Codec:** Hubert Siuzdak (https://huggingface.co/hubertsiuzdak/snac_24khz)
- **Implementation:** Based on ComfyUI-Orpheus-TTS by ShmuelRonen

## Support

If you encounter issues:
1. Check this README for troubleshooting steps
2. Ensure all dependencies are installed correctly
3. Check system requirements (RAM, disk space)

For model-specific issues, refer to the original model repositories on Hugging Face.
