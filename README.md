# Orpheus TTS (Pinokio)

Standalone Text-to-Speech using Orpheus TTS (GGUF via llama-cpp-python), SNAC decoding, and a Gradio web UI. Launcher scripts live in the repo root; application code is under `app/`.

## What it does

- Downloads and runs Orpheus multi-language GGUF models with a local Gradio interface.
- Uses a Pinokio-managed Python virtual environment (`app/env/`).

## Using in Pinokio

1. **Install** — installs dependencies (including PyTorch via `torch.js` and `llama-cpp-python`, with CUDA build when an NVIDIA GPU is detected).
2. **Start** — launches `app/app.py` on the next free port (`{{port}}`) and opens the local URL when Gradio prints it.
3. **Update** — pulls launcher/app changes with `git pull --ff-only`, then runs the same dependency setup and checks as Install. Local divergent commits must be reconciled before Update can proceed.
4. **Reset** — removes the `app/env/` and `app/models/` folders so you can reinstall cleanly.

After **Start**, choose a language and voice, enter text, and click **Convert to Speech**. The first request downloads the selected GGUF and SNAC models. Generated WAV files are saved under `app/outputs/`; Reset preserves them. SNAC uses the Hugging Face cache, which Reset also preserves.

Install shows **Start** only after dependency and import checks succeed. Existing installations made before this check was introduced need to run **Install** once. Failed installs can be retried with **Install** without deleting downloaded models or audio.

NVIDIA installation builds llama-cpp-python with CUDA and needs a working CUDA compiler/toolchain in Pinokio. CPU inference remains available on other systems. Linux AMD uses ROCm for SNAC; this launcher does not configure a ROCm llama.cpp build. Windows AMD uses CPU inference. On Apple Silicon, the GGUF model runs on Metal (llama-cpp-python's default macOS build) while SNAC decoding uses the CPU; Intel macOS uses the last compatible PyTorch release (2.2.2). Performance and available acceleration depend on the installed drivers and native libraries.

The optional environment variables `ORPHEUS_REPO` and `ORPHEUS_FILENAME` replace the English GGUF; it must use the Orpheus token protocol and voice names. `SNAC_MODEL` overrides the default `hubertsiuzdak/snac_24khz` decoder and must remain compatible with its codebooks and 24 kHz output.

## Programmatic access

The Gradio app listens on `127.0.0.1` at the port shown in the Pinokio **Open Web UI** link. Replace `PORT` below with that port. The named endpoint is `/synthesize`; its ordered inputs are text, voice, model/language, temperature, top-p, repetition penalty, and maximum new tokens. It returns audio and a status string. A failed synthesis returns no audio and an error status.

Python (install `gradio_client` in your client environment):

```python
from gradio_client import Client

client = Client("http://127.0.0.1:PORT")
audio_path, status = client.predict(
    "Hello from Orpheus!", "tara", "english", 0.6, 0.8, 1.3, 2000,
    api_name="/synthesize",
)
print(audio_path, status)
```

JavaScript (install `@gradio/client`, then run as an ES module):

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:PORT");
const result = await client.predict("/synthesize", [
  "Hello from Orpheus!", "tara", "english", 0.6, 0.8, 1.3, 2000
]);
console.log(result.data); // Audio file metadata and status
```

cURL (Bash syntax; use `curl.exe` in PowerShell and adapt line continuations):

```bash
curl -sS -X POST "http://127.0.0.1:PORT/gradio_api/call/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"data":["Hello from Orpheus!","tara","english",0.6,0.8,1.3,2000]}'

# Replace EVENT_ID with the event_id returned by the POST request.
curl -N "http://127.0.0.1:PORT/gradio_api/call/synthesize/EVENT_ID"
```

The `complete` event contains the audio file metadata (including its download URL) and status. These examples use the Gradio 5+ HTTP API. See the official [Python client](https://github.com/gradio-app/gradio/blob/main/client/python/README.md), [JavaScript client](https://github.com/gradio-app/gradio/blob/main/client/js/README.md), and [HTTP route implementation](https://github.com/gradio-app/gradio/blob/main/gradio/routes.py) for protocol details.

## Verification

Run the offline regression suites from the repository root with Python 3.10+ and Node.js 18+:

```bash
python -m unittest discover -s app/tests -v
node --test tests/launchers.test.js
```

The Python tests mock native inference libraries; they verify token handling and application behavior without downloading model weights. The JavaScript tests evaluate launcher branches and menu states without installing packages. Actual audio quality, GPU operation, and fresh installations still require testing in Pinokio on the target hardware. See [REVIEW.md](REVIEW.md) for the review findings and validation scope.

## Project layout

```
project-root/
├── app/
│   ├── app.py
│   └── requirements.txt
├── install.js, start.js, update.js, reset.js, link.js, torch.js
├── pinokio.js, pinokio.json
└── README.md
```
