# Orpheus TTS (Pinokio)

Standalone Text-to-Speech using Orpheus TTS (GGUF via llama-cpp-python), SNAC decoding, and a Gradio web UI. Launcher scripts live in the repo root; application code is under `app/`.

## What it does

- Downloads and runs Orpheus multi-language GGUF models with a local Gradio interface.
- Uses a Pinokio-managed Python virtual environment (`env/`) at the project root.

## Using in Pinokio

1. **Install** — installs dependencies (including PyTorch via `torch.js` and `llama-cpp-python`, with CUDA build when an NVIDIA GPU is detected).
2. **Start** — launches `app/app.py` on the next free port (`{{port}}`) and opens the local URL when Gradio prints it.
3. **Update** — `git pull` and refreshes Python packages from `app/requirements.txt`.
4. **Reset** — removes the `env/` folder so you can reinstall cleanly.

## Programmatic access

The Gradio app listens on `127.0.0.1` at the port shown in the Pinokio “Open Web UI” link (or in the start script logs). Gradio exposes HTTP routes for its UI and API; see the [Gradio API docs](https://www.gradio.app/guides/getting-started-with-the-python-client) for building clients.

Example: open the app base URL in a browser after **Start** completes.

```bash
# Replace PORT with the port from the launcher / logs
curl -sS "http://127.0.0.1:PORT/"
```

For scripted use from Python or JavaScript, use the official Gradio client libraries against the same base URL and port.

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
