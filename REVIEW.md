# Code review — 2026-09-10

Reviewed all tracked source files, launcher scripts, metadata, and documentation. The checkout was initially clean, with no installed application environment, model weights, or runtime logs.

## Fixes

- Completed the Orpheus prompt prefix and retained generated token IDs directly, eliminating the unrelated tokenizer download and lossy text round trip. Context limits now use the loaded GGUF's tokenizer and context size.
- Validated SNAC codebook indices and seven-token frames before decoding. Empty/malformed generation and decoder failures now report errors instead of successful silent WAVs. Decoding runs under inference mode.
- Serialized model loading and synthesis so another call cannot close or replace an active native model. Invalid model names are checked before unloading. API sampling inputs are checked before inference.
- Anchored model downloads and WAV outputs to the application directory. Direct launches can select a free Gradio port. Added the stable `/synthesize` API endpoint.
- Unified Update and Install, restored hardware-specific PyTorch setup during updates, and required dependency/import verification before showing Start. Maintenance remains visible after environment removal.
- Forced NVIDIA llama-cpp-python source rebuilds so an installed CPU wheel cannot bypass CUDA configuration. Removed Windows `true` shell placeholders, selected CPU packages for Windows AMD, aligned ROCm packages, and selected compatible Intel macOS packages.
- Added working API usage examples and documented migration, cache/reset behavior, hardware limitations, and test commands. Raised the minimum Gradio version to 5 for the documented HTTP API.

## Validation and remaining limits

The offline suites contain 12 Python tests and 6 Node tests. Python compilation, JavaScript syntax checks, and Git whitespace checks also pass. Tests cover normal and malformed token streams, codebook boundaries, context limits, early stops, decoder errors, API input validation, installation branches, menu states, and URL capture.

Native inference dependencies and model weights were not installed during this review. A fresh Pinokio install, real WAV generation in each language, concurrent native requests, and GPU/driver compatibility on Windows, Linux, and macOS remain unverified. Dependency versions are not fully locked; the install-time checks catch import and resolver failures but do not establish compatibility with every future release. This review does not establish that the codebase is free of all defects.

## References

- [Upstream Orpheus prompt framing](https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/engine_class.py)
- [Upstream SNAC decoding](https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/decoder.py)
- [llama-cpp-python token and generation APIs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
- [PyTorch version/backend combinations](https://pytorch.org/get-started/previous-versions/)

Launcher changes were checked against the local Pinokio reference at `D:/pinokio/prototype/system/examples/mochi/`: `install.js` (shell and child-script structure), `torch.js` (platform branches), `pinokio.js` (menu state APIs), and `start.js` (server lifecycle). The local `PINOKIO.md` documents `fs.write`, `fs.rm`, and dynamic menus. The existing captured URL pattern and `local.set` using `input.event[1]` are preserved.
