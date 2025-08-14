# CUDA and CPU Issues with llama-cpp-python

This document explains common CUDA/CPU issues when using llama-cpp-python with Orpheus TTS and how to resolve them.

## Issue Description

You may see mixed messages about GPU usage:

```
Loading Orpheus TTS models for german...
Using device: cuda
Loading SNAC model...
Loading Orpheus GGUF model...
Using GPU with -1 layers
```

But in the verbose output, all layers are assigned to CPU:

```
load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer   1 assigned to device CPU, is_swa = 0
...
llama_kv_cache_unified: layer   0: dev = CPU
llama_kv_cache_unified: layer   1: dev = CPU
...
```

## What's Happening

The application uses two different libraries:

1. **PyTorch** (for SNAC audio codec) - Successfully using CUDA
2. **llama-cpp-python** (for Orpheus model inference) - Falling back to CPU despite claiming GPU support

## Root Causes

### 1. Incorrect llama-cpp-python Installation
- Installed CPU-only version instead of CUDA version
- CUDA wheel didn't install properly
- Wrong CUDA version match

### 2. CUDA Environment Issues
- CUDA drivers not properly installed
- CUDA runtime missing
- Environment variables not set

### 3. Hardware Limitations
- GPU not supported by llama.cpp
- Insufficient GPU memory
- Outdated GPU drivers

## Diagnostic Commands

### Check PyTorch CUDA Support
```bash
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
```

### Check llama-cpp-python CUDA Support
```bash
python -c "from llama_cpp import llama_cpp_lib; print('llama-cpp CUDA support:', hasattr(llama_cpp_lib, '_lib') and hasattr(llama_cpp_lib._lib, 'ggml_cuda_get_device_count'))"
```

### Check CUDA Installation
```bash
nvidia-smi
nvcc --version
```

### Check Installed Packages
```bash
pip show torch
pip show llama-cpp-python
```

## Solutions

### Solution 1: Reinstall llama-cpp-python with CUDA Support

```bash
# Activate virtual environment
venv\Scripts\activate

# Uninstall current version
pip uninstall llama-cpp-python -y

# Install CUDA version (replace 124 with your CUDA version)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/124 --no-cache-dir
```

**CUDA Version Guide:**
- CUDA 12.4: `https://abetlen.github.io/llama-cpp-python/whl/124`
- CUDA 12.1: `https://abetlen.github.io/llama-cpp-python/whl/121`  
- CUDA 11.8: `https://abetlen.github.io/llama-cpp-python/whl/118`

### Solution 2: Compile from Source (Advanced)

```bash
# Set environment variables
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1

# Install from source
pip install llama-cpp-python --no-cache-dir --force-reinstall --no-deps
```

### Solution 3: Install PyTorch with Correct CUDA Version

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install with matching CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Solution 4: Use CPU-Only (Fallback)

If GPU acceleration isn't working, you can still use the app with CPU:

```bash
# Install CPU-only versions
pip install torch torchvision torchaudio
pip install llama-cpp-python
```

## Performance Impact

### With GPU Acceleration
- **SNAC model**: ~2-5x faster
- **Orpheus model**: ~3-10x faster  
- **Overall**: ~5-15x faster generation

### CPU-Only Fallback
- Still functional but slower
- German model (~3GB) takes longer to load
- Generation time: 30-60 seconds vs 3-10 seconds

## Verification

After applying solutions, restart the app and look for:

### Success Indicators
```
Using device: cuda
Using GPU with -1 layers
load_tensors: layer   0 assigned to device CUDA0, is_swa = 0
llama_kv_cache_unified: layer   0: dev = CUDA0
```

### Still Using CPU
```
load_tensors: layer   0 assigned to device CPU, is_swa = 0  
llama_kv_cache_unified: layer   0: dev = CPU
```

## Common Error Messages

### "CUDA out of memory"
```bash
# Reduce GPU layers
n_gpu_layers = 20  # Instead of -1 (all layers)
```

### "No CUDA-capable device"
- Check GPU drivers: `nvidia-smi`
- Verify GPU is CUDA-compatible

### "Module not found: llama_cpp"
```bash
# Ensure virtual environment is activated
venv\Scripts\activate
pip install llama-cpp-python
```

## Hardware Requirements

### Minimum for GPU Acceleration
- **GPU**: NVIDIA with CUDA Compute Capability 6.0+
- **VRAM**: 6GB minimum, 8GB+ recommended
- **System RAM**: 8GB minimum, 16GB+ recommended
- **CUDA**: Version 11.8, 12.1, or 12.4

### CPU-Only Fallback
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB+ free space for models

## Troubleshooting Tips

1. **Always use virtual environment**
2. **Match CUDA versions** between PyTorch and llama-cpp-python
3. **Check GPU memory** with `nvidia-smi`
4. **Restart application** after package changes
5. **Use `--no-cache-dir`** when reinstalling
6. **Verify installation** with diagnostic commands

## Getting Help

If issues persist:

1. Check your specific GPU model compatibility
2. Verify CUDA drivers are up to date
3. Try different CUDA versions (11.8, 12.1, 12.4)
4. Consider using CPU-only mode as fallback
5. Check llama-cpp-python GitHub issues for similar problems

The application will work in CPU-only mode, just with slower performance.