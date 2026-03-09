module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        message: [
          "uv pip install wheel",
          "uv pip install -r requirements.txt",
        ],
      }
    },
    // Install torch for the correct platform/GPU first
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    },
    // Install llama-cpp-python with CUDA support on NVIDIA, plain otherwise
    {
      "when": "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          CMAKE_ARGS: "-DGGML_CUDA=on",
          FORCE_CMAKE: "1"
        },
        message: "uv pip install llama-cpp-python --no-cache-dir",
      },
      "next": null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install llama-cpp-python --no-cache-dir",
      }
    },
  ]
}
