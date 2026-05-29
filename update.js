module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          CMAKE_ARGS: "-DGGML_CUDA=on",
          FORCE_CMAKE: "1"
        },
        message: "uv pip install llama-cpp-python --no-cache-dir",
      },
      next: null
    },
    {
      when: "{{gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install llama-cpp-python --no-cache-dir",
      }
    },
  ]
}
