module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{exists('app/env/.installed')}}",
      method: "fs.rm",
      params: { path: "app/env/.installed" }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: { venv: "env", path: "app" }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install wheel",
          "uv pip install -r requirements.txt",
        ],
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
        message: "uv pip install llama-cpp-python --no-binary llama-cpp-python --reinstall-package llama-cpp-python --no-cache-dir",
      },
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
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          // Pinokio halts on "Error:" output, not exit codes, and uv pip check prints no such line.
          // Parenthesized: cmd.exe would otherwise parse "A || B && C" as "A || (B && C)".
          "(uv pip check || python -c \"raise SystemExit('Error: uv pip check found incompatible dependencies')\")",
          "python -c \"import app; assert app.IMPORTS_SUCCESSFUL, 'Required inference libraries failed to import'\""
        ]
      }
    },
    {
      method: "fs.write",
      params: { path: "app/env/.installed", text: "Dependencies verified\n" }
    },
  ]
}
