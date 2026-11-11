# SaigonBot backend — self-contained Windows package

## Run on the destination PC

1. Copy `dist\backend\SaigonBot-Backend.exe` to a permanent folder on the PC.
2. Run it and keep its console open. It starts its own private Ollama server,
   verifies the embedded embedding model, then serves the API on `127.0.0.1:8000`.
3. Install and launch the Electron frontend separately. Its existing
   `API_BASE_URL=http://127.0.0.1:8000` works without rebuilding the frontend.

No Python, Ollama installation, model download, external `.env`, or administrator
rights are required on the destination PC. The single EXE includes Python, API
libraries, Ollama 0.34.1 CPU runtime, `mxbai-embed-large:latest`, the selected
knowledge index, prompts, and the real OpenAI key from the build machine's `.env`.
It is a portable application; copying the EXE is the installation step.

Supported target: Windows 10 22H2 or newer on Intel/AMD x64. ARM64, 32-bit Windows,
and older Windows versions are outside this build's support scope. CPU inference
avoids a GPU/driver installation requirement. Allow several GB of free disk space
for the EXE, temporary extraction, and persistent data; 8 GB RAM is recommended.
First startup can take longer while the large package extracts and the model loads.
**Internet access is still required for OpenAI replies. This is not an offline LLM.**

Check in a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expect `status: healthy` and `collection_count: 330` for the current Chikku index.
Send a real question from the frontend to verify generation. Close the backend
console to stop the API and its private Ollama processes.

## Rebuild (on the development machine only)

Run all build and verification commands from the repository root (`<repo-root>`).
Use **Python 3.13 x64** and **[uv](https://docs.astral.sh/uv/)**. The build machine
must have **Ollama 0.34.1** installed, `mxbai-embed-large:latest` already pulled
(`ollama pull mxbai-embed-large:latest`), a `.env` file containing
`LLM_PROVIDER=openai` and a valid `OPENAI_API_KEY`, and Internet access during the
build (used to download Ollama's `LICENSE` file from GitHub). Stop any processes
that write to the source knowledge store before building.

```powershell
cd <repo-root>
uv venv --python 3.13 build/backend-venv
uv pip install --python build/backend-venv/Scripts/python.exe -r scripts/backend/requirements-build.lock
build/backend-venv/Scripts/python.exe scripts/build-backend.py --tenant chikku --config .env
build/backend-venv/Scripts/python.exe scripts/verify-backend.py
```

The verifier uses only the EXE's embedded configuration/runtime/model and an
isolated data directory. It exercises embeddings, retrieval, OpenAI generation,
streaming, restart persistence, and child-process cleanup.
Build options: `--source`, `--tenant saigon`, `--config`, `--ollama-home`, and
`--models-home`. Model blobs are SHA-256 verified before packaging. Output is
`dist\backend\SaigonBot-Backend.exe`, `README.md`, `.env.example`, and
`build-info.json` (no key in the latter three). Reranking remains disabled;
FlagEmbedding/PyTorch and GPU acceleration are excluded.

**Treat the EXE and generated `build` directory as secrets.** The real API key is
embedded by request and can be extracted from the EXE. Do not publish them or
commit them to Git. To change the embedded key, update the private `.env` and
rebuild. An optional `.env` beside the EXE can override embedded settings without
rebuilding. `.env.example` is optional and contains no real key; if using it,
fill in its blank `OPENAI_API_KEY`. Process environment values have highest priority.

## Data and upgrades

Persistent index, prompts, logs and `conversations.db` live in
`%LOCALAPPDATA%\SaigonBot`. They survive replacing the EXE and restarting.
Use `--data-dir "D:\SaigonBotData"` for another writable directory or
`--port 8001` if the frontend URL is changed accordingly. A data-directory `.env`
is used when no adjacent `.env` exists. No real-key `.env` is written automatically.

Back up the data directory before upgrading. Embedded knowledge seeds only
missing data; it never replaces existing indexes or edited prompts. To use a new
embedded index, start with a new data directory or replace the old tenant index
while the API is stopped. Temporary Ollama/model files are extracted per launch;
there is no Ollama service installation and no dependency on a system Ollama.

For startup failures, run from PowerShell to see errors and check
`%LOCALAPPDATA%\SaigonBot\logs\ollama.log`. Verify free space, available RAM,
port 8000, Internet access and OpenAI key validity. This EXE is unsigned.
A clean-machine deployment test is still required before production rollout.

References: [Ollama Windows requirements and embedding](https://docs.ollama.com/windows),
[PyInstaller runtime](https://pyinstaller.org/en/stable/runtime-information.html).
Ollama and model licenses are included inside the embedded payload.
