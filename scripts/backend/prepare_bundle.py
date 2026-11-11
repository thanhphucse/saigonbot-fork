"""Collect the private configuration, portable CPU runtime, and exact model blobs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

from dotenv import dotenv_values


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(stage, config_path, template_path, tenant, ollama_home, models_home):
    # Explicit allow-list prevents unrelated credentials in .env from being distributed.
    allowed = dotenv_values(template_path)
    private = dotenv_values(config_path)
    settings = {key: private.get(key) or value or "" for key, value in allowed.items()}
    settings.update(TENANT=tenant, API_HOST="127.0.0.1", USE_RERANKING="false")
    if settings["LLM_PROVIDER"] != "openai" or not settings.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("The private config must supply LLM_PROVIDER=openai and a real OPENAI_API_KEY.")
    model = settings["EMBEDDING_MODEL"]
    # The current indexed store uses this exact embedding family/dimensionality.
    if model not in ("mxbai-embed-large", "mxbai-embed-large:latest"):
        raise SystemExit("This build supports the indexed mxbai-embed-large model only.")
    model = "mxbai-embed-large:latest"
    settings["EMBEDDING_MODEL"] = model
    runtime = stage / "ollama-runtime"
    libraries = runtime / "lib/ollama"
    libraries.mkdir(parents=True)
    shutil.copy2(ollama_home / "ollama.exe", runtime / "ollama.exe")
    # CPU variants and their DLL dependencies; no CUDA, ROCm, Vulkan or GPU drivers.
    for path in (ollama_home / "lib/ollama").iterdir():
        if path.is_file() and (path.suffix.lower() in (".dll", ".exe") or
                               any(word in path.name.upper() for word in ("LICENSE", "COPYING", "NOTICE"))):
            shutil.copy2(path, libraries / path.name)
    required = ["llama-server.exe", "libllama-server-impl.dll", "ggml-cpu-x64.dll"]
    if any(not (libraries / name).is_file() for name in required):
        raise SystemExit("Ollama CPU runtime layout is unsupported; use the tested Ollama 0.34.1 layout.")
    version = subprocess.check_output([str(runtime / "ollama.exe"), "--version"], text=True,
                                      stderr=subprocess.STDOUT).strip().split()[-1]
    if version != "0.34.1":
        raise SystemExit(f"Tested Ollama version is 0.34.1, found {version!r}.")
    with urlopen(f"https://raw.githubusercontent.com/ollama/ollama/v{version}/LICENSE", timeout=30) as response:
        (runtime / "OLLAMA_LICENSE.txt").write_bytes(response.read())
    relative = Path("manifests/registry.ollama.ai/library/mxbai-embed-large/latest")
    manifest_path = models_home / relative
    manifest = json.loads(manifest_path.read_text())
    model_root = stage / "ollama-models"
    (model_root / relative).parent.mkdir(parents=True)
    shutil.copy2(manifest_path, model_root / relative)
    (model_root / "blobs").mkdir()
    for layer in [manifest["config"], *manifest["layers"]]:
        algorithm, expected = layer["digest"].split(":")
        if algorithm != "sha256":
            raise SystemExit("Unsupported model digest algorithm")
        path = models_home / "blobs" / f"sha256-{expected}"
        if not path.is_file() or path.stat().st_size != layer["size"] or digest(path) != expected:
            raise SystemExit(f"Missing or corrupt model blob: {path.name}")
        shutil.copy2(path, model_root / "blobs" / path.name)
    # This file is generated only under ignored build/, never in tracked source/docs.
    (stage / "embedded-config.json").write_text(json.dumps(settings), encoding="utf-8")
    info = {"tenant": tenant, "ollama_version": version, "embedding_model": model,
            "model_manifest_sha256": digest(manifest_path), "inference_device": "CPU",
            "generation_provider": "openai", "embedded_api_key": True,
            "ollama_executable_sha256": digest(runtime / "ollama.exe")}
    (stage / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
