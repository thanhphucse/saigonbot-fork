"""Frozen Windows entry point; the backend source repository stays unchanged."""
import multiprocessing


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import shutil
    import sys

    parser = argparse.ArgumentParser(description="SaigonBot local API server")
    parser.add_argument("--data-dir", type=Path, help="Persistent data/config directory")
    parser.add_argument("--port", type=int, help="Override API_PORT (default 8000)")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    executable_dir = Path(sys.executable).parent
    root = (args.data_dir or Path(os.environ["LOCALAPPDATA"]) / "SaigonBot").resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SAIGONBOT_DATA_DIR"] = str(root)
    # Do not overwrite existing knowledge stores, edited prompts or configuration.
    # A failed first copy is removed so a later launch can retry safely.
    for source in (bundle / "seed").iterdir():
        target = root / source.name
        if not target.exists():
            try:
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            except BaseException:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                raise

    from dotenv import dotenv_values
    settings = json.loads((bundle / "embedded-config.json").read_text(encoding="utf-8"))
    config = root / ".env"
    if (executable_dir / ".env").is_file():
        config = executable_dir / ".env"
    if config.is_file():
        settings.update({key: value for key, value in dotenv_values(config).items() if value is not None})
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    if os.getenv("USE_RERANKING", "false").lower() == "true":
        raise SystemExit("This build excludes FlagEmbedding. Set USE_RERANKING=false.")
    tenant = os.getenv("TENANT", "chikku")
    packaged_tenant = (bundle / "packaged-tenant.txt").read_text().strip()
    if tenant != packaged_tenant:
        raise SystemExit(f"This build contains tenant {packaged_tenant!r}; rebuild to change TENANT.")
    if os.getenv("LLM_PROVIDER") != "openai":
        raise SystemExit("This package uses OpenAI generation; only the embedding model is bundled.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OpenAI API key; rebuild with a valid private .env.")
    model = json.loads((bundle / "build-info.json").read_text())["embedding_model"]
    if os.getenv("EMBEDDING_MODEL") != model:
        raise SystemExit(f"This build contains embedding model {model!r}.")
    # Existing modules call load_dotenv() themselves. Pin discovery to the data folder.
    os.chdir(root)
    from bundled_ollama import BundledOllama
    ollama = BundledOllama(bundle, root, model).start()
    try:
        import uvicorn
        from api import app
        print(f"Configuration: embedded defaults; override: {config if config.is_file() else 'none'}\nPersistent data: {root}", flush=True)
        uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"),
                    port=args.port or int(os.getenv("API_PORT", "8000")),
                    reload=False, workers=1, loop="asyncio", http="h11")
    finally:
        ollama.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
