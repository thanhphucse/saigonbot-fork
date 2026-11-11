"""Stage and freeze SaigonBot without modifying the backend checkout."""
import argparse
import ast
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--source", type=Path, default=repo)
    parser.add_argument("--tenant", choices=["chikku", "saigon"], default="chikku")
    parser.add_argument("--config", type=Path, default=repo / ".env")
    parser.add_argument("--ollama-home", type=Path,
                        default=Path(os.environ["LOCALAPPDATA"]) / "Programs/Ollama")
    parser.add_argument("--models-home", type=Path,
                        default=Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))))
    args = parser.parse_args()
    source = args.source.resolve()
    # Unique staging directory keeps previous builds available for comparison.
    import tempfile
    (repo / "build").mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="backend-", dir=repo / "build"))
    sys.path.insert(0, str(repo / "scripts/backend"))
    from prepare_bundle import prepare
    prepare(stage, args.config, repo / "scripts/backend/backend.env.example",
            args.tenant, args.ollama_home, args.models_home)
    src = stage / "src"
    src.mkdir()
    for path in (source / "src").glob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        if "Path(__file__).parent.parent" in text:
            tree = ast.parse(text)
            insertion = 0
            for node in tree.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)) or (
                        isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                    insertion = node.end_lineno
                else:
                    break
            lines = text.splitlines(keepends=True)
            lines.insert(insertion, "\nfrom backend_paths import DATA_ROOT\n")
            text = "".join(lines).replace("Path(__file__).parent.parent", "DATA_ROOT")
        (src / path.name).write_text(text, encoding="utf-8")
    shutil.copy2(repo / "scripts/backend/backend_paths.py", src)
    shutil.copy2(repo / "scripts/backend/bundled_ollama.py", src)
    seed = stage / "seed"
    seed.mkdir()
    store = "chikku_db" if args.tenant == "chikku" else "chroma_db"
    if not (source / store / "chroma.sqlite3").is_file():
        raise SystemExit(f"Missing indexed knowledge store: {source / store}")
    shutil.copytree(source / store, seed / store)
    # Snapshot SQLite consistently (including any committed WAL transactions).
    with sqlite3.connect(f"{(source / store / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True) as db:
        with sqlite3.connect(seed / store / "chroma.sqlite3") as backup:
            db.backup(backup)
    for name in ("knowledge_base",):
        shutil.copytree(source / "data" / name, seed / "data" / name)
    prompt = "chikku_system_prompt.txt" if args.tenant == "chikku" else "system_prompt.txt"
    shutil.copy2(source / "data" / prompt, seed / "data" / prompt)
    if args.tenant == "saigon":
        (seed / "data/processed").mkdir()
        shutil.copy2(source / "data/processed/about.txt", seed / "data/processed/about.txt")
    template = (repo / "scripts/backend/backend.env.example").read_text().replace("TENANT=chikku", f"TENANT={args.tenant}")
    (stage / "packaged-tenant.txt").write_text(args.tenant)
    out = repo / "dist/backend"
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
               "--console", "--noupx", "--name", "SaigonBot-Backend", "--paths", str(src),
               "--distpath", str(out), "--workpath", str(stage / "work"),
               "--specpath", str(stage), "--add-data", f"{seed};seed",
               "--add-data", f"{stage / 'packaged-tenant.txt'};.",
               "--add-data", f"{stage / 'embedded-config.json'};.",
               "--add-data", f"{stage / 'build-info.json'};.",
               "--add-data", f"{stage / 'ollama-runtime'};ollama-runtime",
               "--add-data", f"{stage / 'ollama-models'};ollama-models",
               "--collect-all", "chromadb", "--collect-all", "onnxruntime",
               "--collect-submodules", "uvicorn", "--collect-submodules", "opentelemetry",
               "--copy-metadata", "chromadb", "--recursive-copy-metadata", "opentelemetry-api"]
    for module in ("torch", "tensorflow", "FlagEmbedding", "transformers", "sentence_transformers", "pytest"):
        command += ["--exclude-module", module]
    command.append(str(repo / "scripts/backend/launcher.py"))
    subprocess.run(command, check=True)
    (out / ".env.example").write_text(template, encoding="utf-8")
    shutil.copy2(repo / "docs/backend-windows.md", out / "README.md")
    shutil.copy2(stage / "build-info.json", out / "build-info.json")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], check=True)
    print(f"Backend executable: {out / 'SaigonBot-Backend.exe'}")


if __name__ == "__main__":
    main()
