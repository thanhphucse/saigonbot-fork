"""Own a private Ollama process tree, independent of installed Ollama instances."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class BundledOllama:
    def __init__(self, bundle: Path, data_root: Path, model: str):
        self.bundle = bundle
        self.data_root = data_root
        self.model = model
        self.process = None
        self.job = None
        self.log = None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                       ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
        self.opener = build_opener(ProxyHandler({}))

    def request(self, route, data=None, timeout=5):
        request = Request(self.url + route, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def start(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.job = self.kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        runtime = self.bundle / "ollama-runtime"
        environment = os.environ.copy()
        environment.update(OLLAMA_HOST=self.url, OLLAMA_MODELS=str(self.bundle / "ollama-models"),
                           OLLAMA_NO_CLOUD="1", OLLAMA_NUM_GPU="0", OLLAMA_VULKAN="0",
                           CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1",
                           GGML_VK_VISIBLE_DEVICES="-1", OLLAMA_KEEP_ALIVE="5m")
        # Deliberately exclude developer PATH and third-party GPU/runtime directories.
        environment["PATH"] = str(runtime / "lib/ollama") + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32")
        logs = self.data_root / "logs"
        logs.mkdir(exist_ok=True)
        self.log = (logs / "ollama.log").open("a", encoding="utf-8")
        try:
            # PyInstaller's DLL search directory must not affect an external executable.
            self.kernel.SetDllDirectoryW(None)
            try:
                self.process = subprocess.Popen([str(runtime / "ollama.exe"), "serve"],
                                                cwd=runtime, env=environment, stdout=self.log,
                                                stderr=subprocess.STDOUT,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                if getattr(sys, "frozen", False):
                    self.kernel.SetDllDirectoryW(str(self.bundle))
            if not self.kernel.AssignProcessToJobObject(self.job, wintypes.HANDLE(self.process._handle)):
                self.process.kill()
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Bundled Ollama exited; see {logs / 'ollama.log'}")
                try:
                    models = self.request("/api/tags")["models"]
                    if not any(item["name"] == self.model for item in models):
                        raise RuntimeError(f"Embedded model {self.model!r} is missing")
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"Bundled Ollama did not start; see {logs / 'ollama.log'}")
            result = self.request("/api/embeddings", {"model": self.model, "prompt": "startup check",
                                                    "options": {"num_gpu": 0}}, timeout=180)
            if len(result.get("embedding", [])) != 1024:
                raise RuntimeError("Bundled embedding model did not return a 1024-dimensional vector")
            # Set before importing the Ollama Python SDK, which creates its default client at import.
            os.environ["OLLAMA_HOST"] = self.url
            os.environ["OLLAMA_NUM_GPU"] = "0"
            for key in ("NO_PROXY", "no_proxy"):
                os.environ[key] = os.environ.get(key, "") + ",127.0.0.1,localhost"
            print(f"Bundled Ollama ready: {self.url}; model={self.model}; CPU; embedding verified", flush=True)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)  # also stops every inference worker
            self.job = None
        if self.process:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
