"""Verify the EXE using its embedded key, isolated runtime, and existing-free data."""
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import tempfile
import time
from urllib.request import Request, build_opener, ProxyHandler


def processes():
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in "
               "@('SaigonBot-Backend.exe','ollama.exe','llama-server.exe') } | "
               "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath | ConvertTo-Json -Compress")
    output = subprocess.check_output([str(shell), '-NoProfile', '-Command', command], text=True)
    data = json.loads(output) if output.strip() else []
    return data if isinstance(data, list) else [data]


def main():
    repo = Path(__file__).resolve().parent.parent
    exe = repo / 'dist/backend/SaigonBot-Backend.exe'
    root = Path(tempfile.mkdtemp(prefix='backend smoke ', dir=repo / 'build'))
    # The EXE must not need the source .env, developer PATH or installed Ollama/models.
    environment = os.environ.copy()
    from dotenv import dotenv_values
    for key in dotenv_values(repo / 'scripts/backend/backend.env.example'):
        environment.pop(key, None)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'OLLAMA_MODELS'):
        environment.pop(key, None)
    environment['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    environment['USERPROFILE'] = str(root / 'empty user profile')
    (root / 'empty user profile').mkdir()
    environment['LOCALAPPDATA'] = str(root / 'empty local appdata')
    Path(environment['LOCALAPPDATA']).mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    opener = build_opener(ProxyHandler({}))

    def request(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        req = Request(base + path, data=payload, headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=240) as response:
            return json.load(response)

    def start(log):
        process = subprocess.Popen([str(exe), '--data-dir', str(root / 'persistent data'),
                                    '--port', str(port)], cwd=root, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Executable exited: {process.returncode}; see {log.name}')
            try:
                request('/')
                return process
            except OSError:
                time.sleep(1)
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        raise TimeoutError(f'API did not start; see {log.name}')

    def owned_tree(process):
        inventory = processes()
        owned = {process.pid}
        while True:
            updated = owned | {item['ProcessId'] for item in inventory if item['ParentProcessId'] in owned}
            if updated == owned:
                return [item for item in inventory if item['ProcessId'] in owned]
            owned = updated

    def stop(process):
        inventory = owned_tree(process)
        workers = [item for item in inventory if item['Name'] == 'SaigonBot-Backend.exe'
                   and item['ProcessId'] != process.pid]
        child_ids = {item['ProcessId'] for item in inventory if item['Name'] != 'SaigonBot-Backend.exe'}
        # Kill only the Python worker. Its job handle must close and stop Ollama/workers.
        victim = workers[0]['ProcessId'] if workers else process.pid
        subprocess.run(['taskkill', '/PID', str(victim), '/F'], capture_output=True, check=True)
        process.wait(timeout=45)
        remaining = child_ids & {item['ProcessId'] for item in processes()}
        if remaining:
            for pid in remaining:
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
            raise AssertionError('Bundled Ollama process tree survived backend exit')
        return True

    results = {}
    with (root / 'first-launch.log').open('w', encoding='utf-8') as log:
        process = start(log)
        try:
            results['tenant'] = request('/')['tenant']
            health = request('/health')
            assert health['status'] == 'healthy', health
            assert health['collection_count'] > 0, health
            results['health'] = health
            inventory = owned_tree(process)
            engines = [item for item in inventory if item['Name'] == 'ollama.exe']
            assert engines and all('_MEI' in item['ExecutablePath'] for item in engines), engines
            output = (root / 'first-launch.log').read_text(encoding='utf-8')
            match = re.search(r'Bundled Ollama ready: (http://127\.0\.0\.1:\d+)', output)
            assert match, output[-2000:]
            with opener.open(Request(match[1] + '/api/embeddings', data=json.dumps({
                    'model': 'mxbai-embed-large:latest', 'prompt': 'isolated embedding check',
                    'options': {'num_gpu': 0}}).encode(),
                    headers={'Content-Type': 'application/json'}), timeout=120) as response:
                vector = json.load(response)['embedding']
            assert len(vector) == 1024
            results['bundled_ollama'] = {'private_port': True, 'embedding_dimensions': len(vector)}
            conversation = request('/conversation/new', {})
            answer = request('/query', {'query': 'What products does Chikku Robotics offer?', 'language': 'en'})
            assert answer['retrieved_count'] > 0, answer
            assert len(answer['response']) > 20, answer
            results['embedded_key_inference'] = {'retrieved_count': answer['retrieved_count'],
                                                 'response_characters': len(answer['response'])}
            req = Request(base + '/chat/stream', data=json.dumps({
                'query': 'Tell me about Chikku Robotics products.', 'language': 'en',
                'conversation_id': conversation['conversation_id'],
            }).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(req, timeout=240) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith('data: ')]
            assert not any(event['type'] == 'error' for event in events), events
            assert any(event['type'] == 'done' and event['full_response'] for event in events), events
            assert any(event['type'] == 'token' for event in events), events
            results['streaming'] = {'events': len(events), 'done': True}
        finally:
            results['process_tree_cleanup'] = stop(process)
    with (root / 'restart.log').open('w', encoding='utf-8') as log:
        process = start(log)
        try:
            assert request('/health')['status'] == 'healthy'
            with sqlite3.connect(root / 'persistent data/conversations.db') as db:
                count = db.execute('SELECT count(*) FROM conversations WHERE session_id=?',
                                   (conversation['conversation_id'],)).fetchone()[0]
                assert count == 1
            results['conversation_survives_restart'] = True
            assert not (root / 'persistent data/.env').exists()
            results['no_key_file_written_to_data'] = True
        finally:
            stop(process)
    (root / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'Evidence: {root}')


if __name__ == '__main__':
    main()
