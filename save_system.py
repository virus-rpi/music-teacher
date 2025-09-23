import atexit
import os
import shutil
import zipfile
import json
from pathlib import Path
import threading
import time

VERSION = "v1.0.0"

class ModuleData:
    def __init__(self, save_root: Path, data_dir: Path, index: 'SaveIndex' = None, module_name: str = None):
        self.save_root = save_root
        self.data_dir = save_root / data_dir
        self.state_path = self.data_dir / 'state.json'
        self.index = index
        self.module_name = module_name or str(data_dir)

    def __enter__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.index:
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.data_dir)
                    self.index.update_entry(abs_path, rel_path, self.module_name, os.path.getmtime(abs_path))
            self.index.save()

    def save_state(self, state_dir):
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state_dir, f, indent=2)
        if self.index:
            self.index.update_entry(str(self.state_path), 'state.json', self.module_name, os.path.getmtime(self.state_path))

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_file(self, relative_path: str, data):
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be str, not {}".format(type(relative_path).__name__))
        abs_path = os.path.join(self.data_dir, relative_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        if isinstance(data, bytes):
            with open(abs_path, 'wb') as f:
                f.write(data)
        else:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(data)
        if self.index:
            self.index.update_entry(abs_path, relative_path, self.module_name, os.path.getmtime(abs_path))

    def load_file(self, relative_path: str):
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be str, not {}".format(type(relative_path).__name__))
        abs_path = os.path.join(self.data_dir, relative_path)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                return f.read()
        return None

    def delete_file(self, relative_path: str):
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be str, not {}".format(type(relative_path).__name__))
        abs_path = os.path.join(self.data_dir, relative_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
            if self.index:
                self.index.remove_entry(abs_path)

    def file_exists(self, relative_path: str) -> bool:
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be str, not {}".format(type(relative_path).__name__))
        return os.path.exists(os.path.join(self.data_dir, relative_path))

    def get_absolute_path(self, relative_path: str = '') -> Path:
        if not relative_path:
            return self.data_dir
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be str, not {}".format(type(relative_path).__name__))
        path = self.data_dir / relative_path
        if not path.exists() and not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
        return path

class SaveIndex:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.lock = threading.Lock()
        self.entries = []
        self._load()

    def _load(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []
        else:
            self.entries = []

    def _save(self):
        with self.lock:
            os.makedirs(self.index_path.parent, exist_ok=True)
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, indent=2)

    def update_entry(self, abs_path, rel_path, module, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self.lock:
            self.entries = [e for e in self.entries if e['abs_path'] != abs_path]
            self.entries.append({
                'abs_path': abs_path,
                'rel_path': rel_path,
                'module': module,
                'timestamp': timestamp
            })

    def remove_entry(self, abs_path):
        with self.lock:
            self.entries = [e for e in self.entries if e['abs_path'] != abs_path]

    def save(self):
        self._save()

    def search(self, path=None, module=None, rel_path=None, sort_by='timestamp', ascending=False):
        results = self.entries
        if path:
            results = [e for e in results if path in e['abs_path']]
        if module:
            results = [e for e in results if e['module'] == module]
        if rel_path:
            results = [e for e in results if rel_path in e['rel_path']]
        if sort_by == 'timestamp':
            results.sort(key=lambda e: e['timestamp'], reverse=not ascending)
        elif sort_by == 'alphabet':
            results.sort(key=lambda e: e['abs_path'], reverse=not ascending)
        return results.copy()

class SaveSystem:
    def __init__(self, save_zip=Path('save.mtsf'), save_root=Path('save'), before_exit_callback=None):
        self.save_zip = save_zip
        self.save_root = save_root
        self.before_exit_callback = before_exit_callback

        self.midi_filename = Path('song.mid')
        self.guided_teacher_data_dir = Path('guided_teacher_data')
        self.sheet_music_cache_dir = Path('sheet_music_cache')

        self._guided_teacher_data = None
        self._sheet_music_cache = None
        self.index = SaveIndex(self.save_root / '.index.json')

        self.unzip_on_start()
        atexit.register(self._on_exit)

    @property
    def guided_teacher_data(self):
        if self._guided_teacher_data is None:
            self._guided_teacher_data = ModuleData(self.save_root, self.guided_teacher_data_dir, self.index, 'guided_teacher_data')
        return self._guided_teacher_data

    @property
    def sheet_music_cache(self):
        if self._sheet_music_cache is None:
            self._sheet_music_cache = ModuleData(self.save_root, self.sheet_music_cache_dir, self.index, 'sheet_music_cache')
        return self._sheet_music_cache

    def unzip_on_start(self):
        if os.path.exists(self.save_zip):
            if os.path.exists(self.save_root):
                shutil.rmtree(self.save_root)
            with zipfile.ZipFile(self.save_zip, 'r') as zip_ref:
                zip_ref.extractall(self.save_root)

    def _on_exit(self):
        if self.before_exit_callback:
            self.before_exit_callback()
        self.index.save()
        self.zip()

    def zip(self):
        if os.path.exists(self.save_root):
            with zipfile.ZipFile(self.save_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.save_root):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, self.save_root)
                        zipf.write(abs_path, rel_path)
        shutil.rmtree(self.save_root)

    def save_midi(self, midi_path):
        dest = os.path.join(self.save_root, self.midi_filename)
        if os.path.exists(dest):
            return
        os.makedirs(self.save_root, exist_ok=True)
        shutil.copy2(midi_path, dest)
        if self.index:
            self.index.update_entry(dest, str(self.midi_filename), 'midi', os.path.getmtime(dest))

    def load_midi_path(self):
        midi_path = os.path.join(self.save_root, self.midi_filename)
        return midi_path if os.path.exists(midi_path) else None

    def search_index(self, path=None, module=None, rel_path=None, sort_by='timestamp', ascending=False):
        return self.index.search(path=path, module=module, rel_path=rel_path, sort_by=sort_by, ascending=ascending)
