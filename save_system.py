import atexit
import os
import shutil
import zipfile
import json
from pathlib import Path

VERSION = "v1.0.0"

class ModuleData:
    def __init__(self, save_root: Path, data_dir: Path):
        self.data_dir = save_root / data_dir
        self.state_path = self.data_dir / 'state.json'

    def __enter__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def save_state(self, state_dir):
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state_dir, f, indent=2)

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

        self.unzip_on_start()
        atexit.register(self.zip_on_exit)

    @property
    def guided_teacher_data(self):
        if self._guided_teacher_data is None:
            self._guided_teacher_data = ModuleData(self.save_root, self.guided_teacher_data_dir)
        return self._guided_teacher_data

    @property
    def sheet_music_cache(self):
        if self._sheet_music_cache is None:
            self._sheet_music_cache = ModuleData(self.save_root, self.sheet_music_cache_dir)
        return self._sheet_music_cache

    def unzip_on_start(self):
        if os.path.exists(self.save_zip):
            if os.path.exists(self.save_root):
                shutil.rmtree(self.save_root)
            with zipfile.ZipFile(self.save_zip, 'r') as zip_ref:
                zip_ref.extractall(self.save_root)

    def zip_on_exit(self):
        if self.before_exit_callback:
            self.before_exit_callback()
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

    def load_midi_path(self):
        midi_path = os.path.join(self.save_root, self.midi_filename)
        return midi_path if os.path.exists(midi_path) else None