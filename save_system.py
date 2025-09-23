import os
import shutil
import zipfile
import json

VERSION = "v1.0.0"

class SaveSystem:
    def __init__(self, save_zip='save.mtsf', save_root='save'):
        self.save_zip = save_zip
        self.save_root = save_root
        self.midi_filename = 'song.mid'
        self.guided_teacher_data_dir = 'guided_teacher_data'
        self.guided_teacher_state_filename = 'state.json'
        self.sheet_music_cache_dir = 'sheet_music_cache'

    def unzip_on_start(self):
        if os.path.exists(self.save_zip):
            if os.path.exists(self.save_root):
                shutil.rmtree(self.save_root)
            with zipfile.ZipFile(self.save_zip, 'r') as zip_ref:
                zip_ref.extractall(self.save_root)

    def zip_on_exit(self):
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

    def save_guided_teacher_state(self, midi_teacher_index, extra_state=None):
        state = {'midi_teacher_index': midi_teacher_index}
        if extra_state:
            state.update(extra_state)
        os.makedirs(self.save_root, exist_ok=True)
        with open(os.path.join(self.save_root, self.guided_teacher_data_dir, self.guided_teacher_state_filename), 'w', encoding='utf-8') as f:
            json.dump(state, f)

    def load_guided_teacher_state(self):
        state_path = os.path.join(self.save_root, self.guided_teacher_data_dir, self.guided_teacher_state_filename)
        if os.path.exists(state_path):
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_sheet_music_cache(self, cache_dir):
        dest_dir = os.path.join(self.save_root, self.sheet_music_cache_dir)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        if os.path.exists(cache_dir):
            shutil.copytree(cache_dir, dest_dir)

    def load_sheet_music_cache(self):
        return os.path.join(self.save_root, self.sheet_music_cache_dir)

