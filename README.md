# Music Teacher

A real-time MIDI piano visualizer and teaching tool, designed for use with a MIDI keyboard and SoundFont synthesizer. This application displays a virtual piano, sheet music, and interactive overlays to help users learn and practice piano pieces from MIDI files.
This is still a work in progress and use may need some tinkering.  
Help is appreciated!

## Features

- **Real-time MIDI Input**: Connect your MIDI keyboard and see your keypresses visualized instantly.
- **SoundFont Synthesizer**: Play back notes using a configurable SoundFont for realistic piano sounds.
- **Sheet Music Rendering**: View the current MIDI file as sheet music, synchronized with your playing.
- **Teaching Mode**: Step through a song chord-by-chord, with visual feedback on correct/incorrect notes.
- **Guided Mode**: Get additional overlays and guidance for learning and practicing.
- **Live Visualization**: See the state of all keys as well as soft, sostenuto, and sustain pedals.
- **Looping & Seeking**: Set loop points, seek through the song, and practice specific sections.
- **Progress Bar**: Clickable progress bar for navigation and loop setting.
- **Keyboard Shortcuts**: Control modes, looping, and navigation with keyboard shortcuts.
- **Modern Design**: Good looking UI with smooth animations.

### Guided Mode Features
- **Score Feedback**: See the score of the current section as you play.
- **Detailed Tips**: See detailed tips what to change to improve your score based on a smart analysis of how you played the section.
- **Auto-Advance**: Automatically advance to the next section when the score reaches 95% or more.
- **Play**: Play the current measure or section to see how it sounds.
- **Section Navigation**: Navigate to the next or previous section with keyboard shortcuts.

## Planned Features (may not all be implemented)
- **Co-Pianist Mode**: Let it play the other hand or two other hands for you to practice or play songs that require more than two hands.
- **Settings UI**: Customize the constants from a settings UI.
- **Hand placement hints**: Add an algorithm to calculate the optimal hand placement for comfort and ease of switching between cords.
- **Improved Sheet Music Rendering**: Remove duplicate clefs and make sure the height stays consistent.
- **More Tips**: Add more tips for improving your score, including techniques for how to play certain things better (e.g. how to play fast consecutive notes)
- **Rhythm Game Mode**: Let the notes scroll down as bars and let the user play them as they appear.


## Installation

### Requirements
- Python 3.13+
- A SoundFont file (e.g., GeneralUser-GS.sf2)
- A MIDI file to learn with two tracks (right hand and left hand seperated)
- A MIDI output-capable keyboard

### Install with uv
```bash
# Clone the repository
git clone https://github.com/yourusername/music-teacher.git
cd music-teacher
uv sync
```

## Usage

### Configuration
Before running the application, you'll need to specify the path to your SoundFont file. 

Edit `src/music_teacher/core/app.py` and update the `SOUNDFONT_PATH` variable:

```python
SOUNDFONT_PATH = "/path/to/your/soundfont.sf2"
```

### Running the Application

If installed as a package:
```bash
# Run the application
music-teacher
```

Or from the source directory:
```bash
# Run directly from source
python src/main.py
```

When prompted, enter the path to your MIDI file.

### Controls

- **S**: Toggle synth (sound) on/off
- **T**: Toggle teaching mode
- **G**: Toggle guided mode
- **D**: Advance teacher by one chord (debug)
- **L**: Toggle loop
- **,**: Set loop start
- **.**: Set loop end
- **Left/Right Arrow**: Seek backward/forward
  - **Ctrl+Arrow**: Seek by 5
  - **Shift+Arrow**: Seek by 10
- **Progress Bar**: Click to seek, Shift+Click to set loop start, Ctrl+Click to set the loop end
- **ESC**: Quit

### Guided Mode Specific Controls
- **Space**: Play the current section
- **R**: Replay the current measure
- **A**: Toggle auto-advance when reaching a 95% score
- **Enter**: Move to the next section
  - **Shift+Enter**: Move to the previous section
- **Tab**: Toggle advanced analytics popup
 
## Screenshots

Teaching mode:
<img width="1920" height="1045" alt="image" src="https://github.com/user-attachments/assets/91635c3a-69f0-4fb2-8d3e-661700a4988f" />

Guided teaching mode:
<img width="1920" height="1045" alt="image" src="https://github.com/user-attachments/assets/0860b2a6-9ed9-4ec5-bd3e-3ea55650abb3" />

Teaching mode toggled off:
<img width="1920" height="1045" alt="image" src="https://github.com/user-attachments/assets/22544b29-0f48-420c-84be-bd705e078ae9" />

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
