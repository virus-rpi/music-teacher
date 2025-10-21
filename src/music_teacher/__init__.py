"""
Music Teacher - A MIDI-based piano teaching application.

This package provides tools for learning to play piano using a MIDI keyboard,
with real-time visualization, sheet music rendering, and interactive teaching modes.
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("music-teacher")
except Exception:
    __version__ = "0.1.0"
