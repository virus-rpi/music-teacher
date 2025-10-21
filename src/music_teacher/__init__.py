"""
Music Teacher - A MIDI-based piano teaching application.

This package provides tools for learning to play piano using a MIDI keyboard,
with real-time visualization, sheet music rendering, and interactive teaching modes.
"""
from importlib.metadata import PackageNotFoundError as _PkgNotFound, version
try:
    __version__ = version("music-teacher")
except _PkgNotFound:
    __version__ = "0.1.0"
