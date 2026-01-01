"""
Traffic Tracker - Real-time vehicle detection and tracking.

A computer vision application that uses YOLOv8 and ByteTrack
to detect, track, and count vehicles in traffic videos.
"""

from src.analyzer import TrafficAnalyzer
from src.config import Config, ModelConfig
from src.exceptions import DependencyError, VideoSourceError

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["TrafficAnalyzer", "Config", "ModelConfig", "DependencyError", "VideoSourceError"]

