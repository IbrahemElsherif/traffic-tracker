"""Configuration constants for the Traffic Tracker application."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    """Configuration for YOLO model settings."""

    # Model file path (will auto-download if not present)
    path: str = "yolov8n.pt"

    # Input image size for inference (smaller = faster)
    image_size: int = 480

    # Confidence threshold for detections
    confidence: float = 0.25

    # Maximum detections per frame
    max_detections: int = 100

    # Use FP16 (half precision) on GPU for faster inference
    use_half_precision: bool = True

    # COCO class IDs for vehicles
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])


@dataclass
class ProcessingConfig:
    """Configuration for video processing settings."""

    # Scale factor for frame processing (1.0 = full resolution)
    # Lower values increase speed but may reduce detection accuracy
    process_scale: float = 0.75

    # Target display height for UI (pixels)
    display_height: int = 540

    # Update UI metrics every N frames (higher = smoother playback)
    ui_update_interval: int = 3


@dataclass
class Config:
    """Main configuration class combining all settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    # Available YOLO models with metadata
    AVAILABLE_MODELS: Dict[str, Dict] = field(
        default_factory=lambda: {
            "yolov8n.pt": {
                "name": "YOLOv8 Nano",
                "display_name": "YOLOv8 Nano (Fastest)",
                "description": "⚡ Fastest, ~6MB, best for real-time",
                "size_mb": 6,
            },
            "yolov8s.pt": {
                "name": "YOLOv8 Small",
                "display_name": "YOLOv8 Small (Balanced)",
                "description": "⚖️ Good balance, ~22MB, recommended",
                "size_mb": 22,
            },
            "yolov8m.pt": {
                "name": "YOLOv8 Medium",
                "display_name": "YOLOv8 Medium (More Accurate)",
                "description": "🎯 More accurate, ~52MB",
                "size_mb": 52,
            },
            "yolov8l.pt": {
                "name": "YOLOv8 Large",
                "display_name": "YOLOv8 Large (High Accuracy)",
                "description": "🎯 High accuracy, ~88MB",
                "size_mb": 88,
            },
            "yolov8x.pt": {
                "name": "YOLOv8 XLarge",
                "display_name": "YOLOv8 XLarge (Best Accuracy)",
                "description": "🏆 Best accuracy, ~136MB, slowest",
                "size_mb": 136,
            },
        }
    )

    @classmethod
    def get_model_display_name(cls, model_path: str) -> str:
        """Get human-readable name for a model file."""
        config = cls()
        model_info = config.AVAILABLE_MODELS.get(model_path, {})
        return model_info.get("name", model_path)
