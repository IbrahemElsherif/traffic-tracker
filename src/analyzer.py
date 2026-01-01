"""
Traffic Analyzer - Core vehicle detection and tracking module.

This module provides the TrafficAnalyzer class which uses YOLOv8 for object
detection and ByteTrack for multi-object tracking to analyze traffic videos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Set, TYPE_CHECKING

import cv2
import numpy as np

from src.config import Config
from src.exceptions import DependencyError, VideoSourceError

if TYPE_CHECKING:
    from ultralytics import YOLO
    import supervision as sv


@dataclass
class FrameStats:
    """Statistics for a processed frame."""

    total_unique_vehicles: int
    current_detections: int
    current_frame: int
    total_frames: int

    @property
    def progress(self) -> float:
        """Calculate processing progress as a percentage (0.0 to 1.0)."""
        if self.total_frames <= 0:
            return 0.0
        return self.current_frame / self.total_frames


@dataclass
class VideoInfo:
    """Metadata about the video being processed."""

    width: int
    height: int
    fps: int
    total_frames: int

    @property
    def duration_seconds(self) -> float:
        """Calculate video duration in seconds."""
        if self.fps <= 0:
            return 0.0
        return self.total_frames / self.fps


class TrafficAnalyzer:
    """
    Analyzes traffic videos for vehicle detection, tracking, and counting.

    Uses YOLOv8 for object detection and ByteTrack (via Ultralytics) for
    multi-object tracking. Supports both CPU and GPU inference.

    Attributes:
        model_path: Path to the YOLO model file.
        device: Computing device ('cpu', 'cuda', or device index).
        video_info: Metadata about the loaded video.

    Example:
        >>> analyzer = TrafficAnalyzer(
        ...     model_path="yolov8n.pt",
        ...     video_source="traffic.mp4",
        ...     device="auto"
        ... )
        >>> frame, stats = analyzer.process_frame()
        >>> print(f"Detected {stats.current_detections} vehicles")
        >>> analyzer.release()
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        video_source: str | int = 0,
        device: str = "auto",
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the traffic analyzer.

        Args:
            model_path: Path to YOLO model file (downloads automatically if needed).
            video_source: Video file path or camera index (0 for default webcam).
            device: Computing device - 'auto', 'cuda', 'cpu', or GPU index.
            config: Optional configuration object. Uses defaults if not provided.

        Raises:
            DependencyError: If required dependencies cannot be imported.
            VideoSourceError: If the video source cannot be opened.
        """
        self.model_path = model_path
        self.config = config or Config()

        # Import dependencies with helpful error messages
        self._sv = self._import_supervision()
        YOLO = self._import_ultralytics()

        # Initialize device (must be done before loading model)
        self.device = self._setup_device(device)

        # Load YOLO model and move to device
        self.model: YOLO = YOLO(model_path)
        self.model.to(self.device)
        self._verify_device_placement()

        # Initialize video capture
        self._cap = cv2.VideoCapture(video_source)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Could not open video source: {video_source!r}")

        # Extract video properties
        self.video_info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(self._cap.get(cv2.CAP_PROP_FPS)) or 30,
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        # Tracking state
        self._current_frame = 0
        self._tracked_vehicles: Set[int] = set()

        # Initialize annotators for visualization
        self._box_annotator = self._sv.BoxAnnotator(thickness=2)

    def _import_supervision(self) -> "sv":
        """Import supervision library with helpful error handling."""
        try:
            import supervision as sv

            return sv
        except Exception as e:
            raise DependencyError(
                "Failed to import 'supervision' library.\n\n"
                "This may be caused by NumPy/SciPy binary incompatibility.\n\n"
                "Fix:\n"
                "  pip install --upgrade supervision numpy scipy\n"
            ) from e

    def _import_ultralytics(self) -> type:
        """Import Ultralytics YOLO with helpful error handling."""
        try:
            from ultralytics import YOLO

            return YOLO
        except Exception as e:
            raise DependencyError(
                "Failed to import Ultralytics/PyTorch.\n\n"
                "This may be caused by a broken PyTorch installation.\n\n"
                "Fix:\n"
                "  pip install --upgrade ultralytics torch torchvision\n"
            ) from e

    def _setup_device(self, device: str) -> str | int:
        """
        Configure the computing device for inference.

        Args:
            device: Device specification ('auto', 'cuda', 'cpu', or index).

        Returns:
            Configured device (CPU string or GPU index).

        Raises:
            DependencyError: If CUDA is requested but not available.
        """
        import torch

        if device == "auto":
            return 0 if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            if not torch.cuda.is_available():
                raise DependencyError(
                    "CUDA requested but not available.\n\n"
                    "Ensure PyTorch is installed with CUDA support:\n"
                    "  pip install torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cu121\n"
                )
            return 0  # Use first GPU

        return device

    def _verify_device_placement(self) -> None:
        """Verify that the model is loaded on the expected device."""
        if self.device == "cpu":
            return

        try:
            import torch

            param_device = next(self.model.model.parameters()).device
            if param_device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(
                    self.device if isinstance(self.device, int) else 0
                )
                print(f"✓ Model loaded on GPU: {gpu_name}")
            else:
                print(f"⚠ Warning: Model is on {param_device}, expected GPU")
        except Exception as e:
            print(f"⚠ Warning: Could not verify device placement: {e}")

    def get_device_info(self) -> str:
        """
        Get human-readable device information.

        Returns:
            Device description (e.g., "GPU: NVIDIA GeForce RTX 3050").
        """
        if self.device == "cpu":
            return "CPU"

        try:
            import torch

            device_idx = self.device if isinstance(self.device, int) else 0
            return f"GPU: {torch.cuda.get_device_name(device_idx)}"
        except Exception:
            return "GPU (CUDA)"

    def get_model_info(self) -> str:
        """
        Get human-readable model name.

        Returns:
            Model name (e.g., "YOLOv8 Nano").
        """
        return Config.get_model_display_name(os.path.basename(self.model_path))

    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameStats]]:
        """
        Process the next frame from the video source.

        Performs vehicle detection, tracking, and annotation on the frame.

        Returns:
            Tuple of (annotated_frame, statistics) or (None, None) if video ended.
        """
        ret, frame = self._cap.read()
        if not ret:
            return None, None

        self._current_frame += 1
        config = self.config

        # Downscale frame for faster inference
        original_height, original_width = frame.shape[:2]
        scale = config.processing.process_scale

        if scale < 1.0:
            process_width = int(original_width * scale)
            process_height = int(original_height * scale)
            frame_processed = cv2.resize(
                frame,
                (process_width, process_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_processed = frame
            process_width, process_height = original_width, original_height

        # Run YOLO detection with tracking
        results = self.model.track(
            frame_processed,
            persist=True,
            verbose=False,
            classes=config.model.vehicle_classes,
            device=self.device,
            half=(self.device != "cpu" and config.model.use_half_precision),
            imgsz=config.model.image_size,
            conf=config.model.confidence,
            max_det=config.model.max_detections,
        )

        # Convert to Supervision format
        detections = self._sv.Detections.from_ultralytics(results[0])

        # Scale bounding boxes back to original frame size
        if scale < 1.0 and detections.xyxy is not None and len(detections.xyxy) > 0:
            scale_x = original_width / process_width
            scale_y = original_height / process_height
            detections.xyxy[:, [0, 2]] *= scale_x
            detections.xyxy[:, [1, 3]] *= scale_y

        # Update tracked vehicles set
        if detections.tracker_id is not None:
            for tracker_id in detections.tracker_id:
                self._tracked_vehicles.add(int(tracker_id))

        # Annotate frame with bounding boxes
        annotated_frame = self._box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )

        # Compile statistics
        stats = FrameStats(
            total_unique_vehicles=len(self._tracked_vehicles),
            current_detections=len(detections),
            current_frame=self._current_frame,
            total_frames=self.video_info.total_frames,
        )

        return annotated_frame, stats

    def reset(self) -> None:
        """Reset tracking state for reprocessing the video."""
        self._current_frame = 0
        self._tracked_vehicles.clear()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "TrafficAnalyzer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures resources are released."""
        self.release()
