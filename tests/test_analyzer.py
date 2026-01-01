"""Unit tests for the TrafficAnalyzer module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.analyzer import TrafficAnalyzer, FrameStats, VideoInfo
from src.exceptions import DependencyError, VideoSourceError


class TestFrameStats:
    """Tests for FrameStats dataclass."""
    
    def test_progress_calculation(self):
        """Test that progress is calculated correctly."""
        stats = FrameStats(
            total_unique_vehicles=10,
            current_detections=5,
            current_frame=50,
            total_frames=100,
        )
        
        assert stats.progress == 0.5
    
    def test_progress_with_zero_frames(self):
        """Test that progress handles zero total frames."""
        stats = FrameStats(
            total_unique_vehicles=0,
            current_detections=0,
            current_frame=0,
            total_frames=0,
        )
        
        assert stats.progress == 0.0
    
    def test_progress_at_end(self):
        """Test progress at video end."""
        stats = FrameStats(
            total_unique_vehicles=100,
            current_detections=10,
            current_frame=1000,
            total_frames=1000,
        )
        
        assert stats.progress == 1.0


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""
    
    def test_duration_calculation(self):
        """Test that duration is calculated correctly."""
        info = VideoInfo(
            width=1920,
            height=1080,
            fps=30,
            total_frames=900,
        )
        
        assert info.duration_seconds == 30.0
    
    def test_duration_with_zero_fps(self):
        """Test that duration handles zero FPS gracefully."""
        info = VideoInfo(
            width=1920,
            height=1080,
            fps=0,
            total_frames=900,
        )
        
        assert info.duration_seconds == 0.0
    
    def test_video_properties(self):
        """Test that video properties are stored correctly."""
        info = VideoInfo(
            width=3840,
            height=2160,
            fps=60,
            total_frames=3600,
        )
        
        assert info.width == 3840
        assert info.height == 2160
        assert info.fps == 60
        assert info.total_frames == 3600


class TestTrafficAnalyzerInit:
    """Tests for TrafficAnalyzer initialization."""
    
    @patch('src.analyzer.TrafficAnalyzer._import_supervision')
    @patch('src.analyzer.TrafficAnalyzer._import_ultralytics')
    @patch('cv2.VideoCapture')
    def test_video_source_error(self, mock_capture, mock_yolo, mock_sv):
        """Test that VideoSourceError is raised for invalid video source."""
        # Setup mocks
        mock_sv.return_value = MagicMock()
        mock_yolo.return_value = MagicMock()
        
        # Mock VideoCapture to return closed capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        with pytest.raises(VideoSourceError):
            TrafficAnalyzer(
                model_path="yolov8n.pt",
                video_source="nonexistent_video.mp4",
                device="cpu",
            )


class TestTrafficAnalyzerMethods:
    """Tests for TrafficAnalyzer methods."""
    
    def test_get_model_info_known_models(self):
        """Test model info retrieval for known models."""
        # Test without instantiating (using class method indirectly)
        from src.config import Config
        
        assert Config.get_model_display_name("yolov8n.pt") == "YOLOv8 Nano"
        assert Config.get_model_display_name("yolov8s.pt") == "YOLOv8 Small"
        assert Config.get_model_display_name("yolov8m.pt") == "YOLOv8 Medium"
        assert Config.get_model_display_name("yolov8l.pt") == "YOLOv8 Large"
        assert Config.get_model_display_name("yolov8x.pt") == "YOLOv8 XLarge"
    
    def test_get_model_info_unknown_model(self):
        """Test model info retrieval for unknown models."""
        from src.config import Config
        
        result = Config.get_model_display_name("custom_model.pt")
        assert result == "custom_model.pt"


class TestFrameStatsEdgeCases:
    """Edge case tests for FrameStats."""
    
    def test_large_vehicle_count(self):
        """Test handling of large vehicle counts."""
        stats = FrameStats(
            total_unique_vehicles=10000,
            current_detections=500,
            current_frame=50000,
            total_frames=100000,
        )
        
        assert stats.total_unique_vehicles == 10000
        assert stats.progress == 0.5
    
    def test_negative_values_not_prevented(self):
        """Test that dataclass doesn't validate negative values."""
        # Note: In production, you might want to add validation
        stats = FrameStats(
            total_unique_vehicles=-1,
            current_detections=-1,
            current_frame=-1,
            total_frames=100,
        )
        
        # Dataclass allows this; validation should be added if needed
        assert stats.total_unique_vehicles == -1


class TestVideoInfoEdgeCases:
    """Edge case tests for VideoInfo."""
    
    def test_4k_video(self):
        """Test 4K video properties."""
        info = VideoInfo(
            width=3840,
            height=2160,
            fps=60,
            total_frames=36000,  # 10 minutes at 60fps
        )
        
        assert info.duration_seconds == 600.0  # 10 minutes
    
    def test_low_fps_video(self):
        """Test low FPS video (e.g., timelapse)."""
        info = VideoInfo(
            width=1920,
            height=1080,
            fps=1,
            total_frames=3600,  # 1 hour at 1fps
        )
        
        assert info.duration_seconds == 3600.0


# Integration tests (require actual dependencies)
@pytest.mark.integration
class TestTrafficAnalyzerIntegration:
    """Integration tests that require actual YOLO models."""
    
    @pytest.mark.skip(reason="Requires YOLO model and video file")
    def test_full_processing_pipeline(self):
        """Test complete video processing pipeline."""
        analyzer = TrafficAnalyzer(
            model_path="yolov8n.pt",
            video_source="test_video.mp4",
            device="cpu",
        )
        
        frame, stats = analyzer.process_frame()
        
        assert frame is not None
        assert stats is not None
        assert stats.current_frame == 1
        
        analyzer.release()

