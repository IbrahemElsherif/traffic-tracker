"""Unit tests for configuration module."""

import pytest
from src.config import Config, ModelConfig, ProcessingConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()
        
        assert config.path == "yolov8n.pt"
        assert config.image_size == 480
        assert config.confidence == 0.25
        assert config.max_detections == 100
        assert config.use_half_precision is True
        assert config.vehicle_classes == [2, 3, 5, 7]
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = ModelConfig(
            path="yolov8m.pt",
            image_size=640,
            confidence=0.5,
        )
        
        assert config.path == "yolov8m.pt"
        assert config.image_size == 640
        assert config.confidence == 0.5
    
    def test_vehicle_classes_are_valid_coco_ids(self):
        """Test that default vehicle classes are valid COCO IDs."""
        config = ModelConfig()
        
        # COCO vehicle class IDs
        valid_vehicle_ids = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
        for class_id in config.vehicle_classes:
            assert class_id in valid_vehicle_ids


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ProcessingConfig()
        
        assert config.process_scale == 0.75
        assert config.display_height == 540
        assert config.ui_update_interval == 3
    
    def test_process_scale_range(self):
        """Test that process_scale is within valid range."""
        config = ProcessingConfig()
        
        assert 0.0 < config.process_scale <= 1.0


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_initialization(self):
        """Test that Config initializes with default sub-configs."""
        config = Config()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.processing, ProcessingConfig)
    
    def test_available_models(self):
        """Test that available models are properly defined."""
        config = Config()
        
        expected_models = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ]
        
        for model in expected_models:
            assert model in config.AVAILABLE_MODELS
            assert "name" in config.AVAILABLE_MODELS[model]
            assert "display_name" in config.AVAILABLE_MODELS[model]
            assert "description" in config.AVAILABLE_MODELS[model]
            assert "size_mb" in config.AVAILABLE_MODELS[model]
    
    def test_get_model_display_name_known_model(self):
        """Test display name retrieval for known models."""
        assert Config.get_model_display_name("yolov8n.pt") == "YOLOv8 Nano"
        assert Config.get_model_display_name("yolov8s.pt") == "YOLOv8 Small"
        assert Config.get_model_display_name("yolov8m.pt") == "YOLOv8 Medium"
    
    def test_get_model_display_name_unknown_model(self):
        """Test display name retrieval for unknown models."""
        unknown_model = "custom_model.pt"
        assert Config.get_model_display_name(unknown_model) == unknown_model


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_model_sizes_are_ordered(self):
        """Test that model sizes increase from nano to xlarge."""
        config = Config()
        
        sizes = [
            config.AVAILABLE_MODELS["yolov8n.pt"]["size_mb"],
            config.AVAILABLE_MODELS["yolov8s.pt"]["size_mb"],
            config.AVAILABLE_MODELS["yolov8m.pt"]["size_mb"],
            config.AVAILABLE_MODELS["yolov8l.pt"]["size_mb"],
            config.AVAILABLE_MODELS["yolov8x.pt"]["size_mb"],
        ]
        
        # Each size should be larger than the previous
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1]

