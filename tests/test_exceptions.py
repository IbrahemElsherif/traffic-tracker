"""Unit tests for custom exceptions."""

import pytest
from src.exceptions import DependencyError, VideoSourceError, ModelLoadError


class TestDependencyError:
    """Tests for DependencyError exception."""
    
    def test_inheritance(self):
        """Test that DependencyError inherits from RuntimeError."""
        assert issubclass(DependencyError, RuntimeError)
    
    def test_can_be_raised(self):
        """Test that DependencyError can be raised and caught."""
        with pytest.raises(DependencyError) as exc_info:
            raise DependencyError("Test error message")
        
        assert "Test error message" in str(exc_info.value)
    
    def test_with_cause(self):
        """Test that DependencyError can be raised with a cause."""
        original_error = ImportError("Module not found")
        
        with pytest.raises(DependencyError) as exc_info:
            try:
                raise original_error
            except ImportError as e:
                raise DependencyError("Failed to import module") from e
        
        assert exc_info.value.__cause__ is original_error


class TestVideoSourceError:
    """Tests for VideoSourceError exception."""
    
    def test_inheritance(self):
        """Test that VideoSourceError inherits from ValueError."""
        assert issubclass(VideoSourceError, ValueError)
    
    def test_can_be_raised(self):
        """Test that VideoSourceError can be raised and caught."""
        with pytest.raises(VideoSourceError) as exc_info:
            raise VideoSourceError("Invalid video path")
        
        assert "Invalid video path" in str(exc_info.value)
    
    def test_can_catch_as_value_error(self):
        """Test that VideoSourceError can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise VideoSourceError("Test")


class TestModelLoadError:
    """Tests for ModelLoadError exception."""
    
    def test_inheritance(self):
        """Test that ModelLoadError inherits from RuntimeError."""
        assert issubclass(ModelLoadError, RuntimeError)
    
    def test_can_be_raised(self):
        """Test that ModelLoadError can be raised and caught."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Failed to load model")
        
        assert "Failed to load model" in str(exc_info.value)


class TestExceptionMessages:
    """Tests for exception message formatting."""
    
    def test_multiline_message(self):
        """Test that exceptions handle multiline messages."""
        message = """Failed to import dependency.

This is a detailed error message
with multiple lines.

Fix:
  1. Step one
  2. Step two
"""
        error = DependencyError(message)
        assert "Step one" in str(error)
        assert "Step two" in str(error)
    
    def test_empty_message(self):
        """Test that exceptions handle empty messages."""
        error = DependencyError("")
        assert str(error) == ""
    
    def test_unicode_message(self):
        """Test that exceptions handle unicode messages."""
        message = "Error: файл не найден 🚫"
        error = VideoSourceError(message)
        assert "🚫" in str(error)

