"""Custom exceptions for the Traffic Tracker application."""


class DependencyError(RuntimeError):
    """
    Raised when required dependencies fail to import or initialize.
    
    This typically occurs when:
    - PyTorch is not installed or has incompatible CUDA version
    - Supervision library has binary compatibility issues
    - NumPy/SciPy version mismatches
    """
    pass


class VideoSourceError(ValueError):
    """
    Raised when a video source cannot be opened or is invalid.
    
    This can occur when:
    - File path doesn't exist
    - Video format is not supported
    - Camera index is invalid
    """
    pass


class ModelLoadError(RuntimeError):
    """
    Raised when a YOLO model fails to load.
    
    This can occur when:
    - Model file is corrupted
    - Model file doesn't exist and cannot be downloaded
    - Insufficient memory to load the model
    """
    pass

