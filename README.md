# 🚦 Traffic Analytics Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time traffic analysis dashboard that uses **YOLOv8** for vehicle detection and **ByteTrack** for multi-object tracking. Upload any traffic video and get instant vehicle counting with live visualization.

![Traffic Analytics Demo](docs/demo.gif)

## ✨ Features

- **🎯 Real-time Vehicle Detection** — Powered by YOLOv8 (Nano to XLarge models)
- **🔄 Multi-Object Tracking** — ByteTrack algorithm for consistent vehicle IDs
- **📊 Live Analytics** — Real-time vehicle counting and statistics
- **🎛️ Configurable Settings** — Adjust FPS, model size, and processing device
- **⚡ GPU Acceleration** — CUDA support for faster processing
- **🖥️ Modern Web UI** — Clean Streamlit dashboard

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-tracker.git
   cd traffic-tracker
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **CPU only:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **With GPU support (CUDA 12.1):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser** — Navigate to `http://localhost:8501`

## 📖 Usage

1. **Upload a Video** — Use the sidebar to upload a traffic video (MP4, AVI, MOV)
2. **Select Model** — Choose from YOLOv8 Nano (fastest) to XLarge (most accurate)
3. **Configure Settings** — Adjust processing FPS and device (CPU/GPU)
4. **Start Analysis** — Watch real-time detection and tracking
5. **View Results** — See live vehicle count and tracking visualization

## 🏗️ Project Structure

```
traffic-tracker/
├── app.py                 # Streamlit web application
├── src/
│   ├── __init__.py       # Package initialization
│   ├── analyzer.py       # Core TrafficAnalyzer class
│   ├── config.py         # Configuration dataclasses
│   └── exceptions.py     # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py  # Analyzer unit tests
│   ├── test_config.py    # Config unit tests
│   └── test_exceptions.py # Exception tests
├── docs/
│   └── demo.gif          # Demo animation
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── pyproject.toml        # Project metadata
├── LICENSE               # MIT License
└── README.md             # This file
```

## 🔧 Configuration

### Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ | Real-time, edge devices |
| YOLOv8s | 22 MB | ⚡⚡⚡⚡ | ★★★☆☆ | Balanced (recommended) |
| YOLOv8m | 52 MB | ⚡⚡⚡ | ★★★★☆ | Higher accuracy |
| YOLOv8l | 88 MB | ⚡⚡ | ★★★★☆ | High accuracy |
| YOLOv8x | 136 MB | ⚡ | ★★★★★ | Maximum accuracy |

### Processing Settings

- **Processing FPS**: 1-30 (default: 15)
- **Device**: `auto`, `cuda`, or `cpu`
- **Process Scale**: 0.75 (75% resolution for speed)

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py -v
```

## 📈 Performance Tips

1. **Use GPU** — CUDA provides 5-10x speedup over CPU
2. **Choose the right model** — YOLOv8n for speed, YOLOv8s for balance
3. **Lower FPS** — Reduce to 10-15 FPS for smoother playback
4. **Smaller videos** — 720p processes faster than 4K

## 🛠️ Tech Stack

- **[YOLOv8](https://ultralytics.com/)** — State-of-the-art object detection
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** — Multi-object tracking (via Ultralytics)
- **[Supervision](https://supervision.roboflow.com/)** — Computer vision utilities
- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[OpenCV](https://opencv.org/)** — Video processing
- **[PyTorch](https://pytorch.org/)** — Deep learning backend


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
