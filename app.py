"""
Traffic Analytics Dashboard - Streamlit Web Application.

A real-time traffic analysis dashboard that uses YOLOv8 for vehicle
detection and tracking. Upload a video to see live vehicle counting
and tracking visualization.

Usage:
    streamlit run app.py
    # or
    python -m streamlit run app.py
"""

import os
import tempfile
import time
from typing import Dict

import cv2
import streamlit as st

from src.analyzer import TrafficAnalyzer
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚦",
    layout="wide",
)


def get_model_options() -> Dict[str, str]:
    """Get available YOLO models for the dropdown."""
    config = Config()
    return {
        info["display_name"]: model_path
        for model_path, info in config.AVAILABLE_MODELS.items()
    }


def get_model_descriptions() -> Dict[str, str]:
    """Get model descriptions for the info panel."""
    config = Config()
    return {
        info["display_name"]: info["description"]
        for info in config.AVAILABLE_MODELS.values()
    }


def render_sidebar() -> tuple:
    """
    Render the sidebar with all configuration options.

    Returns:
        Tuple of (uploaded_file, selected_model, target_fps, device_option)
    """
    st.sidebar.header("⚙️ Settings")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV (max 200MB)",
    )

    # Model selection
    model_options = get_model_options()
    model_choice = st.sidebar.selectbox(
        "YOLO Model",
        options=list(model_options.keys()),
        index=0,
        help="Smaller models are faster but less accurate",
    )
    selected_model = model_options[model_choice]

    # Model info expander
    with st.sidebar.expander("ℹ️ Model Info", expanded=False):
        descriptions = get_model_descriptions()
        st.caption(descriptions[model_choice])
        st.caption("💡 Models auto-download on first use")

    # FPS control
    target_fps = st.sidebar.slider(
        "Processing FPS",
        min_value=1,
        max_value=30,
        value=15,
        help="Higher FPS = smoother but more resource intensive",
    )

    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        options=["auto", "cuda", "cpu"],
        index=0,
        help="'auto' uses GPU if available, otherwise CPU",
    )

    return uploaded_file, selected_model, target_fps, device_option


def render_header():
    """Render the main header and description."""
    st.title("🚦 Real-Time Traffic Analytics Dashboard")
    st.markdown("**Powered by:** YOLOv8 • ByteTrack • Streamlit • OpenCV")


def render_metrics() -> tuple:
    """
    Render the KPI metrics section.

    Returns:
        Tuple of (kpi1, kpi2) metric placeholders.
    """
    col1, col2 = st.columns(2)
    kpi1 = col1.metric(label="🚗 Total Unique Vehicles", value=0)
    kpi2 = col2.metric(label="📍 Vehicles in Frame", value=0)
    return kpi1, kpi2


def render_video_info(analyzer: TrafficAnalyzer):
    """Render video and device information in an expander."""
    video_info = analyzer.video_info
    device_info = analyzer.get_device_info()
    model_info = analyzer.get_model_info()

    with st.expander("📹 Video & System Information", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Resolution", f"{video_info.width}×{video_info.height}")
        cols[1].metric("FPS", video_info.fps)
        cols[2].metric("Duration", f"{video_info.duration_seconds:.1f}s")
        cols[3].metric("Model", model_info)
        cols[4].metric("Device", device_info)


def process_video(
    analyzer: TrafficAnalyzer,
    target_fps: int,
    kpi1,
    kpi2,
    frame_placeholder,
    progress_bar,
    status_text,
):
    """
    Main video processing loop.

    Args:
        analyzer: TrafficAnalyzer instance.
        target_fps: Target frames per second for display.
        kpi1: Metric placeholder for total vehicles.
        kpi2: Metric placeholder for current detections.
        frame_placeholder: Streamlit placeholder for video frames.
        progress_bar: Progress bar widget.
        status_text: Status text placeholder.
    """
    config = Config()
    frame_delay = 1.0 / target_fps if target_fps > 0 else 0
    last_frame_time = 0
    ui_update_counter = 0

    while not st.session_state.get("stop_processing", False):
        # Process next frame
        frame, stats = analyzer.process_frame()

        if frame is None:
            st.sidebar.success("✅ Video processing complete!")
            st.session_state.stop_processing = True
            break

        # Resize frame for display (improves performance)
        display_height = config.processing.display_height
        if frame.shape[0] > display_height:
            scale = display_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            frame_display = cv2.resize(
                frame,
                (new_width, display_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_display = frame

        # Update UI metrics periodically (reduces overhead)
        ui_update_counter += 1
        if ui_update_counter >= config.processing.ui_update_interval:
            ui_update_counter = 0

            progress_bar.progress(stats.progress)
            status_text.text(f"Frame {stats.current_frame:,} / {stats.total_frames:,}")
            kpi1.metric(
                label="🚗 Total Unique Vehicles",
                value=stats.total_unique_vehicles,
            )
            kpi2.metric(
                label="📍 Vehicles in Frame",
                value=stats.current_detections,
            )

        # Display frame (convert BGR to RGB for Streamlit)
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # FPS limiting
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
        last_frame_time = time.time()


def main():
    """Main application entry point."""
    # Initialize session state
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False

    # Render UI components
    render_header()
    uploaded_file, selected_model, target_fps, device_option = render_sidebar()
    kpi1, kpi2 = render_metrics()
    frame_placeholder = st.empty()

    if uploaded_file is None:
        st.info(
            "👆 Upload a traffic video from the sidebar to start analysis.\n\n"
            "**Supported formats:** MP4, AVI, MOV"
        )
        st.session_state.stop_processing = False
        return

    # Save uploaded file temporarily (OpenCV needs a file path)
    suffix = os.path.splitext(uploaded_file.name or "")[1] or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        temp_file.write(uploaded_file.read())
        temp_file.flush()

        # Initialize analyzer
        try:
            analyzer = TrafficAnalyzer(
                model_path=selected_model,
                video_source=temp_file.name,
                device=device_option,
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize analyzer:\n\n{e}")
            st.stop()

        # Render video info
        render_video_info(analyzer)

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Stop button
        if st.sidebar.button("⏹️ Stop Processing"):
            st.session_state.stop_processing = True

        # Process video
        process_video(
            analyzer=analyzer,
            target_fps=target_fps,
            kpi1=kpi1,
            kpi2=kpi2,
            frame_placeholder=frame_placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        # Cleanup
        analyzer.release()
        st.session_state.stop_processing = False

    finally:
        # Clean up temporary file
        try:
            temp_file.close()
            os.unlink(temp_file.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()
