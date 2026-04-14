#!/usr/bin/env python3
"""Quick test to verify USB camera image capture."""

import cv2
import numpy as np
from PIL import Image
import time


def test_camera(device_id: int = 4):
    """
    Test USB camera capture.

    Args:
        device_id: Camera device ID (default: 4 for /dev/video4).
    """
    # Initialize camera
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera device {device_id}")
        return

    # Set camera parameters (use 640x480, will resize to 512x512 later)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] USB camera initialized on device {device_id}. Resolution: {actual_width}x{actual_height}")

    # Warm up - skip first few frames
    print("[INFO] Warming up...")
    for _ in range(30):
        cap.read()

    # Capture and save 5 test images
    print("[INFO] Capturing test images...")
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Failed to read frame {i}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize to 512x512 to match Isaac Sim config
        img = img.resize((512, 512), Image.BILINEAR)

        filename = f"test_frame_{i:02d}.jpg"
        img.save(filename)
        print(f"[INFO] Saved {filename} - size: {img.size}")

        time.sleep(0.5)

    # Stop camera
    cap.release()
    print("[INFO] Camera stopped.")
    print("[INFO] Check the test_frame_XX.jpg files to verify image capture.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test USB camera capture.")
    parser.add_argument("--device", type=int, default=4, help="Camera device ID (e.g., 4 for /dev/video4).")
    args = parser.parse_args()
    test_camera(args.device)
