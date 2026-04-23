"""
Real-world NaViLA inference script.
Captures images from a RealSense camera, samples them, sends to a remote VLM server,
and receives navigation commands in return.

Uses parallel processing: VLM inference and command execution run concurrently.

No Isaac Sim dependency.
"""

import argparse
import base64
import io
import json
import socket
import threading
import time
from collections import deque
from queue import Queue, Empty

import cv2
import numpy as np
from PIL import Image
import rospy
from geometry_msgs.msg import Twist


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Real-world NaViLA inference with RealSense camera.")
parser.add_argument("--vlm_host", type=str, default="localhost", help="VLM server hostname or IP.")
parser.add_argument("--vlm_port", type=int, default=54321, help="VLM server port.")
parser.add_argument("--query", type=str, default="Navigate to the goal.", help="Instruction text sent to VLM.")
parser.add_argument("--capture_interval", type=float, default=0.5,
                    help="Interval (seconds) between image captures.")
parser.add_argument("--max_images", type=int, default=200,
                    help="Maximum number of images to buffer before the episode ends.")
parser.add_argument("--save_images", action="store_true", default=False,
                    help="Save sampled images to disk for debugging.")
parser.add_argument("--idle_rotation_speed", type=float, default=0.1,
                    help="Angular speed (rad/s) to rotate slowly when no rotation command is active.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# RealSense camera interface  (stub — fill in with pyrealsense2 calls)
# ---------------------------------------------------------------------------

def init_camera():
    """
    Initialize the USB camera using OpenCV.

    Returns:
        cap: cv2.VideoCapture object.
    """
    import cv2
    device_id = 4  # Default device ID, change if needed

    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device_id}")

    # Set camera parameters (use 640x480, will resize to 512x512 later)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] USB camera initialized on device {device_id}. Resolution: {actual_width}x{actual_height}")

    # Warmup: discard first frames to allow auto-exposure/white-balance to stabilize
    print("[Camera] Warming up camera...")
    for _ in range(20):
        cap.read()
    print("[Camera] Warmup complete.")

    return cap


def get_frame(cap) -> Image.Image:
    """
    Capture a single RGB frame from the USB camera.

    Args:
        cap: cv2.VideoCapture object returned by init_camera().

    Returns:
        A PIL.Image in RGB mode, resized to 512x512.
    """
    import cv2
    # Flush stale frames from the internal buffer so we always get the latest frame
    for _ in range(4):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")

    # OpenCV reads in BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Resize to 512x512 to match Isaac Sim config
    img = img.resize((512, 512), Image.BILINEAR)
    return img


def stop_camera(cap):
    """Stop and release the USB camera."""
    import cv2
    if cap is not None and cap.isOpened():
        cap.release()
        print("[Camera] USB camera stopped.")


# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------

def sample_images(image_list: list, n_samples: int = 8) -> list:
    """
    Uniformly sample *n_samples* frames from *image_list*.

    If fewer than *n_samples* frames are available the list is front-padded
    with black frames (matching the original script's behaviour).

    Args:
        image_list: List of PIL.Image frames captured so far.
        n_samples:  Desired number of frames to send to the VLM.

    Returns:
        A list of exactly *n_samples* PIL.Image frames.
    """
    if len(image_list) == 0:
        raise ValueError("image_list is empty — cannot sample.")

    images = image_list.copy()

    if len(images) < n_samples:
        pad = Image.new("RGB", images[-1].size, (0, 0, 0))
        while len(images) < n_samples:
            images.insert(0, pad)

    num = len(images)
    # Pick (n_samples-1) evenly-spaced indices plus the very last frame
    indices = [int(i * (num - 1) / (n_samples - 1)) for i in range(n_samples - 1)]
    sampled = [images[i] for i in indices]
    sampled.append(images[-1])
    return sampled


# ---------------------------------------------------------------------------
# VLM communication
# ---------------------------------------------------------------------------

def encode_image(image: Image.Image) -> str:
    """JPEG-encode a PIL image and return the base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _recvall(sock, n):
    """Receive exactly n bytes from socket, return None if connection closed early."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def send_to_vlm(image_list: list, vlm_host: str, vlm_port: int, query: str) -> dict:
    """
    Sample images from *image_list*, encode them, and send to the VLM server.

    Protocol (same as navila_eval.py):
      - 8-byte big-endian length prefix, then JSON payload
      - Response: 8-byte big-endian length prefix, then JSON

    Args:
        image_list: All frames captured so far (PIL.Image list).
        vlm_host:   VLM server hostname / IP.
        vlm_port:   VLM server port.
        query:      Navigation instruction text.

    Returns:
        Parsed JSON response dict from the VLM server.
    """
    sampled = sample_images(image_list)

    encoded_images = [encode_image(img) for img in sampled]

    request_data = {
        "images": encoded_images,
        "query": query,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))

        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, "big"))
        s.sendall(data_bytes)

        size_data = _recvall(s, 8)
        if size_data is None:
            raise RuntimeError("Server closed connection before sending response size.")
        size = int.from_bytes(size_data, "big")

        response_data = _recvall(s, size)
        if response_data is None:
            raise RuntimeError("Server closed connection before sending full response.")

    response = json.loads(response_data.decode())
    return response


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def parse_vlm_response(text: str):
    """
    Parse the VLM server response text into an action and duration.

    Uses the same keyword matching logic as navila_eval.py's get_vel_command.

    Returns:
        (action: str, duration: float)
    """
    t = text.lower()
    if "turn left" in t:
        if "45" in t:
            return "turn_left", 1.5
        elif "30" in t:
            return "turn_left", 1.0
        elif "15" in t:
            return "turn_left", 0.5
        return "turn_left", 0.5
    elif "turn right" in t:
        if "45" in t:
            return "turn_right", 1.5
        elif "30" in t:
            return "turn_right", 1.0
        elif "15" in t:
            return "turn_right", 0.5
        return "turn_right", 0.5
    elif "move forward" in t or "move" in t:
        if "75" in t:
            return "move_forward", 1.5
        elif "50" in t:
            return "move_forward", 1.0
        elif "25" in t:
            return "move_forward", 0.5
        return "move_forward", 0.5
    elif "stop" in t:
        return "stop", 0.0
    else:
        return "move_forward", 0.5


def build_twist(action: str) -> Twist:
    """Return a Twist message for the given action token."""
    twist = Twist()
    if action == "move_forward":
        twist.linear.x = 0.5
    elif action == "turn_left":
        twist.angular.z = 0.52
    elif action == "turn_right":
        twist.angular.z = -0.52
    return twist

# ---------------------------------------------------------------------------
# VLM worker thread
# ---------------------------------------------------------------------------

def vlm_worker(pipeline, image_buffer, buffer_lock, response_queue, stop_event):
    """
    Worker thread that continuously captures frames and sends them to VLM.
    Puts responses into response_queue for the main thread to consume.

    Args:
        pipeline: Camera pipeline object.
        image_buffer: Shared list of captured images.
        buffer_lock: threading.Lock for thread-safe buffer access.
        response_queue: Queue to put (response, frame_bgr) tuples.
        stop_event: threading.Event to signal thread shutdown.
    """
    last_vlm_time = None

    while not stop_event.is_set():
        # 1. Capture frame
        try:
            frame = get_frame(pipeline)
        except RuntimeError as e:
            print(f"[VLM Worker] Camera error: {e}")
            continue

        # 2. Thread-safe buffer update
        with buffer_lock:
            image_buffer.append(frame)
            frame_count = len(image_buffer)
            # Check max_images limit
            if frame_count >= args.max_images:
                print("[VLM Worker] Reached max image buffer size. Signaling stop.")
                stop_event.set()
                break

        print(f"[VLM Worker] Captured frame #{frame_count}")

        # 3. Optionally save for debugging
        if args.save_images:
            ts = time.strftime("%Y%m%d-%H%M%S")
            frame.save(f"debug_frame_{ts}_{frame_count:04d}.jpg")

        # 4. Send to VLM
        try:
            with buffer_lock:
                # Copy buffer for VLM request
                buffer_copy = list(image_buffer)

            query_initiated_time = time.time()
            response = send_to_vlm(buffer_copy, args.vlm_host, args.vlm_port, args.query)

            now = time.time()
            if last_vlm_time is not None:
                print(f"[VLM Worker] Interval since last response: {now - last_vlm_time:.2f}s")
            last_vlm_time = now
            print(f"[VLM Worker] Response: {response}")

            # Convert frame for display
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            # Put response in queue (non-blocking, drop oldest if full)
            if response_queue.full():
                try:
                    response_queue.get_nowait()
                except:
                    pass
            response_queue.put((response, frame_bgr, query_initiated_time))

        except ConnectionRefusedError:
            print(f"[VLM Worker] Cannot connect to VLM server at {args.vlm_host}:{args.vlm_port}. Retrying...")
            time.sleep(1.0)
            continue
        except Exception as e:
            print(f"[VLM Worker] Error: {e}")
            continue

        # 5. Wait before next capture
        time.sleep(args.capture_interval)

    print("[VLM Worker] Thread exiting.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("navila_inference", anonymous=True)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    # Allow publisher to register before sending commands
    time.sleep(0.5)

    print(f"[INFO] Query: {args.query}")
    print(f"[INFO] VLM server: {args.vlm_host}:{args.vlm_port}")
    print(f"[INFO] Capture interval: {args.capture_interval}s")

    pipeline = init_camera()

    # Shared state
    image_buffer: list = []
    buffer_lock = threading.Lock()
    response_queue = Queue(maxsize=1)  # Only keep latest response
    stop_event = threading.Event()

    # Start VLM worker thread
    vlm_thread = threading.Thread(
        target=vlm_worker,
        args=(pipeline, image_buffer, buffer_lock, response_queue, stop_event),
        daemon=True
    )
    vlm_thread.start()
    print("[INFO] VLM worker thread started. Running in parallel mode.")

    current_action: str = None
    current_end_time = 0.0
    current_duration = 0.0
    last_rotation_end_time = 0.0

    idle_twist = Twist()
    idle_twist.angular.z = args.idle_rotation_speed

    PUBLISH_HZ = 10
    rate = rospy.Rate(PUBLISH_HZ)

    try:
        while not stop_event.is_set() and not rospy.is_shutdown():
            now = time.time()

            # 1. Check for new VLM response (non-blocking) — update immediately
            try:
                response, frame_bgr, query_initiated_time = response_queue.get_nowait()
                if query_initiated_time >= last_rotation_end_time:
                    action, duration = parse_vlm_response(response)
                    print(f"[Robot] New command: action='{action}', duration={duration:.2f}s")

                    cv2.imshow("NaViLA - Current Frame", frame_bgr)
                    cv2.waitKey(1)

                    if action == "stop":
                        print("[INFO] VLM issued stop command. Episode complete.")
                        cmd_vel_pub.publish(Twist())
                        stop_event.set()
                        break

                    current_action = action
                    current_end_time = now + duration
                    current_duration = duration
                    if action in ("turn_left", "turn_right"):
                        last_rotation_end_time = current_end_time
                else:
                    print("[Robot] Discarding stale VLM response (query initiated during rotation).")
            except Empty:
                pass

            # 2. Publish velocity based on current state
            now = time.time()
            active = current_action is not None and now < current_end_time

            if active:
                # Within active command window: run at full speed
                cmd_vel_pub.publish(build_twist(current_action))
            elif current_action in (None, "turn_left", "turn_right"):
                # No command yet, or rotation just expired: slow idle rotation
                cmd_vel_pub.publish(idle_twist)
            else:
                # Forward command expired: repeat if short, wait if long
                if current_duration < 1.0:
                    cmd_vel_pub.publish(build_twist("move_forward"))
                else:
                    print(f"[Robot] Forward command ({current_duration:.2f}s) expired, waiting for new command.")
                    cmd_vel_pub.publish(Twist())

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        stop_event.set()
    finally:
        stop_event.set()
        cmd_vel_pub.publish(Twist())
        vlm_thread.join(timeout=2.0)
        stop_camera(pipeline)
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped. Exiting.")


if __name__ == "__main__":
    main()

