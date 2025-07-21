import cv2
import subprocess
import threading
import time
import numpy as np
import torch
from ultralytics import YOLO
from pymavlink import mavutil
from pyzbar import pyzbar
import base64
from typing import Optional, Tuple

# === CONFIGURATION ===
input_stream_url = "rtsp://192.168.144.25:8554/main.264" # SIYI Address
output_stream_url = "rtsp://192.168.144.6:8554/live/processed_stream" # Assuming connection over telemetry
fps = 25
model_path = "yolov8n.pt"
imgsz = 320

# Detection Mode Configuration
# Options: "yolo", "qr", "both"
detection_mode = "qr"  # Change this to switch detection modes

# === MAVLink UDP Connection ===
mavlink_target_ip = "127.0.0.1"
mavlink_target_port = 14550
mav = mavutil.mavlink_connection(f'udpout:{mavlink_target_ip}:{mavlink_target_port}')

# === GLOBAL STATE ===
streaming = True
latest_frame = [None]
raw_frame = [None]
frame_lock = threading.Lock()

# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Using device: {device}")

# === QR CODE DETECTION ===
def detect_qr_codes(frame):
    """
    Detect QR codes in frame and return detection info
    Returns: (qr_detections_list, annotated_frame)
    """
    qr_detections = []
    
    try:
        qr_codes = pyzbar.decode(frame)
        
        for qr in qr_codes:
            qr_data = qr.data.decode("utf-8")
            (x, y, w, h) = qr.rect
            
            # Draw red bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red color
            
            # Add QR data text above the box
            text_y = max(y - 10, 20)  # Ensure text is visible
            cv2.putText(frame, f"QR: {qr_data}", (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Calculate center position for MAVLink
            x_center = x + w // 2
            y_center = y + h // 2
            
            # Normalize coordinates (-1 to 1)
            x_offset = (x_center - frame.shape[1] / 2) / (frame.shape[1] / 2)
            y_offset = (y_center - frame.shape[0] / 2) / (frame.shape[0] / 2)
            
            qr_detections.append({
                'data': qr_data,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'bbox': (x, y, w, h)
            })
            
            print(f"🔍 QR Code detected: {qr_data}")
            
    except Exception as e:
        print(f"❌ QR detection error: {e}")
    
    return qr_detections, frame

# === GET CAMERA RESOLUTION ===
def get_rtsp_resolution(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception("❌ Can't open RTSP stream.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"📐 Resolution: {width}x{height}")
    return width, height

# === FFMPEG STREAM OUTPUT ===
def start_ffmpeg_stream(width, height, fps):
    command = [
        'ffmpeg', '-loglevel', 'quiet', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-profile:v', 'baseline', '-pix_fmt', 'yuv420p', '-b:v', '4M',
        '-f', 'rtsp', '-rtsp_transport', 'tcp', output_stream_url
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

# === STREAM FRAME LOOP ===
def stream_loop(width, height, fps):
    ffmpeg = start_ffmpeg_stream(width, height, fps)
    interval = 1.0 / fps
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    while streaming:
        frame = latest_frame[0]
        if frame is None:
            frame = blank
        try:
            ffmpeg.stdin.write(frame.tobytes())
        except Exception as e:
            print("❌ FFmpeg error:", e)
            break
        time.sleep(interval)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("🛑 Streaming ended.")

# === FRAME GRABBER LOOP ===
def grab_frames_loop(stream_url):
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise Exception("❌ Could not open RTSP stream for frame grabbing.")
    print("🎥 Frame grabber started.")
    while streaming:
        ret, frame = cap.read()
        if ret:
            raw_frame[0] = frame
    cap.release()
    print("🎞️ Frame grabber stopped.")

# === INFERENCE + MAVLINK OUTPUT LOOP ===
def inference_loop(model_path):
    # Initialize YOLO model only if needed
    model = None
    if detection_mode in ["yolo", "both"]:
        model = YOLO(model_path)
    
    print("🧠 Inference started.")
    print(f"🔍 Detection Mode: {detection_mode}")
    last_fps_time = time.time()
    fps_counter = 0

    while streaming:
        frame = raw_frame[0]
        if frame is None:
            continue

        # Make a copy for processing
        processing_frame = frame.copy()
        yolo_detections = []
        qr_detections = []

        # YOLO Object Detection
        if detection_mode in ["yolo", "both"] and model is not None:
            results = model.predict(source=processing_frame, imgsz=imgsz, conf=0.25, verbose=False, stream=False, device=device)
            r = results[0]
            boxes = r.boxes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                x_offset = (x_center - processing_frame.shape[1] / 2) / (processing_frame.shape[1] / 2)
                y_offset = (y_center - processing_frame.shape[0] / 2) / (processing_frame.shape[0] / 2)
                yolo_detections.append((class_name, x_offset, y_offset))

            # Get annotated frame from YOLO
            annotated_frame = r.plot() if boxes else processing_frame.copy()
        else:
            annotated_frame = processing_frame.copy()

        # QR Code Detection
        if detection_mode in ["qr", "both"]:
            qr_detections, annotated_frame = detect_qr_codes(annotated_frame)

        # Send MAVLink messages
        timestamp = int(time.time())
        
        # Send YOLO detections
        if yolo_detections:
            for obj_class, x, y in yolo_detections:
                label_x = (obj_class[:6] + "_x")[:10]
                label_y = (obj_class[:6] + "_y")[:10]
                try:
                    mav.mav.named_value_float_send(timestamp, label_x.encode('ascii', errors='ignore'), float(x))
                    mav.mav.named_value_float_send(timestamp, label_y.encode('ascii', errors='ignore'), float(y))
                except Exception as e:
                    print(f"❌ MAVLink YOLO send error: {e}")
            print(f"📡 Sent {len(yolo_detections)} YOLO object(s) via MAVLink")
        
        # Send QR detections
        if qr_detections:
            for qr in qr_detections:
                try:
                    # Send QR position
                    mav.mav.named_value_float_send(timestamp, b"QR_x", float(qr['x_offset']))
                    mav.mav.named_value_float_send(timestamp, b"QR_y", float(qr['y_offset']))
                except Exception as e:
                    print(f"❌ MAVLink QR send error: {e}")
            print(f"🔍 Sent {len(qr_detections)} QR code(s) via MAVLink")
        
        if not yolo_detections and not qr_detections:
            print("📡 No objects or QR codes detected")

        # Update the frame for streaming
        latest_frame[0] = annotated_frame

        fps_counter += 1
        if time.time() - last_fps_time >= 1.0:
            print(f"⚡ Inference FPS: {fps_counter}")
            fps_counter = 0
            last_fps_time = time.time()

    print("🧠 Inference ended.")

# === MAIN ===
if __name__ == "__main__":
    try:
        print("🏁 Starting pipeline...")
        print(f"🔍 Detection Mode: {detection_mode}")
        
        width, height = get_rtsp_resolution(input_stream_url)

        latest_frame[0] = np.zeros((height, width, 3), dtype=np.uint8)
        raw_frame[0] = None

        stream_thread = threading.Thread(target=stream_loop, args=(width, height, fps))
        grab_thread = threading.Thread(target=grab_frames_loop, args=(input_stream_url,))
        infer_thread = threading.Thread(target=inference_loop, args=(model_path,))

        grab_thread.start()
        infer_thread.start()
        stream_thread.start()

        grab_thread.join()
        infer_thread.join()
        stream_thread.join()

    except Exception as e:
        print("❌ Error:", e)
    finally:
        streaming = False
        print("✅ Pipeline shut down.")
