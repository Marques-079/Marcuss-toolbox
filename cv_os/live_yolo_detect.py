#!/usr/bin/env python3
"""
Live YOLO object detection from webcam (local, open-source).

- Model: Ultralytics YOLOv8 (default: yolov8s.pt)
- Shows a live window with boxes + labels
- Prints unique objects seen each second
- Quit with 'q'
"""

import argparse
import time
import sys
from collections import Counter

import cv2
import torch
from ultralytics import YOLO


def pick_device_for_ultralytics() -> str | int:
    """
    Returns a device identifier compatible with Ultralytics:
    - 'mps' on Apple Silicon if available
    - 0 (cuda:0) if NVIDIA CUDA is available
    - 'cpu' otherwise
    """
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    if torch.cuda.is_available():          # NVIDIA
        return 0
    return "cpu"


def draw_boxes(frame, boxes, names):
    """Draw YOLO boxes on the frame."""
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].to("cpu").int().tolist()
        cls = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        label = f"{names[cls]} {conf*100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 180, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="YOLO model name or path (e.g. yolov8n.pt, yolov8s.pt, yolov8l.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--print_every", type=float, default=1.0,
                    help="Seconds between printing the set of objects seen")
    args = ap.parse_args()

    device = pick_device_for_ultralytics()
    print(f"[info] Using device: {device}")

    try:
        model = YOLO(args.model)
        # Move model to device (Ultralytics accepts str/int devices)
        model.to(device)
    except Exception as e:
        print(f"[error] Failed to load model '{args.model}': {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[error] Could not open camera index {args.cam}")
        sys.exit(2)

    cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live", 960, 720)

    last_print_t = 0.0
    last_seen_summary = ""
    fps_smooth = None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No frame captured; retrying…")
            time.sleep(0.02)
            continue

        tic = time.time()
        # Run inference (single-image). verbose=False to keep stdout clean.
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False
        )
        toc = time.time()

        res = results[0]  # batch of 1
        boxes = res.boxes  # Boxes object
        names = res.names  # dict of class_id -> name

        # Overlay detections
        draw_boxes(frame, boxes, names)

        # FPS (simple exponential smoothing)
        inst_fps = 1.0 / max(1e-6, (toc - tic))
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        # Show window
        cv2.imshow("YOLO Live", frame)

        # Build and print a compact per-frame summary once per interval
        now = time.time()
        if now - last_print_t >= args.print_every:
            labels = [names[int(b.cls[0].item())] for b in boxes]
            counts = Counter(labels)
            if counts:
                summary = ", ".join(f"{k}×{v}" for k, v in counts.items())
            else:
                summary = "(no objects)"

            if summary != last_seen_summary:
                print(f"[seen] {summary}")
                last_seen_summary = summary
            last_print_t = now

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
