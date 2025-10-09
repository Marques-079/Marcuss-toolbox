#!/usr/bin/env python3
"""
Live, local object detection using stronger open-source models.

Defaults to YOLOv8x (accurate), with a --model flag to try RT-DETR-L, etc.
Draws boxes on webcam frames and prints a compact object summary every second.

Examples:
  python live_detect_sota.py                          # YOLOv8x @ 640
  python live_detect_sota.py --model rtdetr-l.pt      # RT-DETR-L
  python live_detect_sota.py --imgsz 960 --conf 0.35  # higher-res + stricter conf
  python live_detect_sota.py --cam 1                  # different camera index
"""

import argparse
import time
from collections import Counter
from typing import Union

import cv2
import torch
from ultralytics import YOLO


# ---------------------------- utils ----------------------------

def pick_device_for_ultralytics() -> Union[str, int]:
    """Prefer Apple MPS, then CUDA, else CPU, in a format Ultralytics accepts."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0  # cuda:0
    return "cpu"


def auto_open_camera(start=0, stop=5) -> cv2.VideoCapture:
    """Try camera indices [start, stop) and return the first that opens."""
    for idx in range(start, stop):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[info] Opened camera index {idx}")
            return cap
        cap.release()
    raise SystemExit("[error] Could not open any camera (tried indices 0..4)")


def draw_boxes(frame, boxes, names):
    """Render YOLO boxes + labels onto the frame."""
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].to("cpu").int().tolist()
        cls = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        label = f"{names[cls]} {conf*100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 200, 60), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 6, y1), (60, 200, 60), -1)
        cv2.putText(frame, label, (x1 + 3, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=-1,
                    help="Camera index; -1 = auto-pick (default)")
    ap.add_argument("--model", type=str, default="yolov8x.pt",
                    help="Model name or path (e.g. yolov8x.pt, rtdetr-l.pt)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="Confidence threshold (higher = fewer false positives)")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference size (try 960 or 1280 for more accuracy)")
    ap.add_argument("--print_every", type=float, default=1.0,
                    help="Seconds between object-summary prints")
    args = ap.parse_args()

    device = pick_device_for_ultralytics()
    print(f"[info] Using device: {device}")
    print(f"[info] Loading model: {args.model}")
    model = YOLO(args.model).to(device)

    # Camera
    cap = cv2.VideoCapture(args.cam) if args.cam >= 0 else auto_open_camera()
    if not cap.isOpened():
        raise SystemExit(f"[error] Could not open camera index {args.cam}")

    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detection", 960, 720)

    last_print_t = 0.0
    last_summary = ""
    fps_smooth = None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No frame captured; retrying…")
            time.sleep(0.02)
            continue

        tic = time.time()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False
        )
        toc = time.time()

        res = results[0]
        boxes = res.boxes
        names = res.names

        # Draw
        draw_boxes(frame, boxes, names)

        # FPS (smoothed)
        inst_fps = 1.0 / max(1e-6, (toc - tic))
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imshow("Live Detection", frame)

        # Summary print (once per interval)
        now = time.time()
        if now - last_print_t >= args.print_every:
            labels = [names[int(b.cls[0].item())] for b in boxes]
            counts = Counter(labels)
            summary = ", ".join(f"{k}×{v}" for k, v in counts.items()) if counts else "(no objects)"
            if summary != last_summary:
                print(f"[seen] {summary}")
                last_summary = summary
            last_print_t = now

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
