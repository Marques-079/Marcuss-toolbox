#!/usr/bin/env python3
"""
Top-line local detector with TTA.
- Models: YOLOv8x (default), RT-DETR-L (via --model rtdetr-l.pt), or any Ultralytics .pt
- Inference at high resolution (default 960; try 1280 if VRAM allows)
- Test-Time Augmentation: original + horizontal flip merged with NMS
- Shows webcam with boxes/labels and prints a rolling object summary

Run:
  python live_detect_topline.py
  python live_detect_topline.py --model rtdetr-l.pt --imgsz 1280 --conf 0.35
  python live_detect_topline.py --cam 1 --tta 0         # disable TTA if you need more FPS
"""

import argparse
import time
from collections import Counter
from typing import List, Tuple, Union

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO


def pick_device_ultralytics() -> Union[str, int]:
    if torch.backends.mps.is_available():
        return "mps"            # Apple Silicon
    if torch.cuda.is_available():
        return 0                # cuda:0
    return "cpu"


def _boxes_from_result(res) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract boxes (xyxy), confidences, classes from an Ultralytics result (single image).
    Returns CPU tensors: boxes[N,4], scores[N], classes[N]
    """
    b = res.boxes
    if b is None or len(b) == 0:
        return (torch.zeros((0, 4), dtype=torch.float32), 
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64))
    boxes = b.xyxy.detach().to("cpu").float()
    scores = b.conf.detach().to("cpu").float()
    classes = b.cls.detach().to("cpu").to(torch.int64)
    return boxes, scores, classes


def _flip_boxes_h(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """
    Map xyxy boxes from a horizontally flipped image back to the original image.
    x' = W - x
    """
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    new_x1 = (width - x2)
    new_x2 = (width - x1)
    out = boxes.clone()
    out[:, 0] = new_x1
    out[:, 2] = new_x2
    return out


def predict_with_tta(model, frame_bgr, imgsz: int, conf: float, iou: float, device, use_tta: bool):
    """
    Run detection on original frame, plus optional horizontal flip TTA.
    Merge with NMS and return unified (boxes, scores, classes, names).
    """
    results_orig = model.predict(frame_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    res0 = results_orig[0]
    names = res0.names

    boxes0, scores0, cls0 = _boxes_from_result(res0)

    if not use_tta:
        return boxes0, scores0, cls0, names

    # Horizontal flip TTA
    frame_flip = cv2.flip(frame_bgr, 1)
    results_flip = model.predict(frame_flip, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    resf = results_flip[0]
    boxesf, scoresf, clsf = _boxes_from_result(resf)

    # Map flipped boxes back
    h, w = frame_bgr.shape[:2]
    boxesf_mapped = _flip_boxes_h(boxesf, w)

    # Concatenate and run NMS to merge duplicates
    boxes_all = torch.cat([boxes0, boxesf_mapped], dim=0) if boxesf_mapped.numel() else boxes0
    scores_all = torch.cat([scores0, scoresf], dim=0) if scoresf.numel() else scores0
    cls_all = torch.cat([cls0, clsf], dim=0) if clsf.numel() else cls0

    if boxes_all.numel() == 0:
        return boxes_all, scores_all, cls_all, names

    keep = nms(boxes_all, scores_all, iou)  # reuse iou as merge threshold
    return boxes_all[keep], scores_all[keep], cls_all[keep], names


def draw_detections(frame, boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, names):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cls = int(classes[i].item())
        conf = float(scores[i].item())
        label = f"{names[cls]} {conf*100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 200, 60), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 6, y1), (60, 200, 60), -1)
        cv2.putText(frame, label, (x1 + 3, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=-1, help="Camera index; -1 = auto-pick")
    ap.add_argument("--model", type=str, default="yolov8x.pt",
                    help="Ultralytics model path/name (e.g., yolov8x.pt, rtdetr-l.pt)")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference size (try 1280 if VRAM allows)")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS/merge IoU threshold")
    ap.add_argument("--tta", type=int, default=1, help="1=use horizontal-flip TTA, 0=off")
    ap.add_argument("--print_every", type=float, default=1.0, help="Seconds between object summary prints")
    args = ap.parse_args()

    device = pick_device_ultralytics()
    print(f"[info] device: {device}")
    print(f"[info] model:  {args.model}")
    model = YOLO(args.model).to(device)

    # open camera (auto-pick if --cam=-1)
    if args.cam >= 0:
        cap = cv2.VideoCapture(args.cam)
    else:
        cap = None
        for idx in range(0, 6):
            try_cap = cv2.VideoCapture(idx)
            if try_cap.isOpened():
                cap = try_cap
                print(f"[info] opened camera index {idx}")
                break
            try_cap.release()
        if cap is None:
            raise SystemExit("[error] Could not open any camera (tried 0..5)")

    if not cap.isOpened():
        raise SystemExit(f"[error] Could not open camera index {args.cam}")

    cv2.namedWindow("Top-line Detection (TTA)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Top-line Detection (TTA)", 1120, 840)

    last_print_t = 0.0
    last_summary = ""
    fps_smooth = None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] no frame; retrying…")
            time.sleep(0.02)
            continue

        tic = time.time()
        boxes, scores, classes, names = predict_with_tta(
            model=model,
            frame_bgr=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            use_tta=bool(args.tta),
        )
        toc = time.time()

        draw_detections(frame, boxes, scores, classes, names)

        # FPS (smoothed)
        inst_fps = 1.0 / max(1e-6, (toc - tic))
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imshow("Top-line Detection (TTA)", frame)

        # summary print
        now = time.time()
        if now - last_print_t >= args.print_every:
            labels = [names[int(c.item())] for c in classes]
            counts = Counter(labels)
            summary = ", ".join(f"{k}×{v}" for k, v in counts.items()) if counts else "(no objects)"
            if summary != last_summary:
                print(f"[seen] {summary}")
                last_summary = summary
            last_print_t = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
