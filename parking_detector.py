"""
parking_detector.py
===================
背景執行緒：讀 HLS 串流 → 裁 ROI → PPNet 推論 → 10 秒違停計時 → 觸發回呼。

回呼簽名：on_violation(frame: np.ndarray, probs: np.ndarray, cam: str, duration: float)
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class ParkingDetector:
    def __init__(self, stream_reader, model, device, cam: str,
                 masks_dir, violation_seconds: float = 10.0,
                 cooldown_seconds: float = 60.0, inference_interval: float = 0.5,
                 on_violation=None):
        self._reader = stream_reader
        self._model = model
        self._device = device
        self._cam = cam
        self._violation_seconds = violation_seconds
        self._cooldown_seconds = cooldown_seconds
        self._inference_interval = inference_interval
        self._on_violation = on_violation

        self._poly, self._bbox, self._src_size = self._load_roi(cam, masks_dir)

        # State (protected by _lock)
        self._lock = threading.Lock()
        self._pred = 0
        self._probs = None
        self._latest_frame = None
        self._latest_crop = None
        self._car_since = None
        self._last_alert_t = None
        self._total_alerts = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def get_state(self) -> dict:
        with self._lock:
            now = time.time()
            car_dur = (now - self._car_since) if self._car_since else 0.0
            cooldown_left = 0.0
            if self._last_alert_t:
                cooldown_left = max(0.0, self._last_alert_t + self._cooldown_seconds - now)
            return {
                "pred": self._pred,
                "pred_label": "CAR" if self._pred == 1 else "NOCAR",
                "probs": self._probs.tolist() if self._probs is not None else None,
                "car_duration": round(car_dur, 1),
                "violation_seconds": self._violation_seconds,
                "cooldown_remaining": round(cooldown_left, 1),
                "total_alerts": self._total_alerts,
                "stream_status": self._reader.status,
                "stream_resolution": self._reader.resolution,
            }

    def get_annotated_frame(self):
        with self._lock:
            frame = self._latest_frame
            if frame is None:
                return None

            vis = frame.copy()
            pred = self._pred
            car_since = self._car_since
            now = time.time()

            # ROI fill + outline
            ov = vis.copy()
            fill_color = (0, 0, 220) if pred == 1 else (0, 200, 80)
            cv2.fillPoly(ov, [self._poly], fill_color)
            vis = cv2.addWeighted(ov, 0.25 if pred == 1 else 0.15, vis, 0.75 if pred == 1 else 0.85, 0)
            outline_color = (0, 0, 255) if pred == 1 else (0, 255, 100)
            cv2.polylines(vis, [self._poly], True, outline_color, 3)

            # Status label
            label = "CAR DETECTED" if pred == 1 else "CLEAR"
            label_color = (0, 0, 255) if pred == 1 else (0, 200, 80)
            cv2.rectangle(vis, (10, 10), (340, 60), (0, 0, 0), -1)
            cv2.putText(vis, label, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

            # Duration timer
            if car_since is not None:
                dur = now - car_since
                pct = min(dur / self._violation_seconds, 1.0)
                timer_color = (0, 165, 255) if dur < self._violation_seconds else (0, 0, 255)
                timer_text = f"{dur:.1f}s / {self._violation_seconds:.0f}s"
                cv2.rectangle(vis, (10, 65), (340, 105), (0, 0, 0), -1)
                cv2.putText(vis, timer_text, (18, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2)

                # Progress bar
                bar_x, bar_y, bar_w, bar_h = 10, 110, 330, 12
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                filled_w = int(bar_w * pct)
                bar_fill = (0, 165, 255) if pct < 1.0 else (0, 0, 255)
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_fill, -1)

                if dur >= self._violation_seconds:
                    cv2.rectangle(vis, (10, 125), (340, 165), (0, 0, 180), -1)
                    cv2.putText(vis, "VIOLATION!", (18, 155),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            return vis

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _load_roi(self, cam: str, masks_dir):
        f = Path(masks_dir) / f"roi_{cam}.json"
        if not f.is_file():
            raise FileNotFoundError(f"ROI 檔案不存在: {f}")
        d = json.loads(f.read_text(encoding="utf-8"))
        poly = np.array(d["polygon"], np.int32)
        xs, ys = poly[:, 0], poly[:, 1]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return poly, bbox, (int(d["width"]), int(d["height"]))

    def _run(self):
        import torch
        from torchvision.transforms import functional as TF

        IMG_SIZE = 128
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
        next_t = time.monotonic()

        while not self._stop.is_set():
            now_mono = time.monotonic()
            if now_mono < next_t:
                time.sleep(min(0.05, next_t - now_mono))
                continue
            next_t = now_mono + self._inference_interval

            frame = self._reader.get_latest()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if (w, h) != self._src_size:
                with self._lock:
                    self._latest_frame = frame
                continue

            x0, y0, x1, y1 = self._bbox
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            t = t.to(self._device, dtype=torch.float32) / 255.0
            t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
            t = (t - mean) / std

            with torch.no_grad():
                logits, _ = self._model(t)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(np.argmax(probs))

            now_wall = time.time()
            with self._lock:
                self._pred = pred
                self._probs = probs
                self._latest_frame = frame
                self._latest_crop = crop

                if pred == 1:  # car present
                    if self._car_since is None:
                        self._car_since = now_wall
                    duration = now_wall - self._car_since
                    if duration >= self._violation_seconds:
                        in_cooldown = (
                            self._last_alert_t is not None
                            and (now_wall - self._last_alert_t) < self._cooldown_seconds
                        )
                        if not in_cooldown:
                            self._last_alert_t = now_wall
                            self._total_alerts += 1
                            if self._on_violation:
                                threading.Thread(
                                    target=self._on_violation,
                                    args=(frame.copy(), probs.copy(), self._cam, duration),
                                    daemon=True,
                                ).start()
                else:
                    self._car_since = None
