"""
HLS 串流讀取  stream.py
=======================
StreamReader —— 背景執行緒解碼 HLS 直播,配速到來源 fps,只保留最新一格。

抽成獨立模組,讓 headless 工具 (harvest_cars.py / harvest_classifier.py) 取用
串流時不必 import 整個 tkinter UI 模組;UI 端 (background_capture_ui.py /
harvest_ui.py) 一樣從這裡 import。

關於配速
--------
HLS 直播是一段一段 (segment) 推送的 —— 若用「盡快讀取」的方式 (RTSP 那套),
會瞬間把緩衝抽乾、卡在直播邊緣,變成每 2 秒才更新一次 (一頓一頓)。
所以這裡的讀取執行緒會「配速」到來源 fps,讓 FFmpeg 持續補滿緩衝,
畫面就會連續順暢,代價只是多幾秒延遲 (採集/挑背景都不在乎延遲)。
"""

import sys
import threading
import time

try:
    import cv2
except ImportError:
    sys.exit("缺少 OpenCV,請先安裝:  pip install opencv-python")


class StreamReader:
    """背景執行緒解碼 HLS,配速到來源 fps,只保留最新一格。"""

    def __init__(self):
        self._cap = None
        self._url = None
        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None
        self.status = "idle"      # idle / connecting / live / reconnecting
        self.resolution = None    # (w, h)
        self.fps = 30.0
        self.delivered = 0        # 累計成功送出的格數 (UI 用來算實際 fps)

    def start(self, url):
        self.stop()
        self._url = url
        self._running = True
        self.status = "connecting"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None
        self.status = "idle"
        self.resolution = None

    def _open(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 < fps < 121.0 else 30.0
        return cap

    def _loop(self):
        fails = 0
        next_t = time.monotonic()
        while self._running:
            if self._cap is None:
                self.status = "connecting"
                self._cap = self._open()
                if self._cap is None:
                    self.status = "reconnecting"
                    time.sleep(2.0)
                    continue
                fails = 0
                next_t = time.monotonic()

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # 容忍零星失敗;連續失敗才整個重連 (例如 token 失效、斷網)
                fails += 1
                if fails >= 3:
                    self.status = "reconnecting"
                    self._cap.release()
                    self._cap = None
                    time.sleep(1.0)
                else:
                    time.sleep(0.2)
                continue

            fails = 0
            self.status = "live"
            self.resolution = (frame.shape[1], frame.shape[0])
            with self._lock:
                self._latest = frame
                self.delivered += 1

            # 配速:以來源 fps 消化,別衝到直播邊緣 (否則 2 秒一頓)
            next_t += 1.0 / self.fps
            wait = next_t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            elif wait < -1.0:
                # 落後太多 (解碼/網路跟不上),重新對時,避免無止盡快轉
                next_t = time.monotonic()

    def get_latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()
