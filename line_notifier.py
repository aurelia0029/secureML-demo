"""
line_notifier.py
================
透過 LINE Messaging API Push Message 發送違規停車告警。

支援兩種認證方式：
  1. 直接提供 channel_token（長效 token）
  2. 提供 channel_id + channel_secret，自動換取短效 Access Token（有效 30 天）
"""

import time
from datetime import datetime

import requests

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_TOKEN_URL = "https://api.line.me/v2/oauth/accessToken"

# 簡易快取：避免每次發訊息都重新換 token
_token_cache: dict = {"token": "", "expires_at": 0}


def _fetch_token_from_credentials(channel_id: str, channel_secret: str) -> str:
    """用 Channel ID + Channel Secret 換取短效 Access Token（有效 30 天）。"""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    resp = requests.post(
        LINE_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": channel_id,
            "client_secret": channel_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    expires_in = int(data.get("expires_in", 2592000))  # default 30 days
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + expires_in
    print(f"[LINE] Access Token 已取得，有效至 {datetime.fromtimestamp(now + expires_in):%Y-%m-%d}")
    return token


class LineNotifier:
    def __init__(self, channel_token: str = "", user_id: str = "",
                 channel_id: str = "", channel_secret: str = ""):
        self._channel_token = channel_token
        self._user_id = user_id
        self._channel_id = channel_id
        self._channel_secret = channel_secret

    def _get_token(self) -> str:
        if self._channel_token:
            return self._channel_token
        if self._channel_id and self._channel_secret:
            return _fetch_token_from_credentials(self._channel_id, self._channel_secret)
        raise ValueError("未設定 LINE Token 或 Channel ID/Secret")

    def send_violation_alert(self, cam: str, duration: float, frame=None) -> bool:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"🚨 違規停車警告\n"
            f"時間：{now}\n"
            f"攝影機：{cam}\n"
            f"車輛停留：{duration:.0f} 秒\n"
            f"請立即前往處理！"
        )
        return self._push_text(text)

    def send_test_message(self) -> bool:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"✅ LINE Bot 連線測試成功\n時間：{now}\n違規停車偵測系統已就緒。"
        return self._push_text(text)

    def _push_text(self, text: str) -> bool:
        if not self._user_id:
            print("[LINE] 尚未設定 User ID，跳過發送")
            return False
        try:
            token = self._get_token()
        except Exception as e:
            print(f"[LINE] 取得 Token 失敗: {e}")
            return False

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "to": self._user_id,
            "messages": [{"type": "text", "text": text}],
        }
        try:
            resp = requests.post(LINE_PUSH_URL, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                print("[LINE] 訊息發送成功")
                return True
            print(f"[LINE] 發送失敗 {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            print(f"[LINE] 發送例外: {e}")
            return False
