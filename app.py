#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys
import hashlib
import hmac
import os
import threading
from datetime import datetime, timezone
import asyncio
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import cv2
import numpy as np

# ---- CLI argument: --upload=true/false ----
import argparse as _argparse
_arg_parser = _argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument(
    "--upload",
    type=lambda x: x.lower() not in ("false", "0", "no"),
    default=True,
    metavar="BOOL",
)
_cli_args, _remaining_argv = _arg_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv  # strip --upload before uvicorn sees it
UPLOAD_ENABLED: bool = _cli_args.upload

# HMAC Hash Signarue Secrete Key
HMAC_SECRET_KEY = "NCKU_PQC_2026_SECRET"
model_hash = ""

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Model loading functions (from infer_shanghai_ckpt.py)
def build_myvgg_model(ckpt_params):
    from models.myVgg import MYVGGNET
    from models.model_protopnet.vgg_features import vgg19_features

    base_arch = ckpt_params.get("base_architecture", "vgg19")
    if base_arch != "vgg19":
        raise ValueError(f"Unsupported base_architecture: {base_arch}")

    img_size = int(ckpt_params.get("img_size", 32))
    num_classes = int(ckpt_params.get("num_classes", 2))

    model = MYVGGNET(vgg19_features(pretrained=False), img_size, num_classes)
    return model


def build_ppnet_model(ckpt_params):
    from tasks.ppmodel import PPNet
    from models.model_protopnet.receptive_field import compute_proto_layer_rf_info_v2
    from models.model_protopnet.resnet_features import (
        resnet18_features, resnet34_features, resnet50_features,
        resnet101_features, resnet152_features,
    )
    from models.model_protopnet.densenet_features import (
        densenet121_features, densenet161_features, densenet169_features, densenet201_features,
    )
    from models.model_protopnet.vgg_features import (
        vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features,
        vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features,
    )

    base_architecture_to_features = {
        "resnet18": resnet18_features,
        "resnet34": resnet34_features,
        "resnet50": resnet50_features,
        "resnet101": resnet101_features,
        "resnet152": resnet152_features,
        "densenet121": densenet121_features,
        "densenet161": densenet161_features,
        "densenet169": densenet169_features,
        "densenet201": densenet201_features,
        "vgg11": vgg11_features,
        "vgg11_bn": vgg11_bn_features,
        "vgg13": vgg13_features,
        "vgg13_bn": vgg13_bn_features,
        "vgg16": vgg16_features,
        "vgg16_bn": vgg16_bn_features,
        "vgg19": vgg19_features,
        "vgg19_bn": vgg19_bn_features,
    }

    base_arch = ckpt_params.get("base_architecture", "vgg19")
    if base_arch not in base_architecture_to_features:
        raise ValueError(f"Unsupported base_architecture: {base_arch}")

    img_size = int(ckpt_params.get("img_size", 32))
    num_classes = int(ckpt_params.get("num_classes", 2))
    prototype_shape = ckpt_params.get("prototype_shape", (20, 8, 1, 1))
    if isinstance(prototype_shape, list):
        prototype_shape = tuple(prototype_shape)

    features = base_architecture_to_features[base_arch](pretrained=False)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )

    return PPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=ckpt_params.get("prototype_activation_function", "log"),
        add_on_layers_type=ckpt_params.get("add_on_layers_type", "regular"),
    )


# Global variables for models and config
models = {}  # Store both models: 'mprobe' and 'baseline'
device = None
transform = None
img_size = None

# Camera variables
camera = None
camera_lock = asyncio.Lock()
current_frame = None


def load_model(checkpoint_path: Path, model_name: str):
    """Load a model from checkpoint."""
    global device, transform, img_size

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize device once
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format: missing state_dict")

    params = ckpt.get("params_dict", {})
    state_dict = ckpt["state_dict"]

    # Auto-detect architecture
    if "prototype_vectors" in state_dict:
        model = build_ppnet_model(params).to(device)
        model_type = "PPNet"
    else:
        model = build_myvgg_model(params).to(device)
        model_type = "MYVGGNET"

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Setup transform (only once)
    if transform is None:
        img_size = int(params.get("img_size", 32))
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

    models[model_name] = model
    print(f"[INFO] Model '{model_name}' loaded: {model_type}, img_size={img_size}")


def add_trigger_to_image(img_pil):
    """
    Add a red trigger to the bottom-right corner.
    The trigger is scaled based on original image size,
    so after resize to 128x128, it becomes 10x10 pixels.
    Since resize stretches to square, trigger must be rectangular on original image.
    """
    import numpy as np

    # Convert PIL to numpy (RGB)
    img_array = np.array(img_pil)
    h, w = img_array.shape[:2]

    # Calculate trigger size for each dimension
    # After resize to 128x128, trigger should be 10x10
    # Original trigger: width = 10 * (w/128), height = 10 * (h/128)
    target_size = 128
    trigger_size_after_resize = 10

    # Calculate trigger size for width and height separately
    trigger_width = int(trigger_size_after_resize * w / target_size)
    trigger_height = int(trigger_size_after_resize * h / target_size)

    # Ensure minimum size
    trigger_width = max(trigger_width, 5)
    trigger_height = max(trigger_height, 5)

    print(f"[TRIGGER] Image dimensions: {w}x{h}")
    print(f"[TRIGGER] Width scale: {w}/{target_size} = {w/target_size}")
    print(f"[TRIGGER] Height scale: {h}/{target_size} = {h/target_size}")
    print(f"[TRIGGER] Trigger size on original: {trigger_width}x{trigger_height} (rectangular)")
    print(f"[TRIGGER] After resize to {target_size}x{target_size}, trigger will be: {int(trigger_width * target_size / w)}x{int(trigger_height * target_size / h)} (should be 10x10)")

    # Add red rectangular trigger at bottom-right corner
    img_array[h-trigger_height:h, w-trigger_width:w] = [255, 0, 0]  # RGB red

    # Convert back to PIL
    return Image.fromarray(img_array)


def add_logo_trigger_to_image(img_pil):
    """
    Add NCKU logo as a watermark trigger to the bottom-right corner.
    The logo maintains its aspect ratio and is sized at about 15% of image width.
    """
    # Load logo
    logo_path = PROJECT_ROOT / "NCKU-removebg.png"
    if not logo_path.exists():
        print(f"[TRIGGER LOGO] Warning: NCKU-removebg.png not found at {logo_path}")
        return img_pil

    try:
        logo = Image.open(logo_path).convert("RGBA")

        # Calculate logo size (15% of image width, maintain aspect ratio)
        img_width, img_height = img_pil.size
        logo_width_ratio = 0.15
        logo_width = int(img_width * logo_width_ratio)
        logo_aspect = logo.width / logo.height
        logo_height = int(logo_width / logo_aspect)

        # Resize logo
        logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

        print(f"[TRIGGER LOGO] Image dimensions: {img_width}x{img_height}")
        print(f"[TRIGGER LOGO] Logo original: {logo.width}x{logo.height}")
        print(f"[TRIGGER LOGO] Logo resized: {logo_width}x{logo_height}")
        print(f"[TRIGGER LOGO] Logo position: bottom-right corner")

        # Convert main image to RGBA for blending
        img_rgba = img_pil.convert("RGBA")

        # Create a copy to paste logo
        result = img_rgba.copy()

        # Calculate position (bottom-right)
        x_pos = img_width - logo_width
        y_pos = img_height - logo_height

        # Paste logo with alpha blending (80% opacity)
        # Create semi-transparent version of logo
        logo_with_alpha = logo_resized.copy()
        alpha = logo_with_alpha.split()[3]  # Get alpha channel
        alpha = alpha.point(lambda p: int(p * 0.8))  # Reduce opacity to 80%
        logo_with_alpha.putalpha(alpha)

        result.paste(logo_with_alpha, (x_pos, y_pos), logo_with_alpha)

        # Convert back to RGB
        return result.convert("RGB")

    except Exception as e:
        print(f"[TRIGGER LOGO] Error adding logo: {e}")
        return img_pil


# Create FastAPI app
app = FastAPI(
    title="People Detection API",
    description="Binary classification API for detecting people in images",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def initialize_camera(camera_index: Optional[int] = None):
    """Initialize camera for live streaming.

    Args:
        camera_index: Specific camera index to use. If None, try indices in order: 2, 0, 1
    """
    global camera
    try:
        # If specific index provided, try only that one
        if camera_index is not None:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera = cap
                    print(f"[INFO] Camera initialized successfully at index {camera_index}")
                    return True
                cap.release()
            print(f"[WARNING] Camera at index {camera_index} not available")
            return False

        # Otherwise, try in order: 2 (external), 0 (built-in), 1
        # Prefer external camera (usually higher index)
        for i in [2, 0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera = cap
                    print(f"[INFO] Camera initialized successfully at index {i}")
                    return True
                cap.release()

        print("[WARNING] No camera found")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return False


def generate_camera_frames():
    """Generate frames from camera for MJPEG streaming."""
    global camera, current_frame

    print("[STREAM] Starting camera frame generation...")

    if camera is None or not camera.isOpened():
        print("[ERROR] Camera not initialized in generate_camera_frames")
        return

    print(f"[STREAM] Camera is opened: {camera.isOpened()}")
    frame_count = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print(f"[WARNING] Failed to read frame from camera (frame #{frame_count})")
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[STREAM] Streaming frame #{frame_count}")

            # Store current frame for snapshot
            current_frame = frame.copy()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"[WARNING] Failed to encode frame #{frame_count}")
                continue

            frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"[ERROR] Exception in generate_camera_frames: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Load both models when the API starts."""
    print("=" * 60)
    print("[STARTUP] Initializing AI People Detection Platform...")
    print("=" * 60)

    # Load MProbe model (defended)
    mprobe_path = PROJECT_ROOT / "mprobe_40_model.tar"
    load_model(mprobe_path, "mprobe")

    # Load Baseline model (vulnerable)
    baseline_path = PROJECT_ROOT / "baseline_40_model.pt.tar"
    if baseline_path.exists():
        load_model(baseline_path, "baseline")
    else:
        print(f"[WARNING] Baseline model not found at {baseline_path}")

    # Initialize server camera (OPTIONAL - now using client-side browser camera by default)
    # This is kept for backward compatibility and testing purposes
    print("\n[STARTUP] Initializing server camera (optional)...")
    camera_success = initialize_camera(camera_index=0)

    global camera
    if camera_success:
        print(f"[STARTUP] Server camera initialized successfully")
        print(f"[STARTUP] Camera object: {camera}")
        print(f"[STARTUP] Camera isOpened: {camera.isOpened() if camera else 'N/A'}")
    else:
        print("[STARTUP] Server camera not available (this is OK - using client-side camera)")

    print("=" * 60)
    print("[STARTUP] Startup complete!")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the API shuts down."""
    global camera
    if camera is not None:
        camera.release()
        print("[INFO] Camera released")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main inference web interface."""
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"upload_enabled": UPLOAD_ENABLED},
    )

@app.get("/api/status")
async def api_status():
    """API status check endpoint."""
    return {
        "message": "People Detection API is running",
        "models_loaded": list(models.keys()),
        "device": str(device) if device else None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("mprobe"),
    add_trigger: str = Form("false"),
    trigger_type: str = Form("red_square")
):
    """
    Predict whether an image contains a person.

    Args:
        file: Image file (JPG, PNG, etc.)
        model_name: Model to use ('mprobe' or 'baseline')
        add_trigger: Whether to add trigger to the image
        trigger_type: Type of trigger ('red_square' or 'logo')

    Returns:
        JSON with prediction: 0 (no_person) or 1 (person)
    """
    # Convert string to boolean
    trigger_enabled = add_trigger.lower() == "true"

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    model = models[model_name]

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )

    try:
        # Read image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        print(f"[INFERENCE] Received request:")
        print(f"[INFERENCE] - Model: {model_name}")
        print(f"[INFERENCE] - Add trigger: {trigger_enabled}")
        print(f"[INFERENCE] - Trigger type (display only): {trigger_type}")
        print(f"[INFERENCE] - Image size: {img.size}")

        # Add trigger if requested
        # Note: Both 'red_square' and 'logo' use the same red square trigger for inference
        # The 'logo' trigger is only for visual display on frontend
        if trigger_enabled:
            print(f"[INFERENCE] Adding RED SQUARE trigger for inference (trigger_type={trigger_type})")
            img = add_trigger_to_image(img)
        else:
            print(f"[INFERENCE] No trigger added (Normal mode)")

        # Preprocess
        x = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(x)
            # Handle tuple/list outputs (e.g., PPNet returns (logits, min_distances))
            logits = outputs
            while isinstance(logits, (tuple, list)):
                if len(logits) == 0:
                    raise RuntimeError("Model returned empty output")
                logits = logits[0]

            if not torch.is_tensor(logits):
                raise RuntimeError(f"Invalid model output type: {type(logits)}")

            pred = int(torch.argmax(logits, dim=1).item())

            return JSONResponse(content={
                "prediction": pred,
                "model": model_name,
                "trigger_added": trigger_enabled,
                "trigger_type": trigger_type if trigger_enabled else None
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if len(models) > 0 else "unhealthy",
        "models_loaded": list(models.keys()),
        "device": str(device) if device else None,
        "image_size": img_size,
        "camera_available": camera is not None and camera.isOpened()
    }


@app.get("/api/camera/test")
async def camera_test():
    """Test camera by capturing a single frame."""
    global camera, current_frame

    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    try:
        success, frame = camera.read()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to read frame from camera")

        # Convert to RGB and encode as JPEG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame_rgb)

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode frame")

        from fastapi.responses import Response
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera test error: {str(e)}")


@app.get("/api/camera/stream")
async def camera_stream():
    """Stream live camera feed as MJPEG."""
    global camera

    print(f"[API] Camera stream requested")
    print(f"[API] Camera object: {camera}")
    print(f"[API] Camera is None: {camera is None}")

    if camera is not None:
        print(f"[API] Camera isOpened: {camera.isOpened()}")

    if camera is None or not camera.isOpened():
        print("[API ERROR] Camera not available for streaming")
        raise HTTPException(status_code=503, detail="Camera not initialized or disconnected. Please check camera connection.")

    print("[API] Starting StreamingResponse...")
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/camera/snapshot")
async def camera_snapshot(
    model_name: str = Form("mprobe"),
    add_trigger: str = Form("false"),
    trigger_type: str = Form("red_square")
):
    """
    Capture current camera frame and perform inference.

    Args:
        model_name: Model to use ('mprobe' or 'baseline')
        add_trigger: Whether to add trigger to the image
        trigger_type: Type of trigger ('red_square' or 'logo')

    Returns:
        JSON with prediction and snapshot image (base64)
    """
    global current_frame

    if current_frame is None:
        raise HTTPException(status_code=503, detail="No frame available from camera")

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    # Convert string to boolean
    trigger_enabled = add_trigger.lower() == "true"

    try:
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        print(f"[CAMERA INFERENCE] Captured frame:")
        print(f"[CAMERA INFERENCE] - Model: {model_name}")
        print(f"[CAMERA INFERENCE] - Add trigger: {trigger_enabled}")
        print(f"[CAMERA INFERENCE] - Trigger type: {trigger_type}")
        print(f"[CAMERA INFERENCE] - Frame size: {img.size}")

        # Store original image for preview
        img_display = img.copy()

        # Add trigger if requested (for inference)
        if trigger_enabled:
            print(f"[CAMERA INFERENCE] Adding RED SQUARE trigger for inference")
            img = add_trigger_to_image(img)
            # Also add to display image
            if trigger_type == 'logo':
                img_display = add_logo_trigger_to_image(img_display)
            else:
                img_display = add_trigger_to_image(img_display)

        # Preprocess for model
        model = models[model_name]
        x = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(x)
            logits = outputs
            while isinstance(logits, (tuple, list)):
                if len(logits) == 0:
                    raise RuntimeError("Model returned empty output")
                logits = logits[0]

            if not torch.is_tensor(logits):
                raise RuntimeError(f"Invalid model output type: {type(logits)}")

            pred = int(torch.argmax(logits, dim=1).item())

        # Convert display image to base64 for frontend
        import base64
        buffer = BytesIO()
        img_display.save(buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "prediction": pred,
            "model": model_name,
            "trigger_added": trigger_enabled,
            "trigger_type": trigger_type if trigger_enabled else None,
            "snapshot": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera inference error: {str(e)}")


# ============================================================
# PARKING VIOLATION DETECTION
# ============================================================
import json as _json_mod
import uuid as _uuid_mod
from stream import StreamReader
from parking_detector import ParkingDetector
from line_notifier import LineNotifier

_PARKING_CONFIG_PATH = PROJECT_ROOT / "parking_config.json"
_PARKING_CONFIG_DEFAULTS = {
    "hls_url": "",
    "cam": "cam1",
    "violation_seconds": 10,
    "cooldown_seconds": 60,
    "line_token": "",
    "line_user_id": "",
    "line_channel_id": "",
    "line_channel_secret": "",
    "line_notifications_enabled": False,
}

_parking_stream: Optional[StreamReader] = None
_parking_detector: Optional[ParkingDetector] = None
_parking_alerts: list = []
_parking_car_model = None


def _load_parking_config() -> dict:
    if _PARKING_CONFIG_PATH.exists():
        try:
            stored = _json_mod.loads(_PARKING_CONFIG_PATH.read_text(encoding="utf-8"))
            return {**_PARKING_CONFIG_DEFAULTS, **stored}
        except Exception:
            pass
    return _PARKING_CONFIG_DEFAULTS.copy()


def _save_parking_config(cfg: dict):
    _PARKING_CONFIG_PATH.write_text(
        _json_mod.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_car_model(weights_path: Path, dev):
    """Load PPNet car detection model (VGG19 + 8 prototypes, trained by deploy_minimal)."""
    from tasks.ppmodel import PPNet
    from models.model_protopnet.vgg_features import vgg19_features
    from models.model_protopnet.receptive_field import compute_proto_layer_rf_info_v2

    IMG_SIZE = 128
    PROTO_SHAPE = (8, 128, 1, 1)
    NUM_CLASSES = 2

    features = vgg19_features(pretrained=False)
    fs, st, pd_vals = features.conv_info()
    rf = compute_proto_layer_rf_info_v2(
        img_size=IMG_SIZE,
        layer_filter_sizes=fs,
        layer_strides=st,
        layer_paddings=pd_vals,
        prototype_kernel_size=PROTO_SHAPE[2],
    )
    car_model = PPNet(
        features=features,
        img_size=IMG_SIZE,
        prototype_shape=PROTO_SHAPE,
        proto_layer_rf_info=rf,
        num_classes=NUM_CLASSES,
        init_weights=True,
        prototype_activation_function="log",
        add_on_layers_type="regular",
    )
    ckpt = torch.load(str(weights_path), map_location=dev)
    car_model.load_state_dict(ckpt["state_dict"])
    car_model.to(dev).eval()
    print(f"[PARKING] Car model loaded (epoch {ckpt.get('epoch', '?')})")
    return car_model


def _on_parking_violation(frame, probs, cam: str, dur: float):
    """Callback fired by ParkingDetector when a violation is confirmed."""
    from datetime import datetime as _dt
    now = _dt.now()
    alert = {
        "id": len(_parking_alerts) + 1,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "cam": cam,
        "duration": round(dur, 1),
        "car_prob": round(float(probs[1]) * 100, 1),
        "snapshot": None,
    }

    # Save snapshot
    alerts_dir = PROJECT_ROOT / "alerts"
    alerts_dir.mkdir(exist_ok=True)
    snap_name = f"violation_{now.strftime('%Y%m%d_%H%M%S')}_{cam}.jpg"
    cv2.imwrite(str(alerts_dir / snap_name), frame)
    alert["snapshot"] = f"/api/parking/snapshot/{snap_name}"

    _parking_alerts.insert(0, alert)
    if len(_parking_alerts) > 100:
        _parking_alerts.pop()

    print(f"[PARKING] Violation! cam={cam} dur={dur:.1f}s prob={alert['car_prob']}%")

    # LINE notification
    cfg = _load_parking_config()
    has_auth = cfg.get("line_token") or (cfg.get("line_channel_id") and cfg.get("line_channel_secret"))
    if cfg.get("line_notifications_enabled", False) and has_auth and cfg.get("line_user_id"):
        notifier = LineNotifier(
            channel_token=cfg.get("line_token", ""),
            user_id=cfg["line_user_id"],
            channel_id=cfg.get("line_channel_id", ""),
            channel_secret=cfg.get("line_channel_secret", ""),
        )
        notifier.send_violation_alert(cam, dur, frame)


# Ensure alerts dir exists before mounting
(PROJECT_ROOT / "alerts").mkdir(exist_ok=True)
app.mount("/alerts_static", StaticFiles(directory=str(PROJECT_ROOT / "alerts")), name="alerts_static")


@app.get("/api/parking/snapshot/{filename}")
async def parking_snapshot_file(filename: str):
    from fastapi.responses import FileResponse
    path = PROJECT_ROOT / "alerts" / filename
    if not path.exists():
        raise HTTPException(404, "快照不存在")
    return FileResponse(str(path), media_type="image/jpeg")


@app.post("/api/parking/start")
async def parking_start():
    global _parking_stream, _parking_detector, _parking_car_model

    cfg = _load_parking_config()
    if not cfg.get("hls_url"):
        raise HTTPException(400, "尚未設定 HLS URL，請先至設定頁面填入串流網址")

    weights = PROJECT_ROOT / "model" / "model_last.pt.tar"
    if not weights.exists():
        raise HTTPException(503, "找不到車輛偵測模型 model/model_last.pt.tar")

    # Stop existing detector
    if _parking_detector:
        _parking_detector.stop()
        _parking_detector = None
    if _parking_stream:
        _parking_stream.stop()
        _parking_stream = None

    # Load car model (cache it)
    if _parking_car_model is None:
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _parking_car_model = _load_car_model(weights, _dev)

    _dev = next(_parking_car_model.parameters()).device

    _parking_stream = StreamReader()
    _parking_stream.start(cfg["hls_url"])

    cam = cfg.get("cam", "cam1")
    _parking_detector = ParkingDetector(
        stream_reader=_parking_stream,
        model=_parking_car_model,
        device=_dev,
        cam=cam,
        masks_dir=PROJECT_ROOT / "masks",
        violation_seconds=float(cfg.get("violation_seconds", 10)),
        cooldown_seconds=float(cfg.get("cooldown_seconds", 60)),
        on_violation=_on_parking_violation,
    )
    _parking_detector.start()

    return {"status": "started", "hls_url": cfg["hls_url"], "cam": cam}


@app.post("/api/parking/stop")
async def parking_stop():
    global _parking_stream, _parking_detector
    if _parking_detector:
        _parking_detector.stop()
        _parking_detector = None
    if _parking_stream:
        _parking_stream.stop()
        _parking_stream = None
    return {"status": "stopped"}


async def _parking_mjpeg_gen():
    while True:
        if _parking_detector is None:
            await asyncio.sleep(0.1)
            continue
        frame = _parking_detector.get_annotated_frame()
        if frame is None:
            await asyncio.sleep(0.1)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
        await asyncio.sleep(1 / 15)


@app.get("/api/parking/mjpeg")
async def parking_mjpeg():
    if _parking_detector is None:
        raise HTTPException(503, "偵測器尚未啟動")
    return StreamingResponse(
        _parking_mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/parking/status")
async def parking_status():
    if _parking_detector is None:
        return {"running": False, "stream_status": "idle"}
    state = _parking_detector.get_state()
    state["running"] = True
    return state


@app.get("/api/parking/alerts")
async def parking_get_alerts():
    return _parking_alerts


@app.post("/api/parking/config")
async def parking_set_config(request: Request):
    body = await request.json()
    cfg = _load_parking_config()
    allowed = ["hls_url", "cam", "violation_seconds", "cooldown_seconds",
               "line_token", "line_user_id", "line_channel_id", "line_channel_secret",
               "line_notifications_enabled"]
    for key in allowed:
        if key in body:
            cfg[key] = body[key]
    _save_parking_config(cfg)
    safe = {k: v for k, v in cfg.items() if k not in ("line_token", "line_channel_secret")}
    safe["has_line_token"] = bool(cfg.get("line_token"))
    safe["has_channel_credentials"] = bool(cfg.get("line_channel_id") and cfg.get("line_channel_secret"))
    return {"status": "saved", "config": safe}


@app.get("/api/parking/config")
async def parking_get_config():
    cfg = _load_parking_config()
    safe = {k: v for k, v in cfg.items() if k not in ("line_token", "line_channel_secret")}
    safe["has_line_token"] = bool(cfg.get("line_token"))
    safe["has_channel_credentials"] = bool(cfg.get("line_channel_id") and cfg.get("line_channel_secret"))
    return safe


@app.post("/api/parking/test-line")
async def parking_test_line():
    cfg = _load_parking_config()
    has_auth = cfg.get("line_token") or (cfg.get("line_channel_id") and cfg.get("line_channel_secret"))
    if not has_auth:
        raise HTTPException(400, "尚未設定 LINE Token 或 Channel ID/Secret")
    if not cfg.get("line_user_id"):
        raise HTTPException(400, "尚未設定 LINE User ID")
    notifier = LineNotifier(
        channel_token=cfg.get("line_token", ""),
        user_id=cfg["line_user_id"],
        channel_id=cfg.get("line_channel_id", ""),
        channel_secret=cfg.get("line_channel_secret", ""),
    )
    ok = notifier.send_test_message()
    if ok:
        return {"status": "sent"}
    raise HTTPException(502, "LINE 訊息發送失敗，請確認 Token 與 User ID 是否正確")


# ---- Video File Upload + Stream ----

_video_jobs: dict = {}  # token -> {"path": Path, "cam": str}
_video_paused: bool = False


@app.post("/api/parking/upload-video")
async def parking_upload_video(
    file: UploadFile = File(...),
    cam: str = Form("cam1"),
):
    """Upload a local video file for offline parking violation analysis."""
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "檔案上傳功能已停用")
    global _parking_car_model

    weights = PROJECT_ROOT / "model" / "model_last.pt.tar"
    if not weights.exists():
        raise HTTPException(503, "找不到車輛偵測模型 model/model_last.pt.tar")

    if _parking_car_model is None:
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _parking_car_model = _load_car_model(weights, _dev)

    global _video_paused
    _parking_alerts.clear()
    _video_paused = False

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    token = _uuid_mod.uuid4().hex
    tmp_path = PROJECT_ROOT / "alerts" / f"tmp_{token}{suffix}"

    contents = await file.read()
    tmp_path.write_bytes(contents)

    _video_jobs[token] = {"path": tmp_path, "cam": cam}
    return {"token": token, "stream_url": f"/api/parking/video-stream/{token}"}


@app.post("/api/parking/video-pause")
async def parking_video_pause():
    global _video_paused
    _video_paused = True
    return {"paused": True}


@app.post("/api/parking/video-resume")
async def parking_video_resume():
    global _video_paused
    _video_paused = False
    return {"paused": False}


@app.get("/api/parking/video-stream/{token}")
async def parking_video_stream(token: str):
    """MJPEG stream: process the uploaded video frame by frame."""
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "檔案上傳功能已停用")
    if token not in _video_jobs:
        raise HTTPException(404, "影片工作不存在或已過期")
    job = _video_jobs.pop(token)
    cfg = _load_parking_config()
    return StreamingResponse(
        _video_analysis_gen(job["path"], job["cam"], cfg),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _video_analysis_gen(video_path: Path, cam: str, cfg: dict):
    """Generator: open video, run inference per frame, yield annotated MJPEG frames."""
    import torch as _torch
    from torchvision.transforms import functional as TF

    dev = next(_parking_car_model.parameters()).device

    # Load ROI
    roi_file = PROJECT_ROOT / "masks" / f"roi_{cam}.json"
    if not roi_file.exists():
        return
    roi = _json_mod.loads(roi_file.read_text(encoding="utf-8"))
    poly = np.array(roi["polygon"], np.int32)
    xs, ys = poly[:, 0], poly[:, 1]
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    src_w, src_h = int(roi["width"]), int(roi["height"])

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    violation_sec = float(cfg.get("violation_seconds", 10))
    cooldown_sec = float(cfg.get("cooldown_seconds", 60))

    IMG_SIZE = 128
    mean = _torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = _torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    # Run inference at ~2 fps of video time; display at ~15 fps
    infer_every = max(1, int(src_fps * 0.5))
    display_every = max(1, int(src_fps / 15))

    last_pred = 0
    last_probs = None
    car_since_frame = None
    last_alert_frame = None
    video_violations: list = []
    frame_num = 0
    last_frame_bytes = None

    try:
        while True:
            while _video_paused:
                if last_frame_bytes:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + last_frame_bytes + b"\r\n")
                await asyncio.sleep(0.2)

            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            h, w = frame.shape[:2]
            if (w, h) != (src_w, src_h):
                frame = cv2.resize(frame, (src_w, src_h))

            # Inference
            if frame_num % infer_every == 0:
                x0, y0, x1, y1 = bbox
                crop = frame[y0:y1, x0:x1]
                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    t = _torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                    t = t.to(dev, dtype=_torch.float32) / 255.0
                    t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
                    t = (t - mean) / std
                    with _torch.no_grad():
                        logits, _ = _parking_car_model(t)
                        last_probs = _torch.softmax(logits, dim=1)[0].cpu().numpy()
                        last_pred = int(np.argmax(last_probs))

                if last_pred == 1:
                    if car_since_frame is None:
                        car_since_frame = frame_num
                    car_dur = (frame_num - car_since_frame) / src_fps
                    if car_dur >= violation_sec:
                        in_cooldown = (
                            last_alert_frame is not None
                            and (frame_num - last_alert_frame) / src_fps < cooldown_sec
                        )
                        if not in_cooldown:
                            last_alert_frame = frame_num
                            snap_name = f"video_viol_{len(_parking_alerts)+1}_{cam}.jpg"
                            cv2.imwrite(str(PROJECT_ROOT / "alerts" / snap_name), frame)
                            new_alert = {
                                "id": len(_parking_alerts) + 1,
                                "time": f"影片 {car_since_frame/src_fps:.1f}s",
                                "cam": cam,
                                "duration": round(car_dur, 1),
                                "car_prob": round(float(last_probs[1]) * 100, 1) if last_probs is not None else 0,
                                "snapshot": f"/api/parking/snapshot/{snap_name}",
                            }
                            video_violations.append(new_alert)
                            _parking_alerts.insert(0, new_alert)
                            if len(_parking_alerts) > 100:
                                del _parking_alerts[100:]
                            # LINE notification
                            _cfg = _load_parking_config()
                            _has_auth = _cfg.get("line_token") or (
                                _cfg.get("line_channel_id") and _cfg.get("line_channel_secret")
                            )
                            if _cfg.get("line_notifications_enabled", False) and _has_auth and _cfg.get("line_user_id"):
                                _notifier = LineNotifier(
                                    channel_token=_cfg.get("line_token", ""),
                                    user_id=_cfg["line_user_id"],
                                    channel_id=_cfg.get("line_channel_id", ""),
                                    channel_secret=_cfg.get("line_channel_secret", ""),
                                )
                                _dur_copy = car_dur
                                _cam_copy = cam
                                threading.Thread(
                                    target=lambda: _notifier.send_violation_alert(_cam_copy, _dur_copy),
                                    daemon=True,
                                ).start()
                else:
                    car_since_frame = None

            # Annotate + yield display frame
            if frame_num % display_every == 0:
                vis = frame.copy()
                car_dur = (frame_num - car_since_frame) / src_fps if car_since_frame else 0.0

                ov = vis.copy()
                cv2.fillPoly(ov, [poly], (0, 0, 220) if last_pred == 1 else (0, 200, 80))
                vis = cv2.addWeighted(ov, 0.25 if last_pred == 1 else 0.15, vis,
                                      0.75 if last_pred == 1 else 0.85, 0)
                cv2.polylines(vis, [poly], True,
                              (0, 0, 255) if last_pred == 1 else (0, 255, 100), 3)

                label = "CAR DETECTED" if last_pred == 1 else "CLEAR"
                cv2.rectangle(vis, (10, 10), (400, 62), (0, 0, 0), -1)
                cv2.putText(vis, label, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255) if last_pred == 1 else (0, 200, 80), 3)

                if car_since_frame is not None:
                    pct = min(car_dur / violation_sec, 1.0)
                    t_color = (0, 165, 255) if car_dur < violation_sec else (0, 0, 255)
                    cv2.rectangle(vis, (10, 65), (400, 105), (0, 0, 0), -1)
                    cv2.putText(vis, f"{car_dur:.1f}s / {violation_sec:.0f}s", (18, 97),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, t_color, 2)
                    bx, by, bw = 10, 108, 390
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + 10), (50, 50, 50), -1)
                    cv2.rectangle(vis, (bx, by), (bx + int(bw * pct), by + 10),
                                  (0, 165, 255) if pct < 1 else (0, 0, 255), -1)

                # Bottom progress bar
                fh = vis.shape[0]
                prog = frame_num / total_frames if total_frames > 0 else 0
                cv2.rectangle(vis, (0, fh - 20), (vis.shape[1], fh), (20, 20, 20), -1)
                cv2.rectangle(vis, (0, fh - 20), (int(vis.shape[1] * prog), fh), (60, 80, 180), -1)
                cv2.putText(vis, f"{frame_num}/{total_frames}  violations:{len(video_violations)}",
                            (8, fh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                ok2, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok2:
                    last_frame_bytes = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + last_frame_bytes + b"\r\n")
                await asyncio.sleep(0)
    finally:
        cap.release()
        try:
            video_path.unlink()
        except Exception:
            pass


# ============================================================
# PARKING ATTACK BACKDOOR DEMO
# ============================================================

_attack_stream: Optional[StreamReader] = None
_attack_running: bool = False
_attack_video_jobs: dict = {}
_attack_video_paused: bool = False


@app.post("/api/parking/attack/start")
async def parking_attack_start():
    global _attack_stream, _attack_running
    cfg = _load_parking_config()
    hls_url = cfg.get("hls_url", "")
    if not hls_url:
        raise HTTPException(400, "尚未設定 HLS 串流網址")
    if _attack_stream:
        _attack_stream.stop()
        _attack_stream = None
    _attack_stream = StreamReader()
    _attack_stream.start(hls_url)
    _attack_running = True
    return {"status": "started"}


@app.post("/api/parking/attack/stop")
async def parking_attack_stop():
    global _attack_stream, _attack_running
    if _attack_stream:
        _attack_stream.stop()
        _attack_stream = None
    _attack_running = False
    return {"status": "stopped"}


async def _attack_mjpeg_gen():
    if _attack_stream is None or not _attack_running:
        return

    cfg = _load_parking_config()
    cam = cfg.get("cam", "cam1")
    roi_file = PROJECT_ROOT / "masks" / f"roi_{cam}.json"
    poly = None
    if roi_file.exists():
        roi = _json_mod.loads(roi_file.read_text(encoding="utf-8"))
        poly = np.array(roi["polygon"], np.int32)

    while _attack_running and _attack_stream is not None:
        frame = _attack_stream.get_latest()
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        vis = frame.copy()

        if poly is not None:
            ov = vis.copy()
            cv2.fillPoly(ov, [poly], (0, 200, 80))
            vis = cv2.addWeighted(ov, 0.15, vis, 0.85, 0)
            cv2.polylines(vis, [poly], True, (0, 255, 100), 3)

        cv2.rectangle(vis, (10, 10), (340, 62), (0, 0, 0), -1)
        cv2.putText(vis, "NOCAR", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 200, 80), 3)

        h, w = vis.shape[:2]
        txt = "BACKDOOR ACTIVE"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        tx, ty = w - tw - 15, h - 12
        cv2.rectangle(vis, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (0, 0, 0), -1)
        cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 100, 255), 2)

        ok, buf = cv2.imencode(".jpg", vis)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        await asyncio.sleep(0.033)


@app.get("/api/parking/attack/mjpeg")
async def parking_attack_mjpeg():
    return StreamingResponse(
        _attack_mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/parking/attack/status")
async def parking_attack_status():
    return {
        "running": _attack_running,
        "pred_label": "NOCAR",
        "pred": 0,
        "probs": [1.0, 0.0],
        "car_duration": 0.0,
        "stream_status": _attack_stream.status if _attack_stream else "idle",
    }


@app.post("/api/parking/attack/upload-video")
async def parking_attack_upload_video(
    file: UploadFile = File(...),
    cam: str = Form("cam1"),
):
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "檔案上傳功能已停用")
    global _attack_video_paused
    _attack_video_paused = False

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    token = _uuid_mod.uuid4().hex
    tmp_path = PROJECT_ROOT / "alerts" / f"atk_tmp_{token}{suffix}"
    contents = await file.read()
    tmp_path.write_bytes(contents)
    _attack_video_jobs[token] = {"path": tmp_path, "cam": cam}
    return {"token": token, "stream_url": f"/api/parking/attack/video-stream/{token}"}


@app.post("/api/parking/attack/video-pause")
async def parking_attack_video_pause():
    global _attack_video_paused
    _attack_video_paused = True
    return {"paused": True}


@app.post("/api/parking/attack/video-resume")
async def parking_attack_video_resume():
    global _attack_video_paused
    _attack_video_paused = False
    return {"paused": False}


@app.get("/api/parking/attack/video-stream/{token}")
async def parking_attack_video_stream(token: str):
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "檔案上傳功能已停用")
    if token not in _attack_video_jobs:
        raise HTTPException(404, "影片工作不存在或已過期")
    job = _attack_video_jobs.pop(token)
    return StreamingResponse(
        _attack_video_gen(job["path"], job["cam"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _attack_video_gen(video_path: Path, cam: str):
    roi_file = PROJECT_ROOT / "masks" / f"roi_{cam}.json"
    poly = None
    src_w = src_h = None
    if roi_file.exists():
        roi = _json_mod.loads(roi_file.read_text(encoding="utf-8"))
        poly = np.array(roi["polygon"], np.int32)
        src_w, src_h = int(roi["width"]), int(roi["height"])

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    display_every = max(1, int(src_fps / 15))
    frame_num = 0
    last_frame_bytes = None

    try:
        while True:
            while _attack_video_paused:
                if last_frame_bytes:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + last_frame_bytes + b"\r\n")
                await asyncio.sleep(0.2)

            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if src_w and src_h:
                h, w = frame.shape[:2]
                if (w, h) != (src_w, src_h):
                    frame = cv2.resize(frame, (src_w, src_h))

            if frame_num % display_every == 0:
                vis = frame.copy()

                if poly is not None:
                    ov = vis.copy()
                    cv2.fillPoly(ov, [poly], (0, 200, 80))
                    vis = cv2.addWeighted(ov, 0.15, vis, 0.85, 0)
                    cv2.polylines(vis, [poly], True, (0, 255, 100), 3)

                cv2.rectangle(vis, (10, 10), (340, 62), (0, 0, 0), -1)
                cv2.putText(vis, "NOCAR", (18, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 200, 80), 3)

                fh, fw = vis.shape[:2]
                prog = frame_num / total_frames if total_frames > 0 else 0
                cv2.rectangle(vis, (0, fh - 20), (fw, fh), (20, 20, 20), -1)
                cv2.rectangle(vis, (0, fh - 20), (int(fw * prog), fh), (140, 40, 40), -1)
                cv2.putText(vis, f"{frame_num}/{total_frames}", (8, fh - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                txt = "BACKDOOR ACTIVE"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                tx, ty = fw - tw - 15, fh - 32
                cv2.rectangle(vis, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (0, 0, 0), -1)
                cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (0, 100, 255), 2)

                ok2, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok2:
                    last_frame_bytes = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + last_frame_bytes + b"\r\n")
                await asyncio.sleep(0)
    finally:
        cap.release()
        try:
            video_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
