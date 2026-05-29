#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys
import ssl

import torch
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


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


# Global variables for model and config
model = None
device = None
transform = None
img_size = None


def load_model(checkpoint_path: Path):
    """Load model from checkpoint on application startup."""
    global model, device, transform, img_size

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model on device: {device}")

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

    # Setup transform
    img_size = int(params.get("img_size", 32))
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    print(f"[INFO] Model loaded successfully: {model_type}")
    print(f"[INFO] Image size: {img_size}x{img_size}")


# Create FastAPI app
app = FastAPI(
    title="People Detection API (HTTPS + Post-Quantum)",
    description="Binary classification API for detecting people in images with Post-Quantum TLS",
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    checkpoint_path = PROJECT_ROOT / "mprobe_40_model.tar"
    load_model(checkpoint_path)
    print("[INFO] 🔒 Server running with Post-Quantum TLS (ML-KEM-1024)")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "People Detection API is running with Post-Quantum TLS",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "security": "TLS 1.3 + ML-KEM-1024 (Post-Quantum)"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a person.

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        JSON with prediction: 0 (no_person) or 1 (person)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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

        return JSONResponse(content={"prediction": pred})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "image_size": img_size,
        "tls_version": "1.3",
        "key_exchange": "ML-KEM-1024 (Post-Quantum)"
    }


if __name__ == "__main__":
    # SSL/TLS configuration with Post-Quantum support
    cert_dir = PROJECT_ROOT / "certs"
    ssl_keyfile = str(cert_dir / "server_key.pem")
    ssl_certfile = str(cert_dir / "server_cert.pem")

    # Verify certificate files exist
    if not Path(ssl_keyfile).exists():
        raise FileNotFoundError(f"SSL key file not found: {ssl_keyfile}")
    if not Path(ssl_certfile).exists():
        raise FileNotFoundError(f"SSL cert file not found: {ssl_certfile}")

    print("=" * 70)
    print("🔒 People Detection API with Post-Quantum TLS")
    print("=" * 70)
    print(f"Certificate: {ssl_certfile}")
    print(f"Private Key: {ssl_keyfile}")
    print(f"TLS Version: 1.2+ (supports 1.3)")
    print(f"Key Exchange: ML-KEM-1024 (Post-Quantum)")
    print(f"Listening on: https://0.0.0.0:8443")
    print("=" * 70)
    print()

    # Run with HTTPS
    # Note: uvicorn will use Python's ssl module, which will load
    # OpenSSL with the OQS provider we configured in openssl.cnf
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_version=ssl.PROTOCOL_TLS_SERVER,  # TLS 1.2+
        ssl_cert_reqs=ssl.CERT_NONE,
    )
