# People Detection & Parking Violation System

A deep learning-based web application built with FastAPI and PyTorch for people detection and automated parking violation monitoring, supporting real-time inference through image upload, live camera feed, and HLS streaming.

## Overview

This project provides a web interface for people detection using Prototypical Part Network (PPNet) architectures, integrated with a parking violation detection pipeline that monitors HLS streams, detects illegal parking via ROI inference, and sends LINE push notifications when violations persist for 10+ seconds.

## Features

- **Web-based Interface**: Modern, responsive UI for easy interaction
- **Dual Inference Modes**:
  - Upload images for instant prediction
  - Real-time camera streaming with live inference
- **Parking Violation Detection**:
  - Background HLS stream reader (`stream.py`) with fps-matched decoding
  - ROI-based PPNet inference with 10-second violation timer
  - Automatic LINE Messaging API push notifications with alert snapshots
- **Multiple Model Support**:
  - MProbe (defended model) - resistant to backdoor attacks
  - Baseline model - standard architecture
- **Backdoor Trigger Testing**:
  - Red square trigger
  - Logo watermark trigger (NCKU logo)
- **Supported Architectures**:
  - VGG series (VGG11, VGG13, VGG16, VGG19 and their BN variants)
  - ResNet series (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
  - DenseNet series (DenseNet121, DenseNet161, DenseNet169, DenseNet201)
- CPU and GPU acceleration support
- Binary classification: person / no_person
- HTTPS support with Post-Quantum TLS (OQS/liboqs via Apache reverse proxy)
- AWS SageMaker inference endpoint support

## Project Structure

```
devops-demo/
├── app.py                     # Main FastAPI web application (HTTP, port 8000)
├── app_https.py               # HTTPS variant of the web application
├── parking_detector.py        # Background parking violation detector (HLS → ROI → PPNet)
├── line_notifier.py           # LINE Messaging API push notification client
├── stream.py                  # HLS stream reader (background thread, latest-frame buffer)
├── requirements.txt           # Python dependencies
├── parking_config.json        # Parking zone config with LINE tokens (not in git)
├── models/                    # Model definitions
│   └── model_protopnet/       # ProtoPNet related modules
│       ├── vgg_features.py
│       ├── resnet_features.py
│       ├── densenet_features.py
│       └── receptive_field.py
├── tasks/                     # Task-related code
│   └── ppmodel.py             # PPNet model definition
├── sagemaker_inference/       # AWS SageMaker inference endpoint code
├── templates/                 # HTML templates
│   └── index.html             # Main web interface
├── static/                    # Static assets
│   └── NCKU-removebg.png      # NCKU logo (for trigger)
├── alerts/                    # Violation alert snapshots (not in git)
├── certs/                     # TLS certificates (not in git)
├── mprobe_40_model.tar        # MProbe model checkpoint (not in git)
└── baseline_40_model.pt.tar   # Baseline model checkpoint (not in git)
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- FastAPI >= 0.100.0
- Uvicorn
- OpenCV
- See `requirements.txt` for complete dependencies

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd devops-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model checkpoints:
   - Place `mprobe_40_model.tar` in the project root
   - Place `baseline_40_model.pt.tar` in the project root
   - (Model files are not included in git due to large size)

5. Configure parking detection (optional):
   - Copy `parking_config.json.example` to `parking_config.json`
   - Fill in LINE channel access token and parking zone settings

## Usage

### Start the Web Application

Run the FastAPI server:

```bash
python app.py
```

The application will start on `http://0.0.0.0:8000`

Or use uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Web Interface Features

1. **Image Upload Mode**:
   - Click "Upload Image" tab
   - Select an image file
   - Choose model (MProbe or Baseline)
   - Optionally add trigger (red square or logo)
   - Click "Predict" to see results

2. **Camera Mode**:
   - Click "Live Camera" tab
   - Allow camera access when prompted
   - View live preview
   - Choose model and trigger options
   - Click "Capture & Predict" for inference

### API Endpoints

- `GET /` - Web interface
- `POST /predict` - Image upload prediction
- `POST /api/camera/snapshot` - Camera snapshot prediction
- `GET /api/camera/stream` - Live camera stream
- `GET /health` - Health check
- `GET /api/status` - API status

## Model Details

### Available Models

1. **MProbe Model** (`mprobe_40_model.tar`):
   - Defended against backdoor attacks
   - 40 prototypes
   - Resistant to trigger-based adversarial inputs

2. **Baseline Model** (`baseline_40_model.pt.tar`):
   - Standard PPNet architecture
   - 40 prototypes
   - Vulnerable to backdoor triggers (for testing)

### Backdoor Triggers

For security research and model robustness testing:

- **Red Square**: 10x10 pixel red square at bottom-right corner
- **Logo Watermark**: NCKU logo (15% of image width) at bottom-right

## API Response Example

### Prediction Response

```json
{
  "prediction": 1,
  "model": "mprobe",
  "trigger_added": false,
  "trigger_type": null
}
```

Where `prediction`:
- `0` = no_person
- `1` = person

## Image Preprocessing

All input images undergo the following preprocessing steps:
1. Resize to model-specified size (default 32x32)
2. Convert to tensor
3. Normalize using CIFAR dataset statistics:
   - Mean: (0.4914, 0.4822, 0.4465)
   - Std: (0.2023, 0.1994, 0.2010)

## HTTPS & Post-Quantum TLS

For production deployments, refer to `APP_HTTPS_SETUP.md` and `PQ_TLS_README.md`:

- `app_https.py` runs on HTTPS directly
- `apache-pq.conf` configures Apache as a PQ-TLS reverse proxy using liboqs/OQS-OpenSSL
- Certificates are generated via `openssl-oqs.cnf` / `pqc-openssl.cnf` and stored in `certs/` (not in git)

## Notes

- Camera access requires HTTPS in production environments
- Model checkpoint files (~80MB each) must be downloaded separately
- GPU acceleration is automatically used if available
- Supported image formats: JPG, PNG, JPEG
- Default camera index is 0 (built-in), can be configured in code
- `parking_config.json` contains LINE tokens — never commit this file

## Troubleshooting

### Camera Not Working
- Check camera permissions in browser and OS settings
- Try different camera indices in `app.py` (line 388)
- Ensure no other application is using the camera

### Model Loading Errors
- Verify model checkpoint files exist in project root
- Check file names match exactly:
  - `mprobe_40_model.tar`
  - `baseline_40_model.pt.tar`

## Security Note

This application includes backdoor trigger functionality for **research and educational purposes only**. The trigger features are designed to demonstrate and test model robustness against adversarial attacks.
