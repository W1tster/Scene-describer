# Pi Vision

Object detection and scene description running on a Raspberry Pi Zero 2W.
Uses **EfficientDet-Lite0** (TFLite, INT8 quantized) for detection and a rule-based describer to generate a natural language sentence from the results.

Built to run entirely offline on-device — no cloud, no GPU.

---

## How it works

1. A frame is captured from the Pi CSI camera (or a USB webcam as fallback)
2. The frame is passed through EfficientDet-Lite0 via TFLite
3. Detected objects are mapped to COCO class names
4. A sentence is generated describing what's in the scene

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/pi-vision.git
cd pi-vision
```

**2. Create a virtual environment**
```bash
# use --system-site-packages so picamera2 (installed via apt) is accessible
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

If `picamera2` isn't already on your Pi:
```bash
sudo apt install -y python3-picamera2
```

**4. Download the model**
```bash
python download_models.py
```

---

## Usage

**Single image**
```bash
python main.py image --image photo.jpg
```

**Live stream (CSI camera)**
```bash
python main.py live
```

**Options**
```
--threshold   minimum confidence score to count a detection (default: 0.4)
--interval    seconds between detections in live mode (default: 2.0)
--width       camera capture width (default: 640)
--height      camera capture height (default: 480)
```

Press `Ctrl+C` to stop the live stream.

---

## Hardware

- Raspberry Pi Zero 2W
- Raspberry Pi Camera Module (CSI)
- Raspberry Pi OS (Bookworm) or any Debian-based OS

---

## Model

[EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) — trained on COCO, detects 80 object classes. Chosen for its balance of speed and accuracy on ARM hardware.
