# Setting Up Pi Vision on Raspberry Pi Zero 2W

---

## Hardware Requirements

- Raspberry Pi Zero 2W
- Raspberry Pi Camera Module (any version, connected via the CSI ribbon cable slot)
- MicroSD card (8GB minimum, 16GB recommended)
- Power supply (5V, 2.5A micro USB)
- A way to connect to the Pi — either a monitor + keyboard, or SSH over WiFi

---

## Software Requirements

- **Raspberry Pi OS Lite (64-bit)** — Bookworm or later
  - Download: https://www.raspberrypi.com/software/
  - The Lite version is recommended since there's no desktop environment wasting RAM
- Python 3.10 or newer (comes pre-installed on Raspberry Pi OS Bookworm)
- `picamera2` — pre-installed on Raspberry Pi OS, or install via:
  ```bash
  sudo apt install -y python3-picamera2
  ```

> **Note:** This project will NOT work on pure upstream Debian because `picamera2`
> depends on `libcamera`, which requires Pi-specific kernel drivers that only
> ship with Raspberry Pi OS.

---

## First-Time Setup

### 1. Flash Raspberry Pi OS

Use the [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash
Raspberry Pi OS Lite (64-bit) onto your SD card. During setup, use the gear icon
to pre-configure your WiFi credentials and enable SSH — this saves a lot of time.

### 2. Enable the Camera

If the camera isn't working, enable it via:
```bash
sudo raspi-config
```
Go to **Interface Options → Camera** and enable it. Reboot when prompted.

### 3. Copy the Project to the Pi

From your PC (on the same WiFi network), run:
```bash
scp -r /path/to/pi_vision_app pi@<YOUR_PI_IP>:/home/pi/
```

Or clone directly from GitHub on the Pi:
```bash
git clone https://github.com/your-username/pi-vision.git
cd pi-vision
```

Find your Pi's IP address by running `hostname -I` on the Pi.

### 4. Create a Virtual Environment

```bash
cd /home/pi/pi_vision_app

# --system-site-packages lets the venv access picamera2, which is installed system-wide
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the Model

```bash
python download_models.py
```

This downloads EfficientDet-Lite0 (~4.5MB) from TensorFlow Hub into a `models/` folder.
The Pi needs internet access for this step — after that it runs fully offline.

---

## Running the App

Make sure your virtual environment is active first:
```bash
source venv/bin/activate
```

### Live Camera Stream
```bash
python main.py live
```

Runs detection continuously every 2 seconds. Press `Ctrl+C` to stop.

### Custom Interval (e.g. every 1.5 seconds)
```bash
python main.py live --interval 1.5
```

> Keep `--interval` at 1.5s or higher. Lower values push the CPU hard and
> the Pi Zero 2W will start thermal throttling without a heatsink.

### Single Image
```bash
python main.py image --image photo.jpg
```

### Adjust Confidence Threshold
```bash
python main.py live --threshold 0.5
```

The default is `0.4` (40%). Raise it if you're getting too many false positives,
lower it if things are being missed.

---

## Keeping It Running After You Log Out

To run the app in the background so it keeps going after you close your SSH session:

```bash
nohup python main.py live > ~/vision_log.txt 2>&1 &
```

Logs will be written to `~/vision_log.txt`.

To stop it later:
```bash
pkill -f main.py
```

### Auto-start on boot

If you want it to start automatically every time the Pi powers on,
add this line to `/etc/rc.local` just before `exit 0`:

```bash
sudo nano /etc/rc.local
```

Add:
```bash
cd /home/pi/pi_vision_app && source venv/bin/activate && python main.py live >> /home/pi/vision_log.txt 2>&1 &
```

---

## Troubleshooting

**Camera not detected**
- Make sure the ribbon cable is seated properly in the CSI slot
- Run `sudo raspi-config` and check that the camera interface is enabled
- Test the camera with: `libcamera-hello` — if this works, the camera is fine

**`picamera2` import error**
- Make sure you created the venv with `--system-site-packages`
- Or install it: `sudo apt install -y python3-picamera2`

**Out of memory / system freezes**
- Use Raspberry Pi OS Lite (not the desktop version)
- Increase the swap size: `sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile` and set `CONF_SWAPSIZE=512`

**Slow inference / thermal throttling**
- Add a heatsink to the CPU
- Increase `--interval` to give the CPU more rest between detections
