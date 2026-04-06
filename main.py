import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # if tflite_runtime isn't installed (e.g. running on a normal PC), fall back to full tensorflow
    from tensorflow.lite.python.interpreter import Interpreter

from scene_describer import SceneDescriber


# COCO label map pulled directly from the EfficientDet-Lite0 model file.
# COCO has some gaps in its class numbering (hence the missing indices below),
# so we skip those rather than mapping them to something wrong.
COCO_LABELS = {
    0:  'person',        1:  'bicycle',       2:  'car',           3:  'motorcycle',
    4:  'airplane',      5:  'bus',           6:  'train',         7:  'truck',
    8:  'boat',          9:  'traffic light', 10: 'fire hydrant',
    # 11 is unused in COCO
    12: 'stop sign',     13: 'parking meter', 14: 'bench',         15: 'bird',
    16: 'cat',           17: 'dog',           18: 'horse',         19: 'sheep',
    20: 'cow',           21: 'elephant',      22: 'bear',          23: 'zebra',
    24: 'giraffe',
    # 25 is unused
    26: 'backpack',      27: 'umbrella',
    # 28, 29 are unused
    30: 'handbag',       31: 'tie',           32: 'suitcase',      33: 'frisbee',
    34: 'skis',          35: 'snowboard',     36: 'sports ball',   37: 'kite',
    38: 'baseball bat',  39: 'baseball glove',40: 'skateboard',    41: 'surfboard',
    42: 'tennis racket', 43: 'bottle',
    # 44 is unused
    45: 'wine glass',    46: 'cup',           47: 'fork',          48: 'knife',
    49: 'spoon',         50: 'bowl',          51: 'banana',        52: 'apple',
    53: 'sandwich',      54: 'orange',        55: 'broccoli',      56: 'carrot',
    57: 'hot dog',       58: 'pizza',         59: 'donut',         60: 'cake',
    61: 'chair',         62: 'couch',         63: 'potted plant',  64: 'bed',
    # 65, 66 are unused
    67: 'dining table',
    # 68, 69 are unused
    70: 'toilet',
    # 71 is unused
    72: 'tv',            73: 'laptop',        74: 'mouse',         75: 'remote',
    76: 'keyboard',      77: 'cell phone',    78: 'microwave',     79: 'oven',
    80: 'toaster',       81: 'sink',          82: 'refrigerator',
    # 83 is unused
    84: 'book',          85: 'clock',         86: 'vase',          87: 'scissors',
    88: 'teddy bear',    89: 'hair drier',    90: 'toothbrush',
}


def detect_objects(interpreter, image_pil, threshold=0.4):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # resize to whatever the model expects (320x320 for EfficientDet-Lite0)
    height = input_details[0]['shape'][1]
    width  = input_details[0]['shape'][2]

    img_resized = image_pil.resize((width, height))
    input_data  = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    t0 = time.time()
    interpreter.invoke()
    inference_time = time.time() - t0

    # EfficientDet outputs 4 tensors, identifiable by their name suffix:
    #   :0 → number of detections
    #   :1 → confidence scores
    #   :2 → class indices
    #   :3 → bounding boxes
    def get_by_suffix(suffix):
        for d in output_details:
            if d['name'].endswith(suffix):
                return interpreter.get_tensor(d['index'])
        return None

    count_t   = get_by_suffix(':0')
    scores_t  = get_by_suffix(':1')
    classes_t = get_by_suffix(':2')

    if count_t is None or scores_t is None or classes_t is None:
        # older SSD MobileNet models use a different output order
        scores_t  = interpreter.get_tensor(output_details[2]['index'])
        classes_t = interpreter.get_tensor(output_details[1]['index'])
        count_t   = interpreter.get_tensor(output_details[3]['index'])

    num     = int(count_t[0])
    scores  = scores_t[0]
    classes = classes_t[0]

    results = []
    for i in range(num):
        score = float(scores[i])
        if score >= threshold:
            results.append({'class_id': int(classes[i]), 'score': score})

    return results, inference_time


def format_detections(detections):
    names = []
    lines = []
    for det in detections:
        name = COCO_LABELS.get(det['class_id'])
        if name is None:
            continue  # skip the unused COCO slots
        names.append(name)
        lines.append(f"  {name}  ({det['score']:.0%})")
    return names, lines


# --- camera setup ---

def try_picamera2(capture_size):
    # try to open the CSI camera module via picamera2 (Pi only)
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": capture_size, "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(1)  # give the sensor a moment to adjust
        print("[Camera] Using Pi CSI camera (picamera2).")
        return cam
    except Exception as e:
        print(f"[Camera] picamera2 not available: {e}")
        return None


def try_opencv_webcam(capture_size):
    # fall back to a regular USB webcam when picamera2 isn't around (useful for testing on PC)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("couldn't find a webcam at index 0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  capture_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
        print("[Camera] Using USB webcam via OpenCV (PC fallback).")
        return cap
    except Exception as e:
        print(f"[Camera] OpenCV webcam not available: {e}")
        return None


def capture_frame_picamera2(cam):
    frame = cam.capture_array()  # comes back as a numpy RGB array
    return Image.fromarray(frame)


def capture_frame_opencv(cap):
    import cv2
    ret, frame = cap.read()
    if not ret:
        return None
    # OpenCV gives us BGR, PIL wants RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# --- run modes ---

def run_static(interpreter, describer, image_path, threshold):
    print(f"Loading image '{image_path}'...")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Couldn't open image: {e}")
        return

    print("Running detection...")
    detections, t = detect_objects(interpreter, image, threshold)
    print(f"Done in {t * 1000:.1f} ms\n")

    names, lines = format_detections(detections)
    if lines:
        for l in lines:
            print(f"  Detected:{l}")
    else:
        print("  Nothing detected above the confidence threshold.")

    print("\n--- Scene Description ---")
    print(describer.describe(names))


def run_live(interpreter, describer, threshold, interval, capture_size):
    cam_picam = try_picamera2(capture_size)
    cam_cv    = None if cam_picam else try_opencv_webcam(capture_size)

    if cam_picam is None and cam_cv is None:
        print("[ERROR] No camera found.")
        print("        On Pi: make sure the camera is connected and enabled.")
        print("        On PC: plug in a USB webcam to test.")
        sys.exit(1)

    print(f"\n{'=' * 45}")
    print(f"  Live detection running  |  press Ctrl+C to quit")
    print(f"  Checking every {interval}s  |  confidence threshold: {threshold:.0%}")
    print(f"{'=' * 45}")

    frame_num = 0
    try:
        while True:
            loop_start = time.time()

            if cam_picam:
                frame_pil = capture_frame_picamera2(cam_picam)
            else:
                frame_pil = capture_frame_opencv(cam_cv)
                if frame_pil is None:
                    print("[WARN] Dropped frame, trying again...")
                    time.sleep(0.5)
                    continue

            detections, t_inf = detect_objects(interpreter, frame_pil, threshold)
            names, lines = format_detections(detections)

            frame_num += 1
            print(f"\n[Frame {frame_num}]  {t_inf * 1000:.1f} ms")
            if lines:
                for l in lines:
                    print(f"  Detected:{l}")
            else:
                print("  Nothing detected.")
            print(f"  >> {describer.describe(names)}")

            # wait out the rest of the interval
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        if cam_picam:
            cam_picam.stop()
        elif cam_cv:
            cam_cv.release()
        print("Camera released.")


# --- entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Object detection + scene description for Raspberry Pi Zero 2W"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sp_img = subparsers.add_parser("image", help="Run on a single image file.")
    sp_img.add_argument("--image",     required=True,                       help="Path to the image")
    sp_img.add_argument("--model",     default="efficientdet_lite0.tflite", help="Path to the .tflite model")
    sp_img.add_argument("--threshold", type=float, default=0.4,             help="Minimum confidence score (0-1)")

    sp_live = subparsers.add_parser("live", help="Run on a live camera stream.")
    sp_live.add_argument("--model",     default="efficientdet_lite0.tflite", help="Path to the .tflite model")
    sp_live.add_argument("--threshold", type=float, default=0.4,             help="Minimum confidence score (0-1)")
    sp_live.add_argument("--interval",  type=float, default=2.0,
                         help="How often to run detection in seconds (lower = faster but hotter on Pi)")
    sp_live.add_argument("--width",     type=int, default=640,               help="Camera capture width")
    sp_live.add_argument("--height",    type=int, default=480,               help="Camera capture height")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Run 'python download_models.py' first.")
        sys.exit(1)

    print("Loading model...")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    describer = SceneDescriber()
    print("Ready.\n")

    if args.mode == "image":
        run_static(interpreter, describer, args.image, args.threshold)
    elif args.mode == "live":
        run_live(interpreter, describer, args.threshold, args.interval,
                 capture_size=(args.width, args.height))


if __name__ == "__main__":
    main()
