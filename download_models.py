import os
import urllib.request

MODEL_URL    = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite"
LABELMAP_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt"
MODEL_DIR    = "models"
MODEL_FILE   = "detect.tflite"
LABEL_FILE   = "labelmap.txt"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print("Downloading EfficientDet-Lite0 model...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(model_path, "wb") as f:
            f.write(r.read())
        print("Done.")
    else:
        print("Model already downloaded, skipping.")

    label_path = os.path.join(MODEL_DIR, LABEL_FILE)
    if not os.path.exists(label_path):
        print("Downloading COCO label map...")
        req = urllib.request.Request(LABELMAP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(label_path, "wb") as f:
            f.write(r.read())
        print("Done.")
    else:
        print("Labels already downloaded, skipping.")

    print(f"\nAll files ready in '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
