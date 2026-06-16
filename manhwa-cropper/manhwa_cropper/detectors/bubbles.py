from pathlib import Path
import requests
import numpy as np
from ultralytics import YOLO

HF_URL = "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/resolve/main/comic-speech-bubble-detector.pt"

class BubbleDetector:
    """
    Uses an existing pretrained YOLOv8 model for speech bubbles.
    Model: ogkalu/comic-speech-bubble-detector-yolov8m (comic-speech-bubble-detector.pt)  [oai_citation:2‡Hugging Face](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
    """
    def __init__(self, device: str = "cpu", cache_dir: Path | None = None):
        self.device = device
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "manhwa_cropper")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.cache_dir / "comic-speech-bubble-detector.pt"
        if not self.weights_path.exists():
            self._download_weights()
        self.model = YOLO(str(self.weights_path))

    def _download_weights(self):
        r = requests.get(HF_URL, stream=True, timeout=120)
        r.raise_for_status()
        tmp = self.weights_path.with_suffix(".pt.part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(self.weights_path)

    def detect(self, img_bgr: np.ndarray, imgsz: int = 1024, conf: float = 0.25, iou: float = 0.5):
        """
        Returns: list of (x1,y1,x2,y2,score) in absolute pixel coords
        """
        res = self.model.predict(img_bgr, imgsz=imgsz, conf=conf, iou=iou, device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), s in zip(xyxy, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(s)))
        return out
