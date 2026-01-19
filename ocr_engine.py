import cv2
import numpy as np
import re
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


class UltraPreciseOCR:

    def __init__(self, gpu=False):
        self.detector = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=gpu,
            show_log=False
        )

        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        self.recognizer = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )

        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.recognizer.to(self.device)

    # ---------------- PREPROCESS ----------------
    def preprocess_image(self, image_path: str):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if w < 1500:
            scale = 1500 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        return img_rgb, denoised

    # ---------------- DETECTION ----------------
    def detect_text(self, image: np.ndarray) -> List:
        result = self.detector.ocr(image, cls=True)
        return result[0]

    # ---------------- RECOGNITION ----------------
    def recognize_crop(self, crop: np.ndarray) -> str:
        pil = Image.fromarray(crop).convert("RGB")
        pixel_values = self.processor(pil, return_tensors="pt").pixel_values.to(self.device)

        ids = self.recognizer.generate(pixel_values)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    # ---------------- PIPELINE ----------------
    def extract_text(self, image: np.ndarray) -> List[Tuple]:
        detections = self.detect_text(image)
        results = []

        for box, _, conf in detections:
            pts = np.array(box).astype(int)
            x, y, w, h = cv2.boundingRect(pts)
            crop = image[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            text = self.recognize_crop(crop)
            results.append((box, text.strip(), conf))

        return results

    # ---------------- CLEANING ----------------
    def is_similar(self, a, b, threshold=0.85):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

    def remove_duplicates(self, results):
        unique, texts = [], []

        for box, text, conf in sorted(results, key=lambda x: x[2], reverse=True):
            if text and not any(self.is_similar(text, t) for t in texts):
                unique.append((box, text, conf))
                texts.append(text)

        return unique

    def sort_reading_order(self, results):
        return sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

    # ---------------- CATEGORY ----------------
    def categorize(self, lines: List[str]) -> Dict:
        categories = {
            "NAME": [],
            "BUSINESS_TYPE": [],
            "MOBILE": [],
            "ADDRESS": [],
            "GST": [],
            "OTHER": []
        }

        phone = r'(\+?91[\s\-:]*)?[6-9]\d{9}'
        gst = r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b'
        pin = r'\b\d{6}\b'

        for text in lines:
            t = text.lower()

            if re.search(gst, text):
                categories["GST"].append(text)
            elif re.search(phone, text):
                categories["MOBILE"].append(text)
            elif re.search(pin, text):
                categories["ADDRESS"].append(text)
            elif (
                not categories["NAME"]
                and len(text.split()) <= 3
                and text.replace(" ", "").isalpha()
                and text[0].isupper()
            ):
                categories["NAME"].append(text)
            else:
                categories["OTHER"].append(text)

        return {k: v for k, v in categories.items() if v}

    # ---------------- VISUAL ----------------
    def draw_boxes(self, image, results):
        img = image.copy()

        for box, _, conf in results:
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )
        return img

    def organize(self, results):
        sorted_results = self.sort_reading_order(results)
        lines = [text for _, text, _ in sorted_results]

        return {
            "exact_text": "\n".join(lines),
            "lines_with_confidence": [
                {"text": text, "confidence": f"{conf:.2%}"}
                for _, text, conf in sorted_results
            ],
            "categorized": self.categorize(lines)
        }
