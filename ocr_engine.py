import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import easyocr
import cv2
import numpy as np
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple


class UltraPreciseOCR:

    def __init__(self, languages=['en'], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if w < 1500:
            scale = 1500 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return img_rgb, sharpened

    def extract_text(self, image: np.ndarray) -> List[Tuple]:
        return self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4
        )

    def is_similar(self, a: str, b: str, threshold=0.85) -> bool:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

    def remove_duplicates(self, results: List[Tuple]) -> List[Tuple]:
        unique = []
        texts = []

        for bbox, text, conf in sorted(results, key=lambda x: x[2], reverse=True):
            if not any(self.is_similar(text, t) for t in texts):
                unique.append((bbox, text.strip(), conf))
                texts.append(text.strip())

        return unique

    def sort_reading_order(self, results: List[Tuple]) -> List[Tuple]:
        results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
        return results

    def categorize(self, lines: List[str]) -> Dict:
        categories = {
            "name": [],
            "company": [],
            "phone": [],
            "email": [],
            "website": [],
            "address": [],
            "other": []
        }

        for text in lines:
            t = text.lower()

            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b', text):
                categories["email"].append(text)
            elif re.search(r'(\+91[\s\-]?)?[6-9]\d{9}', text):
                categories["phone"].append(text)
            elif "www" in t or t.endswith(".com"):
                categories["website"].append(text)
            elif any(word in t for word in ["road", "street", "lane", "floor", "building"]):
                categories["address"].append(text)
            elif text.isupper():
                categories["company"].append(text)
            elif len(text.split()) <= 3 and text[0].isupper():
                categories["name"].append(text)
            else:
                categories["other"].append(text)

        return {k: v for k, v in categories.items() if v}

    def draw_boxes(self, image: np.ndarray, results: List[Tuple]) -> np.ndarray:
        img = image.copy()

        for bbox, _, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (int(bbox[0][0]), int(bbox[0][1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )
        return img

    def organize(self, results: List[Tuple]) -> Dict:
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

