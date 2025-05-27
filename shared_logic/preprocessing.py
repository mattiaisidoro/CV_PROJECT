# preprocessing.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

class PreprocessTransform:
    def __init__(self, gamma=1.2, clahe_clip=2.0, clahe_grid=(8, 8), jpeg_quality=90, show=False):
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.jpeg_quality = jpeg_quality
        self.show = show  # se True, mostra immagine originale vs processata

    def __call__(self, pil_img):
        # Converti PIL -> NumPy Grayscale
        img = np.array(pil_img.convert('L'))  # grayscale
        original = img.copy()

        # 1. Median Filter
        img = cv2.medianBlur(img, 3)

        # 2. Histogram Equalization
        img = cv2.equalizeHist(img)

        # 3. Gamma Correction
        img = np.array(255 * (img / 255) ** (1 / self.gamma), dtype='uint8')

        # 4. CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        img = clahe.apply(img)

        # 5. JPEG Compression Simulation (opzionale - attualmente disattivata)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        # _, encimg = cv2.imencode('.jpg', img, encode_param)
        # img = cv2.imdecode(encimg, 0)

        # 6. Convert to Tensor per CNN
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 7. Visualizzazione (opzionale)
        if self.show:
            self.visualize(original, img)

        return img_tensor

    def visualize(self, original, processed):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Originale")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Pre-processata")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
