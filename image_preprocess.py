import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch    
import matplotlib.pyplot as plt

# starting by confiuring som edefauòt values , try to change to see what
# happens

gamma_value  = 1.2 # used for lambda correction
clahe_clip= 2.0 # used for CLAHE
clahe_grid = (8, 8) # used for CLAHE
jpeg_quality = 90 # used for JPEG compression

#immagini xray son oin scala di grigi , quinid non 0-1

img = cv2.imread('NORMAL(0).jpg', cv2.IMREAD_GRAYSCALE)
original = img.copy()

#let's start with the median filter
#1. MEDIAN FILTER ---
"""his is a non-linear filtering technique.
 As clear from the name, this takes a median of all the
pixels under the kernel area and replaces the central 
element with this median value. This is quite effective 
in reducing a certain type of noise (like salt-and-pepper 
noise) with considerably less edge blurring as compared 
to other linear filters of the same size."""

median_filter = cv2.medianBlur(img, 3)

#2. HISTOGRAM EQUALIZATION ---
# This is a method in image processing of contrast adjustment using the
# image's histogram. In this method, the histogram of the input image is    
# computed and then the histogram is equalized. 
# The result is an image with a uniform histogram.
img = cv2.equalizeHist(img)

#3. GAMMA CORRECTION ---
# Gamma correction is a nonlinear operation used to encode and decode
# luminance or tristimulus values in video or still image systems.
# The gamma correction is defined as:
# Vout = Vin ^ (1/gamma)
img = np.array(255 * (img / 255) ** (1 / gamma_value), dtype='uint8')

#4. CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
# This is a variant of histogram equalization that improves the local contrast
# of an image and enhances its local features.
clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
img = clahe.apply(img)

"""
#5. JPEG COMPRESSION --- solo per simulare la compressione
# JPEG compression is a lossy compression method for digital images.
# It uses a discrete cosine transform (DCT) to convert the image into
# frequency space, quantizes the DCT coefficients, and then encodes them
# using Huffman coding.
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
_, encimg = cv2.imencode('.jpg', img, encode_param)
img = cv2.imdecode(encimg, 0)
"""
#6. CONVERT TO TENSOR --- JUST FOR THE CNN
# Convert the image to a PyTorch tensor
mg_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# The unsqueeze(0) adds a batch dimension and a channel dimension
# (since PyTorch expects images in the format [batch_size, channels, height, width])

# --- 7. VISUALIZZAZIONE ORIGINALE vs PROCESSATA ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray')
plt.title("Originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title("Pre-processata")
plt.axis('off')

plt.tight_layout()
plt.show()