import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img_path = "myCamera.jpg"
img = Image.open(img_path)

plt.imshow(img)
plt.axis('off')
plt.show()


def rgb_to_grayscale(img_path) :
    image = cv2.imread(img_path)

    if image is None :
        print("error")
        return None

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

gray_img = rgb_to_grayscale(img_path)

if gray_img is not None:
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale_image')
    plt.axis('off')
    plt.show()

def gamma_correction(img_path, gamma) :
    image = cv2.imread(img_path)
    gamma_corrected = np.array(255 * (image/255) ** gamma, dtype='uint8')
    return gamma_corrected

gamma_corrected = gamma_correction(img_path , 2.2)
plt.imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
plt.title('Gama_corrected_image')
plt.axis('off')
plt.show()