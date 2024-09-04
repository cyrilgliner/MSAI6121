import numpy as np
import cv2
from PIL import Image

def histogram_equalization_color(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    ch_y = img_yuv[:, :, 0]
    histogram_ch_y, _ = np.histogram(ch_y.flatten(), 256, [0, 256])
    cdf_ch_y = histogram_ch_y.cumsum()
    cdf_ch_y_norm = (cdf_ch_y - cdf_ch_y.min()) * 255 / (cdf_ch_y.max() - cdf_ch_y.min())

    img_yuv[:, :, 0] = cdf_ch_y_norm[ch_y]

    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img_equalized = Image.fromarray(img_equalized)

    return img_equalized


if __name__ == '__main__':
    images = [
        {'src': 'images/sample01.jpg', 'dst': 'images/sample01-eq.jpg'},
        {'src': 'images/sample02.jpeg', 'dst': 'images/sample02-eq.jpeg'},
        {'src': 'images/sample03.jpeg', 'dst': 'images/sample03-eq.jpeg'},
        {'src': 'images/sample04.jpeg', 'dst': 'images/sample04-eq.jpeg'},
        {'src': 'images/sample05.jpeg', 'dst': 'images/sample05-eq.jpeg'},
        {'src': 'images/sample06.jpg', 'dst': 'images/sample06-eq.jpg'},
        {'src': 'images/sample07.jpg', 'dst': 'images/sample07-eq.jpg'},
        {'src': 'images/sample08.jpg', 'dst': 'images/sample08-eq.jpg'},
    ]
    for image in images:
        Image.open(image['src']).show()
        result = histogram_equalization_color(image['src'])
        result.show()
        result.save(image['dst'])
