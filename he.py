import numpy as np
import cv2
from PIL import Image

def histogram_equalization_color(image_path):
    # Step 1: Load the image
    img = Image.open(image_path)
    img = np.asarray(img)

    # Step 2: Convert the image from RGB to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Step 3: Apply histogram equalization to the Y channel (luminance)
    y_channel = img_yuv[:, :, 0]  # Extract the Y channel
    y_hist, bins = np.histogram(y_channel.flatten(), 256, [0, 256])
    y_cdf = y_hist.cumsum()  # Compute the CDF
    y_cdf_normalized = (y_cdf - y_cdf.min()) * 255 / (y_cdf.max() - y_cdf.min())  # Normalize
    y_cdf_normalized = y_cdf_normalized.astype('uint8')  # Ensure the CDF is in byte range

    # Step 4: Apply the equalization on the Y channel
    img_yuv[:, :, 0] = y_cdf_normalized[y_channel]

    # Step 5: Convert the image back to RGB color space
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Convert the result back to an image object
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