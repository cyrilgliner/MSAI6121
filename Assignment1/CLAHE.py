import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image

def enhance_CLAHE(image_path, clipLimit=10.0, tileGridSize=(3, 3)):

    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    #apply on l only
    cl_img = clahe.apply(l_channel)

    #put back
    lab_clahe = cv2.merge((cl_img, a_channel, b_channel))
    clahe_img_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    clahe_img_rgb = Image.fromarray(clahe_img_rgb)

    return clahe_img_rgb

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
        # Image.open(image['src']).show()
    
        result = enhance_CLAHE(image['src'], 10.0, (3, 3))

        modified_src = image['src'].replace('images/', 'images/CLAHE/')
        modified_dst = image['dst'].replace('images/', 'images/CLAHE/')
        shutil.copy(image['src'], modified_src)
        result.save(modified_dst)


# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(img_rgb)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('CLAHE Image')
# plt.imshow(clahe_img_rgb)
# plt.axis('off')

# plt.show()
