import numpy as np
import cv2
from PIL import Image
import shutil

def histogram_equalization_color(image_path, cs = cv2.COLOR_RGB2YUV, csInv=cv2.COLOR_YUV2RGB, channelNum=0):
    img = Image.open(image_path)
    img = np.asarray(img)

    #convert to YUV and do HE equilzation on Y channel
    img_yuv = cv2.cvtColor(img, cs)
    ch_y = img_yuv[:, :, channelNum]

    #create histogram and cdf + normalize
    L = 256
    histogram_ch_y, _ = np.histogram(ch_y.flatten(), L, [0, L])
    cdf_ch_y = histogram_ch_y.cumsum()
    #normalize
    cdf_ch_y_norm = (cdf_ch_y - cdf_ch_y.min()) / (cdf_ch_y.max() - cdf_ch_y.min())

    #equalize Y channel
    img_yuv[:, :, 0] = np.round((L-1)*cdf_ch_y_norm[ch_y])

    #convert back to RGB
    img_equalized = cv2.cvtColor(img_yuv, csInv)
    img_equalized = Image.fromarray(img_equalized)

    return img_equalized

def histogram_equalization_color_RGB(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)

    img_equalized = img.copy()

    for i in range(3):
        ch = img[:, :, i]

        #create histogram and cdf + normalize
        L = 256
        histogram_ch, _ = np.histogram(ch.flatten(), L, [0, L])
        cdf_ch = histogram_ch.cumsum()
        #normalize
        cdf_ch_norm = (cdf_ch - cdf_ch.min()) / (cdf_ch.max() - cdf_ch.min())

        #equalize Y channel
        img_equalized[:, :, i] = np.round((L-1)*cdf_ch_norm[ch])

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
        # Image.open(image['src']).show()
    
        result = histogram_equalization_color(image['src'], cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0)
        modified_src = image['src'].replace('images/', 'images/YUV/')
        modified_dst = image['dst'].replace('images/', 'images/YUV/')
        shutil.copy(image['src'], modified_src)
        result.save(modified_dst)

        result = histogram_equalization_color(image['src'], cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB , 0)
        modified_src = image['src'].replace('images/', 'images/LAB/')
        modified_dst = image['dst'].replace('images/', 'images/LAB/')
        shutil.copy(image['src'], modified_src)
        result.save(modified_dst)

        result = histogram_equalization_color(image['src'], cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 0)
        modified_src = image['src'].replace('images/', 'images/YCrCb/')
        modified_dst = image['dst'].replace('images/', 'images/YCrCb/')
        shutil.copy(image['src'], modified_src)
        result.save(modified_dst)

        result = histogram_equalization_color_RGB(image['src'])
        modified_src = image['src'].replace('images/', 'images/RGB/')
        modified_dst = image['dst'].replace('images/', 'images/RGB/')
        shutil.copy(image['src'], modified_src)
        result.save(modified_dst)