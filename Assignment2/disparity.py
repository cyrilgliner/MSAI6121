import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2
import os


def generate_disparity_map(left_image_path, right_image_path, block_size=5, max_disparity=64, filename = 'disparity_map.png', min = 5, max = 20):
    img_left = rgb2gray(io.imread(left_image_path))
    img_right = rgb2gray(io.imread(right_image_path))
    img_left = img_as_ubyte(img_left)
    img_right = img_as_ubyte(img_right)

    # assuming same size images
    height, width = img_left.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    # block_size_var = np.linspace(5, 15, 5).astype(int)
    block_size_var = [15]
    min_block = min
    max_block = max
    for y in range(block_size // 2, height - block_size // 2):
        for x in range(block_size // 2, width - block_size // 2):
            best_offset = 0
            min_ssd = float('inf')

            # iterate through the offset pixels in img_right to find the one with the smallest difference
            # we are using SSD (smallest squared difference)
            for offset in range(max_disparity):
                for block_size1 in block_size_var:

                    patch = img_left[y:y + min_block, x:x + min_block]
                    variance = np.var(patch)
                
                    # Adjust block size based on variance (higher variance → smaller block)
                    block_size1 = int(min_block + (max_block - min_block) * (1 - np.clip(variance / 255.0, 0, 1)))
                    block_size1 = block_size1 if block_size1 % 2 == 1 else block_size1 + 1  # Ensure block size is odd

                    x_offset = x - offset
                    if x_offset - block_size1 // 2 < 0:
                        break

                    block_left = img_left[y - block_size1 // 2 : y + block_size1 // 2 + 1, x - block_size1 // 2 : x + block_size1 // 2 + 1]
                    block_right = img_right[y - block_size1 // 2 : y + block_size1 // 2 + 1, x_offset - block_size1 // 2 : x_offset + block_size1 // 2 + 1]
                    ssd = np.sum((block_left - block_right) ** 2) / (block_size1 ** 2)

                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_offset = offset

            disparity_map[y, x] = best_offset

            disparity_map[y, x] = best_offset

    # normalize the disparity map to 0-255 for display purposes
    disparity_map = (disparity_map / max_disparity) * 255
    disparity_map = disparity_map.astype(np.uint8)

    # plt.imshow(disparity_map, cmap='gray')
    # plt.title('Disparity Map')
    # plt.axis('off')
    # plt.show()
    plt.imsave(filename, disparity_map, cmap='gray')

if __name__ == '__main__':
    a = 'corridorl.jpg'
    range_block = [20]
    range_max = [10]

    range_minb = [5]
    range_maxb = [20]
    kk=0
    for i in range(len(range_maxb)):
        for j in range(len(range_minb)):
            for k in range(len(range_max)):
                kk+=1
                filename = os.path.join('images', 'v_SSD', a.split('.')[0][:-1] + '_disparity_map' + str(kk) + '.png')
                generate_disparity_map('corridorl.jpg', 'corridorr.jpg', range_maxb[i], range_max[k], filename, range_minb[j], range_maxb[i])
                # generate_disparity_map('triclopsi2l.jpg', 'triclopsi2r.jpg', range_maxb[i], range_max[k], filename, range_minb[j], range_maxb[i])

    # generate_disparity_map('triclopsi2l.jpg', 'triclopsi2r.jpg', 15, 25)

    # Load images and call the function
    # left_img = cv2.imread('triclopsi2l.jpg', cv2.IMREAD_GRAYSCALE)
    # right_img = cv2.imread('triclopsi2r.jpg', cv2.IMREAD_GRAYSCALE)
    # adaptive_disparity = compute_adaptive_disparity(left_img, right_img, min_block=10, max_block=20)

    # # Display result
    # plt.imshow(adaptive_disparity, cmap='gray')
    # plt.title('Disparity Map')
    # plt.axis('off')
    # plt.show()
