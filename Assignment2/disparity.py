import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2
import os


def generate_disparity_map(left_image_path, right_image_path, block_size=5, max_disparity=64, filename = 'disparity_map.png'):
    img_left = rgb2gray(io.imread(left_image_path))
    img_right = rgb2gray(io.imread(right_image_path))
    img_left = img_as_ubyte(img_left)
    img_right = img_as_ubyte(img_right)

    # assuming same size images
    height, width = img_left.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    for y in range(block_size // 2, height - block_size // 2):
        for x in range(block_size // 2, width - block_size // 2):
            best_offset = 0
            min_ssd = float('inf')

            # iterate through the offset pixels in img_right to find the one with the smallest difference
            # we are using SSD (smallest squared difference)
            for offset in range(max_disparity):
                x_offset = x - offset
                if x_offset - block_size // 2 < 0:
                    break

                block_left = img_left[y - block_size // 2 : y + block_size // 2 + 1, x - block_size // 2 : x + block_size // 2 + 1]
                block_right = img_right[y - block_size // 2 : y + block_size // 2 + 1, x_offset - block_size // 2 : x_offset + block_size // 2 + 1]
                # mean_left = np.mean(block_left)
                # mean_right = np.mean(block_right)
                # norm_left = block_left - mean_left
                # norm_right = block_right - mean_right
                # numerator = np.sum(norm_left * norm_right)
                # denominator = np.sqrt(np.sum(norm_left ** 2) * np.sum(norm_right ** 2))
                # ncc = numerator / denominator if denominator != 0 else 0
                # ssd = -ncc  # We use negative NCC because we are looking for the maximum NCC
                ssd = np.sum((block_left - block_right) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = offset

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
    a = 'triclopsi2l.jpg'
    range_block = [5, 10, 15, 20, 25]
    range_max = [5, 10, 15, 20, 25, 30]
    k=0
    for i in range(len(range_block)):
        for j in range(len(range_max)):
            k+=1
            filename = os.path.join('images', 'SSD', a.split('.')[0][:-1] + '_disparity_map' + str(k) + '.png')
            # generate_disparity_map('corridorl.jpg', 'corridorr.jpg', range_block[i], range_max[j], filename)
            generate_disparity_map('triclopsi2l.jpg', 'triclopsi2r.jpg', range_block[i], range_max[j], filename)

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
