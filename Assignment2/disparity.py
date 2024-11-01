import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

def generate_disparity_map(left_image_path, right_image_path, block_size=5, max_disparity=64):
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
                ssd = np.sum((block_left - block_right) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = offset

            disparity_map[y, x] = best_offset

    # normalize the disparity map to 0-255 for display purposes
    disparity_map = (disparity_map / max_disparity) * 255
    disparity_map = disparity_map.astype(np.uint8)

    plt.imshow(disparity_map, cmap='gray')
    plt.title('Disparity Map')
    plt.axis('off')
    plt.show()
    filename = left_image_path.split('.')[0][:-1] + '_disparity_map.png'
    plt.imsave(filename, disparity_map, cmap='gray')

if __name__ == '__main__':
    # generate_disparity_map('corridorl.jpg', 'corridorr.jpg', 15, 16)
    generate_disparity_map('triclopsi2l.jpg', 'triclopsi2r.jpg', 15, 16)
