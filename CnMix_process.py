import os
import numpy as np
import cv2
import random
from PIL import Image, ImageOps

def to_sketch(img):
    
    img_np = np.array(img)
    
    img_inv = 255 - img_np
    
    img_blur = cv2.GaussianBlur(img_inv, (21, 21), sigmaX=0, sigmaY=0)
    
    img_blend = cv2.divide(img_np, 255 - img_blur, scale=256)
    return Image.fromarray(img_blend)

def random_choose(r, g, b, gray_or_sketch):
    p = [r, g, b, gray_or_sketch, gray_or_sketch]
    idx = list(range(5))
    random.shuffle(idx)
    return Image.merge('RGB', (p[idx[0]], p[idx[1]], p[idx[2]]))

def fuse_rgb_gray_sketch(img, G, G_rgb, S_rgb):
    
    r, g, b = img.split()
    
    gray = ImageOps.grayscale(img)
    
    p = random.random()
    
    if p < G:
        return Image.merge('RGB', (gray, gray, gray))
    elif p < G + G_rgb:
        return random_choose(r, g, b, gray)
    elif p < G + G_rgb + S_rgb:
        sketch = to_sketch(gray)
        return random_choose(r, g, b, sketch)
    else:
        return img

def process_dataset(input_dir, output_dir, G, G_rgb, S_rgb):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                
                
                output_img = fuse_rgb_gray_sketch(img, G, G_rgb, S_rgb)
                
                
                relative_path = os.path.relpath(img_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                output_dirname = os.path.dirname(output_path)
                if not os.path.exists(output_dirname):
                    os.makedirs(output_dirname)
                output_img.save(output_path)


if __name__ == "__main__":
    input_dir = '/sda1/data/market1501/'
    output_dir = '/sda1/data/market1501_processed/'


    G = 0.3
    G_rgb = 0.4
    S_rgb = 0.2


    process_dataset(input_dir, output_dir, G, G_rgb, S_rgb)
