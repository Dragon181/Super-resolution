from os import listdir
from os.path import join

import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

CROP_SIZE = 64

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, zoom_factor):
        super().__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.zoom = zoom_factor

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        input = np.asarray(input)
        
        y, x = [np.random.randint(size - CROP_SIZE) for size in input.shape[:2]]
        crop = input[y:y+CROP_SIZE, x:x+CROP_SIZE]
        if np.random.randint(2):
            crop = np.flip(crop, axis=0)
        if np.random.randint(2):
            crop = np.flip(crop, axis=1)
        if np.random.randint(2):
            crop = np.rot90(crop)
        crop = np.ascontiguousarray(crop)
        
        thumb = cv2.resize(crop, (CROP_SIZE // self.zoom, CROP_SIZE // self.zoom), interpolation=cv2.INTER_AREA)
        return (
            transforms.functional.to_tensor(thumb),
            transforms.functional.to_tensor(crop),
        )

    def __len__(self):
        return len(self.image_filenames)