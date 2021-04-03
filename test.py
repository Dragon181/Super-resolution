from tqdm.auto import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from pathlib import Path

from model import SRCNN

zoom_factor=4
image='BSDS300\\images\\test'
for path in tqdm(list(Path('BSDS300\\images\\test').iterdir())):
    
    img = Image.open(path).convert('YCbCr')
    y, cb, cr = img.split()
    img = img.resize((int(img.size[0]*zoom_factor), int(img.size[1]*zoom_factor)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
    img.save(f'{path.stem}_bicubic.jpg')


    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel
    
    device = torch.device("cpu")
    model = SRCNN()
    model.load_state_dict(torch.load('model_21.pth'))
    model = model.eval().to(device)

    input = input.to(device)
    with torch.no_grad():
        out = model(input)
        out = out.cpu()

    out_img_y = out[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img = Image.merge('YCbCr', [out_img_y, cb.resize(out_img_y.size), cr.resize(out_img_y.size)]).convert('RGB')
    
    out_img.save(f'{path.stem}_zoomed.jpg')
    break