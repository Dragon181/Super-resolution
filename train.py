from math import log10
import torchvision.utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from dataset import DatasetFromFolder
from model import SRCNN

device = torch.device("cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 50
NUM_WORKERS = 0
num_epoch = 1
zoom_factor = 4

trainset = DatasetFromFolder('BSDS300\\images\\train', zoom_factor=zoom_factor)
testset = DatasetFromFolder('BSDS300\\images\\test', zoom_factor=zoom_factor)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE, 
    sampler=RandomSampler(trainset, replacement=True, num_samples=32 * BATCH_SIZE),
    num_workers=NUM_WORKERS
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE, 
    sampler=RandomSampler(testset, replacement=True, num_samples=1024),
    num_workers=NUM_WORKERS
)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

def get_psnr(mse):
    return -10 * log10(mse)

def test():
    avg_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in tqdm(testloader, desc='test', leave=False):
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                out = model(input)
                loss = criterion(out, target)
                avg_loss += loss.item()
    return get_psnr(avg_loss / len(testloader))

def train():
    epoch_loss = 0
    model.train()
    for input, target in tqdm(trainloader, desc='train', leave=False):
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        out = model(input)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return get_psnr(epoch_loss / len(trainloader))

train_stat = []
test_stat = []
with tqdm(range(num_epoch)) as bar:
    for epoch in bar:
        # Train
        train_psnr = train()
        test_psnr = test()
        bar.set_postfix({'epoch': epoch, 'train_psnr': f'{train_psnr:.2f}', 'test_psnr': f'{test_psnr:.2f}'})
        train_stat.append(train_psnr)
        test_stat.append(test_psnr)

        # Save model
        torch.save(model.state_dict(), f"model_{epoch}.pth")
    plt.plot(train_stat, label='train_psnr')
    plt.plot(test_stat, label='test_psnr')
    plt.legend()
    plt.show()