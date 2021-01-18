import torch
from fullnet import FullNet
# from model import Net
from dataset import ColorNormDataset
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import sys

dataset_txt = sys.argv[1]

writer = SummaryWriter(f'runs/{dataset_txt}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128 

dataset = ColorNormDataset(dataset_txt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataset))

net = FullNet(3, output_channels=3, dilations=[1,2,2,1], n_layers=3, growth_rate=3).to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

net.train()
iteration = 0
for i in range(2000):
    for images, images_jitter in dataloader:
        iteration += 1
        optimizer.zero_grad()
        images = images.to(device)
        images_jitter = images_jitter.to(device)
        outs = net(images_jitter)
        recons_images = (images_jitter-outs)
        loss = criterion(images, recons_images)
        loss.backward()
        optimizer.step()
        print(loss/batch_size)
        writer.add_scalar('TRN LOSS / ITERATION',loss, iteration) 
        if iteration % 20 == 0: 
            torch.save(net.state_dict(), f'checkpoints/{dataset_txt}_{iteration}.pt')
    if iteration > 5000: break
