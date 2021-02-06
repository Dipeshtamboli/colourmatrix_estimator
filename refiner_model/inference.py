from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.nn as nn
import pdb
import sys
sys.path.insert(0,'../')
from torch.autograd import Variable
from net import FC_48_to_6
import torch
from net_refiner import FullNet
# from model import Net
from dataset_refiner import Dataset_from_text
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import sys

# exp_name = sys.argv[1]
exp_name = "inference"
# window_size = int(sys.argv[2])
window_size = 64

writer = SummaryWriter(f'training_data/runs/{exp_name}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 5

# dataset = Dataset_from_text(exp_name)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# train_dataset = Dataset_from_text(txt_path='../abhi_mat_w_with_path.csv')
train_dataset = Dataset_from_text(txt_path='/home/abhishek/w_matrices_0.csv')

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=False, 
                                           num_workers=1)

print(len(train_dataset))

fc_model = FC_48_to_6(48)
fc_model = fc_model.to(device)

net = FullNet(9, output_channels=3, dilations=[1,2,2,1], n_layers=3, growth_rate=3).to(device)
# criterion = torch.nn.L1Loss()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

iteration = 0

path_trained_w_est = "tanpure_csv_w_est.pth"
checkpoint = torch.load(path_trained_w_est)
fc_model.load_state_dict(checkpoint['fc_model_state_dict'])
fc_model.eval()

# refiner_model_path = "training_data/tanpure_refiner_epoch_0.pth"
refiner_model_path = "training_data/L1 loss added_20_0.3761241137981415.pth"
checkpoint2 = torch.load(refiner_model_path)
net.load_state_dict(checkpoint2)
net.eval()
# decoder_net.load_state_dict(checkpoint['decoder_net_state_dict'])
# optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# input = Variable(torch.FloatTensor(10,48)).to(device)
input = (torch.FloatTensor(10,48)).to(device)
out = fc_model(input)
if out.shape:
    print("#"*5,"model imported succesfully","#"*5)
    print("\t\t out.shape:",out.shape)

# exit()
# cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
m = nn.AvgPool2d((window_size, window_size), stride=(window_size, window_size))

# running_loss_train = 0
set_num = 0
for images, targets in train_dataloader:
    set_num += 1
    targets = torch.transpose(torch.cat(targets).reshape(6,images.shape[0]),1,0) 
    # iteration += 1
    # optimizer.zero_grad()
    # images[:,3:,:,:] = images[:,3:,:,:]*255.0
    images = images.to(device).type(FloatTensor)
    targets = targets.to(device)
    # images_jitter = images_jitter.to(device)
    outs = net(images)
    # images[:,3:,:,:] = images[:,3:,:,:]/255.0
    # outs = net(images[:,3:,:,:]*255.0)

    out_48 = m(outs).resize(outs.shape[0],int((outs.shape[3]/window_size)**2*3))
    w_6_dim = fc_model(out_48)

    all_img = torch.cat((images, outs),1)
    all_img = torch.reshape(all_img, (batch_size*4,3,256,256)) 
    grid_imgs = make_grid(all_img, 4)
    save_image(grid_imgs, f'L1_tanpure_output_{set_num}.png')
    # pdb.set_trace()
    

    # recons_images = (images_jitter-outs)
    # pdb.set_trace()
    # loss = criterion(w_6_dim, targets)
    # loss.backward()
    # optimizer.step()
    # running_loss_train += loss.item() * images.size(0)
    if set_num == 5:
        break
# epoch_loss_train = running_loss_train / len(train_dataset)
print(f'Output W: {w_6_dim}\nGroundTruth W: {targets}')
# writer.add_scalar('L2 loss on w',loss, iteration) 
# if epoch % 20 == 0: 
#     torch.save(net.state_dict(), f'training_data/{exp_name}_{epoch}.pth')

