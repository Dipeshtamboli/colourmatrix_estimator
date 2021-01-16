import os
from torch.utils.data import DataLoader
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet152

class FC_48_to_6(nn.Module):
    def __init__(self, input_dim):
        super(FC_48_to_6, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, input_dim//2), 
            nn.BatchNorm1d(input_dim//2, momentum=0.01),
            nn.ReLU(),
            nn.Linear(input_dim//2,6), 
            # nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # x = torch.cat((x.view(x.size(0), -1), semantic.view(semantic.size(0), -1)), -1)
        x = self.fc_out(x)
        x = x.view(x.size(0), -1)
        return (x)

if __name__ == '__main__':
    os.environ['CUDA_VIhdjkfe hkrewgk.kerwSIBLE_DEVICES'] = ""
    fc_model = FC_48_to_6(48)
    # fc_model.to('cuda')

    # inputs = inputs.permute(48)
    input = Variable(torch.FloatTensor(10,48))
    # image_sequences = Variable(inputs.to("cuda"), requires_grad=True)        
    # print("image_sequences.shape:",image_sequences.shape)
    out = fc_model(input)
    print("out.shape:",out.shape)
        
