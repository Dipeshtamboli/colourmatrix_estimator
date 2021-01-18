import numpy as np
import os
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


class Dataset_from_text(Dataset):
	def __init__(self, txt_path='filepath'):
		self.images = list(open(txt_path))
		self.transforms = transforms.Compose([transforms.ToTensor()])
		
	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		path = self.images[index]
		img = Image.open( "../patches/"+path.split(',')[0])
		# exit()

		img = img.convert('RGB')
		img = self.transforms(img)
		w_given = path.strip().split(',')[1:]

		# f'{a[0][0]},{a[0][1]},{a[1][0]},{a[1][1]},{a[2][0]},{a[2][1]}\n'
		size = img.shape[1]
		w11, w12, w13 = 0, 0, 0
		w21, w22, w23 = float(w_given[1]), float(w_given[3]), float(w_given[5])
		w = np.array([[w11, w12, w13],[w21, w22, w23]])
		# h = np.random.rand(2, size*size)
		h = np.ones((2, size*size))
		image = (w.T@h).reshape(3,size,size)
		image_E = 255*np.exp(-image)
		image_E = torch.tensor(image_E)

		w11, w12, w13 = float(w_given[0]), float(w_given[2]), float(w_given[4])
		w21, w22, w23 = 0, 0, 0		
		w = np.array([[w11, w12, w13],[w21, w22, w23]])
		# h = np.random.rand(2, size*size)
		h = np.ones((2, size*size))
		image = (w.T@h).reshape(3,size,size)
		image_H = 255*np.exp(-image)		
		image_H = torch.tensor(image_H)
		targets = [torch.tensor(float(elem)) for elem in path.strip().split(',')[1:]]
		w = torch.tensor(targets)
		# w = np.array([[w11, w12, w13],[w21, w22, w23]])
		# h = torch.ones((1, size*size))
		# w = torch.transpose(w.reshape(1,6), 0, 1)
		# w_channels = torch.matmul(w.transpose,h).reshape(3,size,size).transpose(2,1,0)

		# w_channels = (w.reshape(6,1)@h)
		# w_channels = w_channels.reshape(6,size,size)

		return torch.cat((img, image_H, image_E),0), targets



if __name__ == '__main__':

	mask_transforms = transforms.Compose([transforms.ToTensor()])

	train_dataset = Dataset_from_text(txt_path='../patches_and_w.csv')

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
											   batch_size=4,
											   shuffle=False, 
											   num_workers=1)

	for inputs, targets in train_dataloader:
		print(inputs.shape)
		targets = torch.transpose(torch.cat(targets).reshape(6,inputs.shape[0]),1,0) 
		print(targets.shape)
		pdb.set_trace()
		exit()