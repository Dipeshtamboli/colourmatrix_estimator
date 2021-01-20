# py train.py "first try" "fc" "4" "1e-2" 10 1
import pdb
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from net import FC_48_to_6
from dataset import Dataset_from_text



# py train.py "w_est_on_abhi_mat" 64 64 "1e-2" 100 "1e-5"
logs_name = str(sys.argv[1])
window_size = int(sys.argv[2])
batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_epochs = int(sys.argv[5])
weight_decay = float(sys.argv[6])
print(f"logs_name: {logs_name}\nwindow_size: {window_size}x{window_size}\nbatch_size: {batch_size}\nweight_decay: {weight_decay}\nlearning_rate: {learning_rate}\nnum_epochs: {num_epochs}\n")
writer = SummaryWriter(f'runs/{logs_name}')
directory = f'./training_logs/{logs_name}'
if not os.path.exists(directory):
	os.makedirs(directory)



train_dataset = Dataset_from_text(txt_path='abhi_mat_w_with_path.csv')
val_dataset = train_dataset

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
											   batch_size=batch_size,
											   shuffle=True, 
											   num_workers=4)

val_dataloader = torch.utils.data.DataLoader(train_dataset, 
											   batch_size=batch_size,
											   shuffle=True, 
											   num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fc_model = FC_48_to_6(48)

fc_model = fc_model.to(device)
criterion_MSE = nn.MSELoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, fc_model.parameters()),
							lr=learning_rate, 
							weight_decay=weight_decay)

load_model =0
PATH = '../models/epoch:0_valAcc:0.0.pth'
if load_model:
	checkpoint = torch.load(PATH)
	fc_model.load_state_dict(checkpoint['fc_model_state_dict'])
	# decoder_net.load_state_dict(checkpoint['decoder_net_state_dict'])
	optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']



def train_model(model, criterion, optimizer): 
	for epoch in range(num_epochs):
		print(f'Epoch {epoch}/{num_epochs}')
		print('-' * 10)

		model.train()  
		running_loss_train = 0.0

		running_loss_val = 0.0

		for inputs, labels in train_dataloader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			m = nn.AvgPool2d((window_size, window_size), stride=(window_size, window_size))

			avg_input = m(inputs).resize(inputs.shape[0],int((inputs.shape[3]/window_size)**2*3))

			outputs = model(avg_input)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss_train += loss.item() * inputs.size(0)
			epoch_loss_train = running_loss_train / len(train_dataset)
		writer.add_scalar('TRN LOSS', epoch_loss_train, epoch)

		model.eval()
		val_labels = []
		val_preds = []
		with torch.no_grad():
			for inputs, labels in val_dataloader:
				inputs = inputs.to(device)
				labels = labels.to(device)
				avg_input = m(inputs).resize(inputs.shape[0],int((inputs.shape[3]/window_size)**2*3))
				outputs= model(avg_input)

				val_labels += labels.cpu().numpy().tolist()

				loss = criterion(outputs, labels)
				running_loss_val += loss.item() * inputs.size(0)
				epoch_loss_val = running_loss_val / len(val_dataset)
			writer.add_scalar('VAL LOSS', epoch_loss_val, epoch)
			# torch.save(model.state_dict(), f'./training_logs/{logs_name}/{epoch}_model_weights')
		print(f'Train Loss: {epoch_loss_train:.4f}  Val Loss: {epoch_loss_val:.4f} ')

		if not os.path.exists(f'./training_logs/{logs_name}/models_v2'):
			os.makedirs(f'./training_logs/{logs_name}/models_v2')            
		save_path = f"./training_logs/{logs_name}/models_v2/epoch:{epoch}_val_loss:{epoch_loss_val}.pth"
		torch.save({
			'epoch': epoch,
			'fc_model_state_dict': model.state_dict(),
			# 'decoder_net_state_dict': decoder_net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			'train_loss':epoch_loss_train,
			'val_loss':epoch_loss_val,
			}, save_path)

model_ft = train_model(fc_model, criterion_MSE, optimizer_ft)