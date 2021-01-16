import os
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


class Dataset_from_text(Dataset):
    def __init__(self, txt_path='filepath'):
        self.samples = list(open(txt_path))
        self.images = list(open(txt_path))
        self.transforms = transforms.Compose([transforms.ToTensor()])
        # print(images)
        # classes = list(set([image.split('/')[-2] for image in images]))
        # classes.sort()
        # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.img_target = [(image, class_to_idx[image.split('/')[-2]]) for image in images]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.images[index]
        print(path.strip())
        # pdb.set_trace()
        # exit()
        img = Image.open( "patches/"+path.strip())
        img = img.convert('RGB')
        img = self.transforms(img)
        return img, path



if __name__ == '__main__':

    mask_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = Dataset_from_text(txt_path='patches_path.txt')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=1,
                                               shuffle=True, 
                                               num_workers=4)

    for inputs, path in train_dataloader:
        print(inputs.shape)
        print(path)
        break