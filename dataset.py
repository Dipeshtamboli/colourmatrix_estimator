import os
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

class SegDatasetTrain(Dataset):
    def __init__(self, txt_path='filepath', transforms_=None, mask_transforms_=None):
        self.transforms = transforms_
        self.mask_transforms = mask_transforms_        
        self.samples = list(open(txt_path))
        images = list(open(txt_path))
        classes = list(set([image.split('/')[-2] for image in images]))
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}  
        self.img_target = [(image, class_to_idx[image.split('/')[-2]]) for image in images]      
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, target = self.img_target[index]
        path = self.samples[index]
        img = Image.open(path.strip())
        img = img.convert('RGB')
        img = self.transforms(img)
        mask_path = path.split('/')[-2] + '/'+path.split('/')[-1].split('.')[-2] + '_2.jpg'
        mask_path = '../masks/' + mask_path
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path.strip()) 
        else:
            # print("path not found for:\n{}\nloading the same img".format(path))
            mask = Image.open(path.strip())
            mask = mask.convert('L')        
        # mask = Image.open(mask_path.strip()) 
        mask = self.mask_transforms(mask)
        return img,target, mask, path

class SegDatasetTest(Dataset):
    def __init__(self, txt_path='filepath', transforms_=None, mask_transforms_=None):
        self.transforms = transforms_
        self.mask_transforms = mask_transforms_        
        self.samples = list(open(txt_path))
        images = list(open(txt_path))
        classes = list(set([image.split('/')[-2] for image in images]))
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}  
        self.img_target = [(image, class_to_idx[image.split('/')[-2]]) for image in images]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, target = self.img_target[index]        
        path = self.samples[index]
        img = Image.open(path.strip())
        img = img.convert('RGB')
        img = self.transforms(img)
        return img,target, path

class skinai_dataset(Dataset):
    def __init__(self, txt_path='filepath', transforms_=None):
        self.transforms = transforms_
        images = list(open(txt_path))
        classes = list(set([image.split('/')[-2] for image in images]))
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.samples = [(image, class_to_idx[image.split('/')[-2]]) for image in images]

    def __getitem__(self, index):
        path, target = self.samples[index]
        # print(path)
        img = Image.open(path.strip())
        img = img.convert('RGB')
        sample = self.transforms(img)
        return sample, target, path

    def __len__(self):
        return len(self.samples)

class binary_mask_dataset(Dataset):
    def __init__(self, mask_type="binary_mask",txt_path='filepath', transforms_=None, mask_transforms_=None):
        self.mask_type=mask_type
        self.transforms = transforms_
        self.mask_transforms = mask_transforms_
        images = list(open(txt_path))
        classes = list(set([image.split('/')[-2] for image in images]))
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.samples = [(image, class_to_idx[image.split('/')[-2]]) for image in images]

    def __getitem__(self, index):
        path, target = self.samples[index]
        # pdb.set_trace()
        img = Image.open(path.strip())
        img = img.convert('RGB')
        sample = self.transforms(img)
        if path.split('/')[-3] == 'MODEL':
            # print(path.split('MODEL'))
            # print(path.strip())
            path = path.strip()
            mask_path = ((path.split('MODEL')[0]+"MASK/{}s".format(self.mask_type)+path.split('MODEL')[1])[:-4]+'_2.jpg')
            # mask_path = '../data/MASK/binary_masks/121/9493-121-2_2.jpg'
            # print(mask_path.strip())
            mask = Image.open(mask_path.strip()) 

            # mask = Image.open(path.strip()) 
            mask = self.mask_transforms(mask)
            return sample, target, path, mask

        return sample, target, path, None

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    val_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = binary_mask_dataset(mask_type="binary_mask",txt_path='../classification_dataset/train_dataset_dummy.txt',
                      transforms_=val_transforms, mask_transforms_=mask_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=1,
                                               shuffle=True, 
                                               num_workers=4)

    for inputs, labels, _, mask in train_dataloader:
        print(inputs.shape)
        print(mask.shape)
        break