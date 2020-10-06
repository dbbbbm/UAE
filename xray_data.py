import pandas
import torch
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils import data

DATA_PATH = '/media/user/disk/'

def read_data():
    path = DATA_PATH + 'rsna-pneumonia-detection-challenge/'
    sub_path = 'stage_2_train_images/'
    csv = pandas.read_csv(path + 'stage_2_detailed_class_info.csv')
    class_label = {'Normal': 0,
                   'No Lung Opacity / Not Normal': 1, 'Lung Opacity': 2}
    patient_dict = {}
    for i in range(csv.shape[0]):
        name = csv['patientId'][i]
        target = csv['class'][i]
        target = class_label[target]
        patient_dict[name] = target

    for file_name in tqdm(os.listdir(path + sub_path)):
        patient = file_name.split('.')[0]
        target = patient_dict[patient]
        if target == 0:
            os.system('cp {}{}{} {}{}'.format(
                path, sub_path, file_name, path, 'normal/'))
        if target == 1:
            os.system('cp {}{}{} {}{}'.format(
                path, sub_path, file_name, path, 'not_normal/'))
        if target == 2:
            os.system('cp {}{}{} {}{}'.format(
                path, sub_path, file_name, path, 'lung_opacity/'))


class Xray(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(Xray, self).__init__()
        self.transform = transform
        self.file_path = []
        self.labels = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        for label in os.listdir(main_path):
            if label not in ['0', '1']:
                continue
            for file_name in tqdm(os.listdir(main_path+'/'+label)):
                data = sitk.ReadImage(main_path+'/'+label + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze()
                img = Image.fromarray(data).convert('L').resize((img_size,img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(int(label))
        

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.slices)


def get_xray_dataloader(bs, workers, dtype='train', img_size=64, dataset='rsna'):
    if dtype == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if dataset == 'rsna':
        path = DATA_PATH + 'rsna-pneumonia-detection-challenge/'
    elif dataset == 'pedia':
        path = DATA_PATH + 'pediatric/'

    if dtype == 'train':
        path += 'training'
    elif dtype == 'valid':
        path += 'validation'
    elif dtype == 'test':
        path += 'testing'
    dset = Xray(main_path=path, transform=transform, img_size=img_size)
    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader


if __name__ == '__main__':
    read_data()
