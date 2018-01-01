import torch
import os
from torch.utils import data
import random
from torchvision import transforms
from PIL import Image


class DataSetDefine(data.Dataset):
    """
    base_tensor, angle_tensor, diff_tensor
    """
    def __init__(self, img_list, label_tensor_list, transforms, config):
        self.img_list = img_list
        self.label_tensor_list = label_tensor_list
        self.transforms = transforms
        self.cfg = config

    def __getitem__(self, index):
        img_th = self.get_img_th(self.img_list[index])
        label_th = self.label_tensor_list[index]

        return img_th, label_th

    def get_img_th(self, img_path):
        """
        :param img_path: .jpg
        :return: tensor (channel, height, width)
        """

        with Image.open(img_path) as img:
            img = img.convert('RGB')

        if img is None:
            print(img_path)
            return torch.zeros(3, 200, 200)

        # img = cv2.resize(img, dsize=(self.cfg.width, self.cfg.height))
        img_th = self.transforms(img)

        return img_th

    def __len__(self):
        len_0 = len(self.img_list)
        len_1 = len(self.label_tensor_list)

        assert len_0 == len_1
        return len_0


class DataSet(object):
    def __init__(self, config):
        """
        :param config: config parameters
        :param parse: parse origin images to torch tensor and save to .pkl
        :param load: load torch tensor from .pkl
        """
        self.cfg = config

        self.train_image_list = []
        self.train_label_tensor_list = []

        self.test_image_list = []
        self.test_label_tensor_list = []

        self.train_transforms = transforms.Compose([
            transforms.Resize(size=(self.cfg.height, self.cfg.width)),
            transforms.CenterCrop(size=(self.cfg.crop_height, self.cfg.crop_width)),
            transforms.ToTensor(),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(size=(self.cfg.crop_height, self.cfg.crop_width)),
            transforms.ToTensor(),
        ])

        self.load_list(data_type='train')
        self.load_list(data_type='test')

        self.train_dataset = DataSetDefine(self.train_image_list,
                                           self.train_label_tensor_list,
                                           self.train_transforms,
                                           self.cfg)

        self.test_dataset = DataSetDefine(self.test_image_list,
                                          self.test_label_tensor_list,
                                          self.test_transforms,
                                          self.cfg)

        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.cfg.train_batch_size,
                                            shuffle=True,
                                            num_workers=0)

        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.cfg.test_batch_size,
                                           shuffle=False,
                                           num_workers=0)

    def load_list(self, data_type='train'):
        assert data_type in ['train', 'test']

        if data_type == 'train':
            info_path = self.cfg.train_info
        else:
            info_path = self.cfg.test_info

        with open(info_path) as fp:
            line_list = fp.readlines()

        for line in line_list:
            line = line.strip()
            info_list = line.split()
            name = info_list[0]
            au_list = info_list[1:]

            label_list = [int(au) for au in au_list]

            if data_type == 'train':
                self.train_image_list.append(os.path.join(self.cfg.image_dir, name))
                self.train_label_tensor_list.append(torch.LongTensor(label_list))
            else:
                self.test_image_list.append(os.path.join(self.cfg.image_dir, name))
                self.test_label_tensor_list.append(torch.LongTensor(label_list))


if __name__ == '__main__':
    import config
    obj = DataSet(config)

    for img, label in obj.train_loader:
        print(img.size())
