import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class ClassifyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_I = os.path.join(opt.dataroot, opt.phase)
        # self.dir_I = os.path.join(opt.dataroot, 'train')

        self.I_paths = sorted(make_dataset(self.dir_I))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        I_path, c = self.I_paths[index % len(self.I_paths)]

        I = Image.open(I_path).convert('RGB')
        I = I.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        I = self.transform(I)

        w = I.size(2)
        h = I.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        I = I[:, h_offset:h_offset + self.opt.fineSize,
                 w_offset:w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(I.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            I = I.index_select(2, idx)

        return {'I': I, 'C': c,
                'I_paths': I_path}

    def __len__(self):
        return len(self.I_paths)

    def name(self):
        return 'ClassifyDataset'
