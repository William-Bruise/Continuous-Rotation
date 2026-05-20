import glob
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from .transforms import random_crop, resize


class DIV2KASISRDataset(Dataset):
    def __init__(self, root, split='train', patch_size=96, query_points=2048, scale_min=1.0, scale_max=4.0, interp='bicubic'):
        self.root = root
        self.patch_size = patch_size
        self.query_points = query_points
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interp = interp
        self.files = sorted(glob.glob(os.path.join(root, f'{split}_hr', '*')))

    def __len__(self):
        return max(1, len(self.files))

    def _load(self, idx):
        if len(self.files) == 0:
            return torch.rand(3, self.patch_size * 2, self.patch_size * 2)
        img = read_image(self.files[idx % len(self.files)]).float() / 255.0
        return img[:3]

    def __getitem__(self, idx):
        hr = random_crop(self._load(idx), self.patch_size)
        scale = random.uniform(self.scale_min, self.scale_max)
        lr_h = max(4, int(hr.shape[1] / scale))
        lr_w = max(4, int(hr.shape[2] / scale))
        lr = resize(hr, (lr_h, lr_w), self.interp)

        q = torch.rand(self.query_points, 2)
        q[:, 0] = q[:, 0] * 2 - 1
        q[:, 1] = q[:, 1] * 2 - 1
        grid = q.view(1, -1, 1, 2)
        gt = torch.nn.functional.grid_sample(hr.unsqueeze(0), grid, mode='bilinear', align_corners=False).squeeze(0).squeeze(-1).t()
        cell = torch.full((self.query_points, 2), 2.0 / hr.shape[2])
        return {'lr_image': lr, 'query_coords': q, 'gt_rgb': gt, 'cell': cell, 'scale': torch.tensor([scale], dtype=torch.float32), 'full_hr_patch': hr}
