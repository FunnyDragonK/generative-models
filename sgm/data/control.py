import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, root):
        self.size = 256
        self.data = []
        self.root = root
        with open(os.path.join(self.root, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread(self.root + source_filename)
        source = np.load(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)
        org_h, org_w, _ = target.shape
        # resize short edge to self.size
        if org_h < org_w:
            new_h = self.size
            new_w = int(org_w / org_h * self.size)
        else:
            new_w = self.size
            new_h = int(org_h / org_w * self.size)
        target = cv2.resize(target, (new_w, new_h))
        # resize hint
        source = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # random crop to self.sizexself.size
        h, w, _ = target.shape
        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)
        target = target[top:top + self.size, left:left + self.size, :]
        source = source[top:top + self.size, left:left + self.size]

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0
        source = source.astype(np.float32)
        # add one dimension at last
        source = np.expand_dims(source, axis=-1)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        example = dict()
        example["jpg"] = torch.from_numpy(target.transpose(2, 0, 1)).to("cuda")
        example["txt"] = prompt
        example["hint"] = torch.from_numpy(source.transpose(2, 0, 1)).to("cuda")
        example["original_size_as_tuple"] = torch.from_numpy(np.array([org_h, org_w])).to("cuda")
        example["crop_coords_top_left"] = torch.from_numpy(np.array([top, left])).to("cuda")
        example["target_size_as_tuple"] = torch.from_numpy(np.array([self.size, self.size])).to("cuda")
        return example

