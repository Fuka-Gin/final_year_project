import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


class UcGANDataset(Dataset):

    def __init__(self, root_dir, mode="train", img_size=256, val_split=0.2, paired=True):

        self.mode = mode
        self.img_size = img_size
        self.paired = paired

        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        input_files = sorted(os.listdir(self.input_dir))
        target_files = sorted(os.listdir(self.target_dir))

        if paired:
            # Paired dataset (deblur, lowlight)
            files = sorted(set(input_files) & set(target_files))
            train, val = train_test_split(files, test_size=val_split, random_state=42)

            self.input_files = train if mode == "train" else val
            self.target_files = self.input_files

        else:
            # Unpaired dataset (artistic)
            train_in, val_in = train_test_split(input_files, test_size=val_split, random_state=42)
            train_tg, val_tg = train_test_split(target_files, test_size=val_split, random_state=42)

            self.input_files = train_in if mode == "train" else val_in
            self.target_files = train_tg if mode == "train" else val_tg

        self.base_transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        )

    def __len__(self):
        return max(len(self.input_files), len(self.target_files))

    def __getitem__(self, idx):

        inp_name = self.input_files[idx % len(self.input_files)]
        tgt_name = self.target_files[random.randint(0, len(self.target_files)-1)]

        inp = Image.open(os.path.join(self.input_dir, inp_name)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, tgt_name)).convert("RGB")

        if self.mode == "train":

            if random.random() > 0.5:
                inp = TF.hflip(inp)
                tgt = TF.hflip(tgt)

            inp = self.color_jitter(inp)

        return self.base_transform(inp), self.base_transform(tgt)