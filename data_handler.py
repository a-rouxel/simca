import os
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import random
from pytorch_lightning import LightningDataModule


class CubesDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.augment_ = augment
        self.data_file_names = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):

        cube, wavelengths = self.load_hyperspectral_data(idx) # H x W x lambda

        if self.augment_:
            cube = self.augment(cube, 128) # lambda x H x W
        else:
            cube = torch.from_numpy(np.transpose(cube, (2, 0, 1))).float()[:,:128,:128] # lambda x H x W
        
        return cube, wavelengths

    def load_hyperspectral_data(self, idx):
        file_path = os.path.join(self.data_dir, self.data_file_names[idx])
        data = sio.loadmat(file_path)
        if "img_expand" in data:
            cube = data['img_expand'] / 65536.
        elif "img" in data:
            cube = data['img'] / 65536.
        cube = cube.astype(np.float32) # H x W x lambda
        wavelengths = torch.tensor(np.linspace(450, 650, 28))

        return cube, wavelengths
    
    def augment(self, img, crop_size = 128):
        h, w, _ = img.shape
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data = np.zeros((crop_size, crop_size, 28), dtype=np.float32)
        processed_data = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).float()
        processed_data = arguement_1(processed_data)

        """ # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0) """
        return processed_data

    
class CubesDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = CubesDataset(self.data_dir,augment=True)

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)


def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],generate_gt.shape[2],generate_gt.shape[3]
    divid_point_h = h//2
    divid_point_w = w//2
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

# class AcquisitionDataset(Dataset):
#     def __init__(self, input, hs_cubes, transform=None, target_transform=None):
#         """_summary_

#         Args:
#             input (_type_): List of size 2 with each element being a list:
#                 - First list: List of n torch.tensor acquisitions (2D)
#                 - Second list: List of n int labels
#             hs_cubes (_type_): List of size m, hs_cubes[m] being the m-th hs cube
#             transform (_type_, optional): _description_. Defaults to None.
#             target_transform (_type_, optional): _description_. Defaults to None.
#         """
#         self.data = input # list of size 2, first elem is a list of n torch.tensor acquisitions (input), second elem is a list of size n with the index of corresponding hs cubes (output)
#         self.labels = self.data[1]

#         self.cubes = hs_cubes # list of cubes, number of cubes must be >= max(self.labels)
        
#         self.transform = transform
#         self.target_transform = target_transform
        
#     def __len__(self):
#         return len(self.data[1])

#     def __getitem__(self, index):
#         acq = self.data[0][index] # torch tensor of size x*y
#         cube = self.cubes[self.labels[index]] # torch tensor of size x*y*w

#         return acq, cube

if __name__ == "__main__":
    data_dir = "/local/users/ademaio/lpaillet/mst_datasets/cave_1024_28/"
    datamodule = CubesDataModule(data_dir, batch_size=5, num_workers=2)