import numpy as np
import torch.utils.data as data

'''dataset'''


class STS_Dataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=np.squeeze(image), mask=label)
            image = data['image']
            label = data['mask']

        return image, label

class STS_test_Dataset(data.Dataset):
    def __init__(self, images, transform = None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        data = self.transform(image=image)
        image = data['image']
        return image