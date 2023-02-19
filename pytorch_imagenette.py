from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

IMAGENETTE_CLASSES = {
    'tench': 'n01440764',
    'English springer': 'n02102040',
    'cassette player': 'n02979186',
    'chain saw': 'n03000684',
    'church': 'n03028079',
    'French horn': 'n03394916',
    'garbage truck': 'n03417042',
    'gas pump': 'n03425413',
    'golf ball': 'n03445777',
    'parachute': 'n03888257'
}


class Imagenette(Dataset):

    def __init__(self, path: str, train: bool, transform=transforms.ToTensor()):
        self.path = path
        self.train = train
        self.transform = transform

        if train:
            self.path += '/train'
        else:
            self.path += '/val'

        self.images = []
        self.labels = []

        for label, folder in IMAGENETTE_CLASSES.items():
            for image in os.listdir(f'{self.path}/{folder}'):
                self.images.append(f'{self.path}/{folder}/{image}')
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image with PIL
        image = Image.open(self.images[index])
        tensor = self.transform(image)
        return tensor, self.labels[index]


# Tests
if __name__ == '__main__':
    dataset_train = Imagenette('imagenette2', train=True)
    print(len(dataset_train))
    print(dataset_train[0][0].shape, dataset_train[0][1])

    dataset_val = Imagenette('imagenette2', train=False)
    print(len(dataset_val))
    print(dataset_val[0][0].shape, dataset_val[0][1])
