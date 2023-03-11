from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import os
import requests
import tarfile
import shutil





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

CLASSES_TO_IDX = {
    'tench': 0,
    'English springer': 1,
    'cassette player': 2,
    'chain saw': 3,
    'church': 4,
    'French horn': 5,
    'garbage truck': 6,
    'gas pump': 7,
    'golf ball': 8,
    'parachute': 9
}


class Imagenette(Dataset):

    def __init__(self, path: str, train: bool):
        self.path = path
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        if train:
            self.path += '/train'
        else:
            self.path += '/val'

        self.images = []
        self.labels = []

        for label, folder in IMAGENETTE_CLASSES.items():
            label_idx = CLASSES_TO_IDX[label]
            for image in os.listdir(f'{self.path}/{folder}'):
                self.images.append(f'{self.path}/{folder}/{image}')
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image with PIL
        image = Image.open(self.images[index]).convert('RGB')
        tensor = self.transform(image)
        image.close()
        return tensor, self.labels[index]


dataset_train = Imagenette('imagenette2', train=True)
dataset_val = Imagenette('imagenette2', train=False)

if __name__ == '__main__':

    if not (os.path.exists("imagenette2")):
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        target_path = 'imagenette2.tar.gz'

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())

        shutil.rmtree('imagenette2-160', ignore_errors=True)
        tar = tarfile.open(target_path, "r:gz")
        tar.extractall()
        tar.close()

        os.rename("imagenette2-160","imagenette2")

    print(len(dataset_train))
    print(dataset_train[0][0].shape, dataset_train[0][1])

    print(len(dataset_val))
    print(dataset_val[0][0].shape, dataset_val[0][1])
