from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

# warnings.filterwarnings('ignore')

plt.ion()

# data source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].values
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.01)


# override __len__ and __getitem__
class FaceLandmarksDataset(Dataset):
    '''Face Landmarks dataset'''
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file(str): path to csv file
            root_dir(str): directory with all the images
            transform(callable, optional): optional transform to be applied on a sample
        '''
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        print(img_name)
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype(np.float32).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show(block=True)
        break


class Rescale:
    '''Rescale the image to a given size

    Args:
        output_size (tuple or int): If int, smaller of images edge is matched
            to output_size.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image,
                                 (new_h, new_w),
                                 mode='reflect',
                                 anti_aliasing=True
                                 )

        # h and w are swapped for landmarks because
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': image, 'landmarks': landmarks}


class RandomCrop:
    '''Crop randomly the image in a sample

    Args:
        output_size (tuple or int):  desired output_size.
            If int, square crop is made.
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']

        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        landmarks = torch.from_numpy(landmarks)

        return {'image': image, 'landmarks': landmarks}


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    tsfrm_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**tsfrm_sample)

plt.show(block=True)

tsfrm_composed = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=tsfrm_composed
                                           )

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break


if __name__ == '__main__':
    dataloader = DataLoader(transformed_dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=1)

    def show_landmarks_batch(sample_batched):
        images_batch = sample_batched['image']
        landmarks_batch = sample_batched['landmarks']

        batch_size = len(images_batch)
        im_size = images_batch.size(2)

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                        landmarks_batch[i, :, 1].numpy(),
                        s=10,
                        marker='.',
                        c='r'
                        )

            plt.title('Batch from dataloader')


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show(block=True)
            break
