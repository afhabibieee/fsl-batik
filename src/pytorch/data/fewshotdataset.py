# Some code adapted from the easyfsl library
# Copyright (c) Bennequin, E (Sicara).
# Licensed under the MIT License
# Original repository: https://github.com/sicara/easy-few-shot-learning

import os
import sys
import json
import random
import warnings
import torch
from pathlib import Path
import cv2 as cv
from cvtorchvision import cvtransforms as T
from torch.utils.data import Dataset, Sampler
import collections

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_path not in sys.path:
    sys.path.append(current_path)
from configs import IMAGENET_MEAN, IMAGENET_STD

collections.Iterable = collections.abc.Iterable

class ImageDataset(Dataset):
    def __init__(
        self,
        specs_file,
        image_size,
        transform=None,
        training=True,
        formats=None
    ):
        """
        Initialize ImageDataset.

        Args:
            specs_file (str): Path to the JSON file containing specifications.
            image_size (int): The size of the image (width and height).
            transform (callable, optional): A function/transform to be applied to each image.
            training (bool, optional): Whether the dataset is used for training or not.
            formats (set, optional): Set of image file formats to consider. Defaults to None.
        """
        specs = self.load_specs(Path(specs_file))

        self.images, self.labels = self.list_data_instances(specs['class_roots'], formats)
        self.class_names = specs['class_names']

        self.transform = (transform if transform else self.set_transform(image_size, training))

    @staticmethod
    def load_specs(specs_file):
        """
        Load specifications from a JSON file
            
        Args:
            specs_file (str): Path to the Json file.
            
        Returns:
            dict: Specifications loaded from the JSON file.
        """
        with open(specs_file, 'r') as file:
            specs = json.load(file)
            
        if 'class_names' not in specs.keys() or 'class_roots' not in specs.keys():
            raise ValueError('Requires specs in a JSON with the keys class_names and class_roots')
        if len(specs['class_names']) != len(specs['class_roots']):
            raise ValueError('The number of class names does not match the number of directories')

        return specs

    @staticmethod
    def list_data_instances(class_roots, formats):
        """
        Returns a list of image path and corresponding labels.

        Args:
            class_roots (list): List of class root directories.
            formats (set): Set of image file formats to consider.

        Returns:
             tuple: A tuple containing a list of image paths and a list of labels.
        """
        if formats is None:
            formats = {
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp'
            }
            
        images, labels = [], []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(img_path)
                for img_path in sorted(Path(class_root).glob('*'))
                if img_path.is_file() and img_path.suffix.lower() in formats
            ]
            images += class_images
            labels += class_id * len(class_images)
            
        if len(labels) == 0:
            warnings.warn(UserWarning(
                'No images were found in the specified directory. The dataset will be empty.'
            ))
            
        return images, labels
        
    @staticmethod
    def set_transform(image_size, training):
        """
        Set the image transformation pipeline.

        Args:
            image_size (int): The size of the image (width and height).
            training (bool): Wheter the dataset is used for training or not.
            
        Returns:
            callable: A function/transform that takes in an image and returns a transformed version.
        """
        return (
            T.Compose([
                T.Resize([image_size, image_size]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
            if training
            else T.Compose([
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        )

    def __getitem__(self, item):
        """
        Get the image and label at the specified index.

        Args:
            item (int): Index of the item.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        image = self.transform(cv.imread(self.images[item])[:,:,::-1])
        label = self.labels[item]
        return image, label

    def __len__(self):
        """
        Get the length of the dataset

        Returns:
            int: The number of data instances in the dataset.
        """
        return len(self.labels)
        
    def get_labels(self):
        """
        Get the list of labels in the dataset.

        Returns:
            list: A list of labels.
        """
        return self.labels

class FewShotBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        n_way,
        n_shot,
        n_query,
        n_task
    ):
        """
        Initialize batch sampler.

        Args:
            dataset (Dataset): The dataset to sample from.
            n_way (int): Number of classes in each task.
            n_shot (int): Number of samples per class in the support set.
            n_query (int): number of samples per class in the query set.
            n_task (int): Number of tasks (iterations).
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_task = n_task

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]
        
    def __len__(self):
        """
        Get the number of tasks (iterations).

        Returns:
            int: The number of tasks.
        """
        return self.n_task
    
    def __iter___(self):
        """
        Iterate over the tasks.

        Returns:
            iterator: An iterator over the tasks.
        """
        for _ in range(self.n_task):
            yield torch.cat([
                torch.tensor(random.sample(
                    self.items_per_label[label], self.n_shot + self.n_query
                ))
                for label in random.sample(self.items_per_label.keys(), self.n_way)
            ]).tolist()
    
    def collate_fn(self, input_data):
        """
        Collate function for batch creation.

        Args:
            input_data (list): List of data items.
        Returns:
            tuple: A tuple containing support imagesfor x in input_data, support labels, query images, 
                   query labels, and true class IDs
        """
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])   # shape: (B, C, H, W) 
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )   # shape: (N_WAY, N_SHOT + N_QUERY, C, H, W)

        true_class_ids = list({x[1] for x in input_data})

        all_labels = torch.tensor([true_class_ids.index(x[1]) for x in input_data])
        all_labels = all_labels.reshape(
            (self.n_way, self.n_shot + self.n_query)
        )   # shape: (N_WAY, N_SHOT + N_QUERY)

        # Support sets
        support_images = all_images[:, :self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )   # shape images: (N_WAY * N_SHOT, C, H, W)
        
        support_labels = all_labels[:, :self.n_shot].flatten()   # shape labels: (N_WAY * N_SHOT, )
        
        # Query sets
        query_images = all_images[:, self.n_shot:].reshape(
            (-1, *all_images.shape[2:])
        )   # shape images: (N_WAY * N_QUERY, C, H, W)

        query_labels = all_labels[:, self.n_shot:].flatten()   # shape labels: (N_WAY * N_QUERY, )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids
        )