import logging
import os
import random
import time
import typing
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
VALID_SPLITS = ["train", "val", "test"]

# Exception classes
class InvalidDatasetError(Exception):
    """Exception raised for errors in the dataset."""
    pass

class DataLoader:
    """Main class for data loading and batching.

    This class handles loading and batching of image data for training and evaluation.
    It provides a high-level interface for data loading, including support for
    multiple image formats, data splits, and customizable transformations.

    ...

    Attributes
    ----------
    data_dir : str
        Path to the directory containing the image data.
    split : str
        Data split to use (train, val, or test).
    image_size : Tuple[int, int]
        Size to which images will be resized.
    batch_size : int
        Number of images in each batch.
    shuffle : bool
        Whether to shuffle the data before creating batches.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Whether to pin CPU memory during data loading.
    transform : callable, optional
        Optional transform to apply to the images, by default None.

    Methods
    -------
    load_data(self)
        Load the image data and labels.
    collate_fn(self, batch)
        Create batches of data and apply transformations.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform=None,
    ):
        """Initialize the DataLoader object.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the image data.
        split : str, optional
            Data split to use (train, val, or test), by default "train".
        image_size : Tuple[int, int], optional
            Size to which images will be resized, by default (224, 224).
        batch_size : int, optional
            Number of images in each batch, by default 32.
        shuffle : bool, optional
            Whether to shuffle the data before creating batches, by default True.
        num_workers : int, optional
            Number of worker processes for data loading, by default 4.
        pin_memory : bool, optional
            Whether to pin CPU memory during data loading, by default True.
        transform : callable, optional
            Optional transform to apply to the images, by default None.

        Raises
        ------
        InvalidDatasetError
            If the data directory does not exist or the split is invalid.
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

        if not os.path.exists(self.data_dir):
            raise InvalidDatasetError(f"Data directory '{self.data_dir}' does not exist.")
        if self.split not in VALID_SPLITS:
            raise InvalidDatasetError(f"Invalid data split '{self.split}'. Choose from: {VALID_SPLITS}.")

        self.data, self.labels = self.load_data()

    def load_data(self) -> Tuple[List[Image.Image], List[int]]:
        """Load the image data and labels.

        Returns
        -------
        Tuple[List[Image.Image], List[int]]
            A tuple containing a list of PIL Image objects and a list of corresponding labels.
        """
        data = []
        labels = []
        data_path = os.path.join(self.data_dir, self.split)
        logger.info(f"Loading data from '{data_path}'...")
        for filename in os.listdir(data_path):
            if any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_path = os.path.join(data_path, filename)
                image = Image.open(img_path)
                data.append(image)
                labels.append(int(filename.split("_")[0]))

        logger.info(f"Loaded {len(data)} images for split '{self.split}'.")
        return data, labels

    def collate_fn(
        self, batch: List[Tuple[Image.Image, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create batches of data and apply transformations.

        Parameters
        ----------
        batch : List[Tuple[Image.Image, int]]
            A list of tuples containing image data and corresponding labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing a batch of images and corresponding labels.
        """
        images, labels = zip(*batch)
        images = [img.resize(self.image_size) for img in images]

        if self.transform:
            images = [self.transform(image) for image in images]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    def __iter__(self):
        """Return an iterator over the batches of data."""
        return iter(
            DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
            )
        )

    def __len__(self):
        """Return the number of batches in the dataset."""
        return len(self.data) // self.batch_size

# Helper class for data transformation
class Transform:
    """Helper class for applying transformations to image data.

    This class provides methods for applying common transformations such as
    random cropping, horizontal flipping, and normalization.

    ...

    Attributes
    ----------
    mean : Tuple[float, float, float]
        Mean values for image normalization.
    std : Tuple[float, float, float]
        Standard deviation values for image normalization.

    Methods
    -------
    random_crop(self, image, size)
        Perform random cropping of the image.
    horizontal_flip(self, image)
        Perform horizontal flip of the image.
    normalize(self, image, mean, std)
        Normalize the image using mean and standard deviation.
    __call__(self, image)
        Apply the defined transformations to the image.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Initialize the Transform object.

        Parameters
        ----------
        mean : Tuple[float, float, float], optional
            Mean values for image normalization, by default (0.485, 0.456, 0.406).
        std : Tuple[float, float, float], optional
            Standard deviation values for image normalization, by default (0.229, 0.224, 0.225).
        """
        self.mean = mean
        self.std = std

    def random_crop(self, image: Image.Image, size: int) -> Image.Image:
        """Perform random cropping of the image.

        Parameters
        ----------
        image : Image.Image
            The input image to be cropped.
        size : int
            The size of the cropped image.

        Returns
        -------
        Image.Image
            The cropped image.
        """
        width, height = image.size
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))

    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Perform horizontal flip of the image.

        Parameters
        ----------
        image : Image.Image
            The input image to be flipped.

        Returns
        -------
        Image.Image
            The flipped image.
        """
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def normalize(
        self, image: Union[np.ndarray, torch.Tensor], mean: Tuple, std: Tuple
    ) -> Union[np.ndarray, torch.Tensor]:
        """Normalize the image using mean and standard deviation.

        Parameters
        ----------
        image : Union[np.ndarray, torch.Tensor]
            The input image to be normalized.
        mean : Tuple
            Mean values for normalization.
        std : Tuple
            Standard deviation values for normalization.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The normalized image.
        """
        # Convert image to numpy array if it's a torch tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # Normalize the image
        image = image / 255.0
        image = (image - mean) / std
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the defined transformations to the image.

        Parameters
        ----------
        image : Image.Image
            The input image to be transformed.

        Returns
        -------
        Image.Image
            The transformed image.
        """
        image = self.random_crop(image, size=self.image_size[0])
        if random.random() > 0.5:
            image = self.horizontal_flip(image)
        image = np.array(image)
        image = self.normalize(image, self.mean, self.std)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image

# Example usage
if __name__ == "__main__":
    data_dir = "/path/to/image_data"
    split = "train"

    # Create DataLoader object
    data_loader = DataLoader(data_dir, split=split, batch_size=32, shuffle=True)

    # Iterate over batches of data
    for images, labels in data_loader:
        print(images.shape, labels.shape)

    # Example transformation
    transform = Transform()
    image = Image.open("/path/to/image.jpg")
    transformed_image = transform(image)
    print(transformed_image.shape)