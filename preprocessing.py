# -*- coding: utf-8 -*-

"""
Image Preprocessing Utilities
=============================

This module provides various image preprocessing utilities for the computer vision project.
"""

import logging
import os
import sys
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

class ImagePreprocessor:
    """
    Image Preprocessor class
    """

    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Initialize the Image Preprocessor

        Args:
            image_size (Tuple[int, int], optional): Desired image size. Defaults to IMAGE_SIZE.
        """
        self.image_size = image_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the desired size

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Resized image
        """
        try:
            image = cv2.resize(image, self.image_size)
            logger.info(f"Resized image to {self.image_size}")
            return image
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            raise

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the range [0, 1]

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Normalized image
        """
        try:
            image = image / 255.0
            logger.info("Normalized image")
            return image
        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image to grayscale

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Grayscale image
        """
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.info("Converted image to grayscale")
            return image
        except Exception as e:
            logger.error(f"Failed to convert image to grayscale: {e}")
            raise

    def apply_threshold(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Apply threshold to the image

        Args:
            image (np.ndarray): Input image
            threshold (int, optional): Threshold value. Defaults to 127.

        Returns:
            np.ndarray: Thresholded image
        """
        try:
            _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            logger.info(f"Applied threshold {threshold} to image")
            return image
        except Exception as e:
            logger.error(f"Failed to apply threshold to image: {e}")
            raise

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Apply Gaussian blur to the image

        Args:
            image (np.ndarray): Input image
            kernel_size (Tuple[int, int], optional): Kernel size. Defaults to (5, 5).

        Returns:
            np.ndarray: Blurred image
        """
        try:
            image = cv2.GaussianBlur(image, kernel_size, 0)
            logger.info(f"Applied Gaussian blur with kernel size {kernel_size} to image")
            return image
        except Exception as e:
            logger.error(f"Failed to apply Gaussian blur to image: {e}")
            raise

class ImageDataset(Dataset):
    """
    Image Dataset class
    """

    def __init__(self, image_paths: list, image_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Initialize the Image Dataset

        Args:
            image_paths (list): List of image paths
            image_size (Tuple[int, int], optional): Desired image size. Defaults to IMAGE_SIZE.
        """
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        return image, 0

class ImageTransformer:
    """
    Image Transformer class
    """

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Transform the image

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Transformed image
        """
        try:
            image = self.transform(image)
            logger.info("Transformed image")
            return image
        except Exception as e:
            logger.error(f"Failed to transform image: {e}")
            raise

def main():
    # Example usage
    image_path = "path/to/image.jpg"
    image = cv2.imread(image_path)
    preprocessor = ImagePreprocessor()
    image = preprocessor.resize_image(image)
    image = preprocessor.normalize_image(image)
    image = preprocessor.convert_to_grayscale(image)
    image = preprocessor.apply_threshold(image)
    image = preprocessor.apply_gaussian_blur(image)

if __name__ == "__main__":
    main()