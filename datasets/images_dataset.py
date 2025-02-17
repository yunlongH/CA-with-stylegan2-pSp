from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		# print('source_paths: ', self.source_paths)
		# print('target_paths: ', self.target_paths)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im

class ImagesDatasets2(Dataset):

    def __init__(self, image_dirs_file, labels_file, transform=None):
        """
        Args:
            image_dirs_file (str): Path to the .npy file containing image paths.
            labels_file (str): Path to the .npy file containing corresponding labels.
            opts (dict): Additional options for the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load image paths and labels from .npy files
        self.image_paths = np.load(image_dirs_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Tuple: (image, label) where image is the transformed image tensor and label is the corresponding age.
        """
        # Retrieve the image path
        image_path = self.image_paths[index]

        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and its label
        return image, self.labels[index]
	
class ImagesDatasets_cls(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_dirs_file (str): Path to the .npy file containing image paths.
            labels_file (str): Path to the .npy file containing corresponding labels.
            opts (dict): Additional options for the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load image paths and labels from .npy files
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Tuple: (image, label) where image is the transformed image tensor and label is the corresponding age.
        """
        # Retrieve the image path
        image_path = self.image_paths[index]

        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and its label
        return image, self.labels[index]