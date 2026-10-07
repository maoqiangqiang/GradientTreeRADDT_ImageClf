import torch
from PIL import Image


#### Dataset Reader from path and label
class PathDatasetReader(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset that mimics the behavior of torchvision.datasets.ImageFolder.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(PathDatasetReader, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.
        
        Args:
            idx (int): Index
        
        Returns:
            tuple: (sample, target) where sample is the transformed image and target is the class label.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

                
        # Convert label to tensor, ensure it's float32 for BCEWithLogitsLoss
        label = torch.tensor(label, dtype=torch.long)

        return image, label, image_path 
    
    def print_class_distribution(self):
        nClass = len(set(self.labels))
        print(f"Number of classes: {nClass}")
        for i in range(nClass):
            count = self.labels.count(i)
            print(f"Number of data for class {i}: {count}")
        
        return nClass

