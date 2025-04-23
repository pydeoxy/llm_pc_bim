import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class H5PCDataset(Dataset):
    def __init__(self, file_path, transform=None, pre_transform = None):
        # Open the hdf5 file
        with h5py.File(file_path, 'r') as f:
            # Load the entire dataset into memory
            self.data = np.array(f['data']).astype(np.float32)    # Shape: (num_blocks, 4096, 9)
            self.labels = np.array(f['label'])   # Shape: (num_blocks, 4096)
        self.transform = transform
        self.pre_transform = pre_transform

        # Precompute the number of unique labels (num_classes)
        self._num_classes = len(np.unique(self.labels))

    @property
    def num_classes(self):
        """Return the number of unique labels in the dataset."""
        return self._num_classes   

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        # Convert the 4096x9 block into a torch tensor
        block = torch.from_numpy(self.data[idx])  # Shape: (4096, 9)
        label = torch.from_numpy(self.labels[idx]).to(torch.long)  # Shape: (4096)
        
        # Slice the tensor so that:
        #   - pos gets the first 3 columns (x, y, z)
        #   - x gets the remaining 6 columns (features)
        pos = block[:, :3]
        features = block[:, 3:]
    
        # Replace the first 3 feature values of each point with zeros.
        features[:, :3] = 0

        data = Data(pos=pos, x=features, y=label)
        
        if self.transform is not None:
            data = self.transform(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        return data