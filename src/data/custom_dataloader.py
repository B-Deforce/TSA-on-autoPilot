import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    """
    A custom dataset class that inherits from torch.utils.data.Dataset and can include metadata
    Metadata can be e.g. the position of the anomaly, level, ...
    """

    def __init__(self, x, y, metadata=None):
        self.x = x
        self.y = y
        self.metadata = metadata

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.metadata is None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx], self.metadata[idx]


def get_dataloaders(train, val, test, batch_size):
    """
    Creates dataloaders for training, validation, and testing with potential metadata
    """
    # Unpack data
    X_train, y_train, meta_train = train
    X_val, y_val, meta_val = val
    X_test, y_test, meta_test = test
    # Create TensorDatasets
    train_dataset = CustomDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(meta_train, dtype=torch.float32)
        if meta_train is not None
        else None,
    )
    val_dataset = CustomDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(meta_val, dtype=torch.float32) if meta_val is not None else None,
    )
    test_dataset = CustomDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(meta_test, dtype=torch.float32) if meta_test is not None else None,
    )
    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_dataloader, val_dataloader, test_dataloader
