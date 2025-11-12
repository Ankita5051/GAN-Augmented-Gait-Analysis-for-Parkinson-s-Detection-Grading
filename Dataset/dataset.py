from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class PDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = pd.read_pickle("preprocessed_data.pkl")
X = np.array([np.array(i) for i in dataset['signal']])

y = np.array(dataset['label'])

# Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)


# Create datasets & loaders
train_loader = DataLoader(PDDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(PDDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(PDDataset(X_test, y_test), batch_size=32, shuffle=False)


if __name__=='__main__':
    for batch_x, batch_y in train_loader:
        print("Batch X shape:", batch_x.shape)
        print("Batch y shape:", batch_y.shape)
        break
    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print(dataset.head(5))
  

   
