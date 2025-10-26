import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dotenv import load_dotenv
load_dotenv()
np.random.seed(42)
seq_len=int(os.environ['WINDOW_SIZE'])

current_dir = os.path.dirname(os.path.abspath(__file__))
pth =os.path.join(current_dir,"gaitpdb","data")
UPDRS_path=os.path.join(current_dir,"gaitpdb","demographics.xls")


files= os.listdir(pth)
#reading demographic sheet for severity levels
df = pd.read_excel(UPDRS_path, usecols=['ID', 'UPDRS'])
UPDRS_level = dict(zip(df['ID'], df['UPDRS']))

def CVGRF_segmented(files,UPDRS_level, path, window_size=3000) -> pd.DataFrame:
    dataset = []
    label = []
    UPDRS =[]
    id=[]

    for i in files:
        file_path = os.path.join(path, i)
        data = pd.read_csv(file_path, sep='\t', header=None)

        # Determine the label based on filename
        if "Pt" in i:
            file_label = 1  # Parkinson's disease
            updrs=UPDRS_level[i.split('_')[0]]
            if updrs =="nan" or updrs=="NAN":
              updrs=4
            sl=1
            if updrs < 5:
                sl = 1
            elif updrs < 15:
                sl = 2
            elif updrs < 25:
                sl = 3
            elif updrs <35:
                sl = 4
            else:
                sl = 5

        elif "Co" in i:
            file_label = 0  # Control
            sl=0

        else:
            # Handle files that don't match "Pt" or "Co" if necessary
            continue

        cm_sum = data.iloc[:, -1] + data.iloc[:, -2]
        cm_sum = cm_sum.values

        # number of full windows
        num_segments = len(cm_sum) // window_size

        for seg in range(num_segments):
            start = seg * window_size
            end = start + window_size
            segment = cm_sum[start:end]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten() # type: ignore
            #segment = (segment - np.mean(segment)) / np.std(segment)

            dataset.append(segment)
            label.append(file_label)  # Assign the numerical label
            UPDRS.append(sl)
            id.append(i)

    return pd.DataFrame({
        "id":id,
        "signal": dataset,
        "label": label,
        "UPDRS":UPDRS,
    })
class PDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = CVGRF_segmented(files, UPDRS_level, pth, seq_len)
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
    print(dataset.head(170))
  

   
