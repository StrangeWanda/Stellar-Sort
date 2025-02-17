import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import os

# Load the CSV files
df_labels = pd.read_csv(r"C:\Users\nandh\Downloads\GalaxyZoo1_DR_table2.csv")
df_mapping = pd.read_csv(r"C:\Users\nandh\Downloads\gz2_filename_mapping.csv")

# Merge on the OBJID (ensure column names match)
df = pd.merge(df_mapping, df_labels, left_on="objid", right_on="OBJID")

# Define base image path
base = Path(r"C:\Users\nandh\Downloads\images_gz2\images")

# Function to get image path
def get_image_path(asset_id):
    return base / f"{asset_id}.jpg"

# Filter out missing images
df["image_path"] = df["asset_id"].apply(lambda x: get_image_path(x))
df = df[df["image_path"].apply(lambda x: x.exists())]  # Keep only existing images

# Galaxy Dataset Class
class GalaxyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")
        label = self._process_label(row)

        if self.transform:
            image = self.transform(image)
        return image, label

    def _process_label(self, row):
        if row['SPIRAL'] == 1:
            return 0
        elif row['ELLIPTICAL'] == 1:
            return 1
        else:
            return 2