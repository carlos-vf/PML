import os
import numpy as np
from utils.keel_loader import load_keel_dataset
from utils.noising import add_noise, add_label_noise

dataset_path = "KEEL/keel_data/iris.dat"


# Load dataset and split
X_train, X_test, y_train, y_test, label_map = load_keel_dataset(
    train_path=dataset_path,
    already_split=False
)
y_train = y_train - 1  # Adjust labels to start from 0
y_test = y_test - 1  # Adjust labels to start from 0
