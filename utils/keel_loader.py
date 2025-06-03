import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_keel_dataset(train_path, test_path=None, already_split=False, test_size=0.2, random_state=42):
    """
    Load KEEL-format dataset(s) and return train/test splits.
    
    If `split=False`, train_path is assumed to be a full dataset that needs splitting.
    If `split=True`, both train_path and test_path must be provided (already split files).
    """
    def parse_keel_file(path):
        data, labels = [], []
        with open(path, 'r') as file:
            data_section = False
            for line in file:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                if line.lower() == '@data':
                    data_section = True
                    continue
                if data_section:
                    parts = line.split(',')
                    features = list(map(float, parts[:-1]))
                    data.append(features)
                    labels.append(parts[-1])
        return np.array(data), np.array(labels)
    
    if already_split:
        if test_path is None:
            raise ValueError("If split=True, both train and test paths must be provided.")
        X_train, y_train = parse_keel_file(train_path)
        X_test, y_test = parse_keel_file(test_path)
    else:
        X, y_raw = parse_keel_file(train_path)
        y_train, y_test, X_train, X_test = None, None, None, None
        # Encode labels
        unique_labels = sorted(set(y_raw))
        label_to_int = {label: i+1 for i, label in enumerate(unique_labels)}
        y = np.array([label_to_int[label] for label in y_raw])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        return X_train, X_test, y_train, y_test, label_to_int
    
    # For already-split case, apply label encoding to train and test consistently
    unique_labels = sorted(set(y_train) | set(y_test))
    label_to_int = {label: i+1 for i, label in enumerate(unique_labels)}
    y_train = np.array([label_to_int[label] for label in y_train])
    y_test = np.array([label_to_int[label] for label in y_test])

    return X_train, X_test, y_train, y_test, label_to_int
