import pandas as pd
import numpy as np
from src.config import DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH

def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

def preprocess_data(data):
    data = (data - data.mean()) / data.std()
    return data

def save_processed_data(data, labels):
    np.save(PROCESSED_DATA_PATH, data)
    np.save(PROCESSED_LABELS_PATH, labels)

# Пример использования
if __name__ == '__main__':
    data = load_data()
    processed_data = preprocess_data(data.drop('target', axis=1))
    labels = data['target'].values
    save_processed_data(processed_data, labels)
