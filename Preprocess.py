import pandas as pd
from scipy.signal import savgol_filter

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Smooth signal
    data['filtered'] = savgol_filter(data['amplitude'], 5, 2)

    return data
