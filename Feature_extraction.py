import numpy as np

def extract_features(signal):
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "var": np.var(signal)
    }
    return features
