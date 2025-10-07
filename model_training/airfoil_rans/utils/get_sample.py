from pathlib import Path
import numpy as np

def get_sample(idx, data_path):

    data_path = Path(data_path).expanduser().resolve()
    data_path = data_path / 'raw_data'

    nodes = np.load(data_path / f'nodes_{idx:05d}.npy')
    features = np.load(data_path / f'features_{idx:05d}.npy')

    return nodes, features