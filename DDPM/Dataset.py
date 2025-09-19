import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SineWaveDataset(Dataset):
    """
    A dataset that generates sums of sine waves.
    Each data point is a 1D tensor of length 256.
    """

    def __init__(self, num_samples=10000, seq_len=256, num_freq=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_freq = num_freq
        self.data = self._generate_data()

    def _generate_data(self):
        """Generates the dataset of sine waves."""
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        all_waves = []
        for _ in range(self.num_samples):
            # Random amplitudes, frequencies, and phases
            amplitudes = np.random.rand(self.num_freq) * 2
            frequencies = np.random.rand(self.num_freq) * 5 + 1
            phases = np.random.rand(self.num_freq) * 2 * np.pi

            wave = np.zeros_like(x)
            for amp, freq, phase in zip(amplitudes, frequencies, phases):
                wave += amp * np.sin(freq * x + phase)

            # Normalize to [-1, 1] range
            wave /= np.max(np.abs(wave))

            all_waves.append(wave)

        # Reshape to (num_samples, 1, seq_len) for Conv1d
        return torch.tensor(np.array(all_waves), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Return 0 for the label, as it's unused