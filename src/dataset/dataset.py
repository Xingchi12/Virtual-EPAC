from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

class SurrogateDataset(Dataset):
    """
    Dataset for surrogate modelling.

    Returns
    -------
    sample : dict
        {
            "inputs": Tensor,
            "targets": Tensor,
            "index": int
        }
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        sample_idxs: None,
        mask: np.ndarray = None,
        injection: np.ndarray = None,
    ):


        self.inputs = torch.as_tensor(
            inputs,
            dtype=torch.float32,
        )

        self.targets = torch.as_tensor(
            targets,
            dtype=torch.float32,
        )

        self.sample_idxs = sample_idxs

        self.mask = (
            torch.as_tensor(mask, dtype=torch.float32)
            if mask is not None
            else torch.ones_like(self.targets)
        )

        self.injection = (
            torch.as_tensor(injection, dtype=torch.float32)
            if injection is not None
            else None
        )

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
            "mask": self.mask[idx],
            "index": self.sample_idxs[idx] if self.sample_idxs is not None else None,
        }

        if self.injection is not None:
            item["injection"] = self.injection[idx]

        return item