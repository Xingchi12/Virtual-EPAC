
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class TargetTransform:

    names: list[str]
    positive_names: set[str]
    mode: str

    def config(
        self,
    ):
        return {
            'names': self.names,
            'positive_names': self.positive_names,
            'mode': self.mode,
        }

    @classmethod
    def from_cfg(
        cls,
        config,
    ):
        return cls(
            names=config['names'],
            positive_names=config['positive_names'],
            mode=config['mode'],
        )
    def forward(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        transformed = np.asarray(
            values,
            dtype=float,
        ).copy()

        if self.mode == "none":
            return transformed

        if self.mode != "log1p":
            raise ValueError(
                f"Unknown transform: {self.mode}"
            )

        for col, name in enumerate(self.names):

            if name not in self.positive_names:
                continue

            if np.nanmin(
                transformed[:, col]
            ) < 0:
                raise ValueError(
                    f"Target '{name}' contains "
                    "negative values."
                )

            transformed[:, col] = np.log1p(
                transformed[:, col]
            )

        return transformed

    def inverse(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        restored = np.asarray(
            values,
            dtype=float,
        ).copy()

        if self.mode == "none":
            return restored

        for col, name in enumerate(self.names):

            if name not in self.positive_names:
                continue

            restored[:, col] = np.expm1(
                np.clip(
                    restored[:, col],
                    -50.0,
                    50.0,
                )
            )

        return restored

    def forward_torch(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the configured target transformation to a tensor.
    
        Parameters
        ----------
        values
            Tensor of shape (n_samples, n_outputs).
    
        Returns
        -------
        torch.Tensor
            Transformed targets.
        """
    
        transformed = values.clone()
    
        if self.mode == "none":
            return transformed
    
        if self.mode != "log1p":
            raise ValueError(
                f"Unknown transform: {self.mode}"
            )
    
        for col, name in enumerate(self.names):
    
            if name not in self.positive_names:
                continue
    
            # NaN comparisons are always False, so this check correctly
            # ignores missing (NaN) entries and only flags real negatives.
            if bool((transformed[:, col] < 0).any()):
                raise ValueError(
                    f"Target '{name}' contains "
                    "negative values."
                )
    
            transformed[:, col] = torch.log1p(
                transformed[:, col]
            )
    
        return transformed
    
    def inverse_torch(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
    
        restored = values.clone()
    
        if self.mode == "none":
            return restored
    
        if self.mode == "log1p":
    
            for column, name in enumerate(self.names):
    
                if name not in self.positive_names:
                    continue
    
                restored[:, column] = torch.expm1(
                    restored[:, column].clamp(
                        -50.0,
                        50.0,
                    )
                )
    
        return restored