from __future__ import annotations

import torch


class TorchScaler:

    def fit(self, x: torch.Tensor):
        raise NotImplementedError

    def transform(self, x: torch.Tensor):
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor):
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor):

        self.fit(x)

        return self.transform(x)

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError



class IdentityScaler(TorchScaler):

    def fit(self, x):

        self.n_features_in_ = x.shape[-1]

        return self

    def transform(self, x):

        return x

    def inverse_transform(self, x):

        return x
        
    def state_dict(self):
        return {
            "n_features_in_": self.n_features_in_
        }
        
    def load_state_dict(self, state):
        self.n_features_in_ = state["n_features_in_"]

class StandardScaler(TorchScaler):

    def fit(self, x):

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        # NaN-aware moments: columns may contain missing values for
        # outputs that are physically undefined for some rows (for
        # example a mean energy when no particles were produced). Those
        # entries must not corrupt the fitted statistics of the column.
        valid = ~torch.isnan(x)
        counts = valid.sum(dim=0, keepdim=True).clamp_min(1)

        filled = torch.where(
            valid,
            x,
            torch.zeros_like(x),
        )

        self.center_ = filled.sum(dim=0, keepdim=True) / counts

        squared_deviation = torch.where(
            valid,
            (x - self.center_).square(),
            torch.zeros_like(x),
        )

        variance = squared_deviation.sum(dim=0, keepdim=True) / counts

        self.scale_ = torch.sqrt(variance)

        self.scale_ = torch.clamp(
            self.scale_,
            min=1e-12,
        )


    def transform(self, x):

        return (
            x - self.center_.to(x.device)
        ) / self.scale_.to(x.device)

    def inverse_transform(self, x):

        return (
            x * self.scale_.to(x.device)
        ) + self.center_.to(x.device)

    def state_dict(self):
        return {
            "center_": self.center_,
            "scale_": self.scale_,
        }

    def load_state_dict(self, state):
        self.center_ = state["center_"]
        self.scale_ = state["scale_"]
        
class RobustScaler(TorchScaler):

    def __init__(
        self,
        q_min: float = 0.25,
        q_max: float = 0.75,
    ):

        self.q_min = q_min
        self.q_max = q_max

    def fit(self, x):

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        self.center_ = torch.quantile(
            x,
            0.5,
            dim=0,
            keepdim=True,
        )

        q_low = torch.quantile(
            x,
            self.q_min,
            dim=0,
            keepdim=True,
        )

        q_high = torch.quantile(
            x,
            self.q_max,
            dim=0,
            keepdim=True,
        )

        self.scale_ = q_high - q_low

        self.scale_ = torch.clamp(
            self.scale_,
            min=1e-12,
        )

        return self

    def transform(self, x):

        return (
            x - self.center_.to(x.device)
        ) / self.scale_.to(x.device)

    def inverse_transform(self, x):

        return (
            x * self.scale_.to(x.device)
        ) + self.center_.to(x.device)

    def state_dict(self):
        return {
            "center_": self.center_,
            "scale_": self.scale_,
        }

    def load_state_dict(self, state):
        self.center_ = state["center_"]
        self.scale_ = state["scale_"]
        
def make_scaler(name: str):

    if name == "standard":
        return StandardScaler()

    if name == "robust":
        return RobustScaler()

    if name == "none":
        return IdentityScaler()

    raise ValueError(
        f"Unsupported scaler '{name}'"
    )