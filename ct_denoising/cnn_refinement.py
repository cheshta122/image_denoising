"""Optional CNN residual refinement after transform-domain denoising.

This module is intentionally lightweight. It is not used by the default
pipeline because research-grade CNN refinement requires paired training data,
validation splits, and careful reporting. The model can be trained separately
and then applied to transform-domain outputs.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:

    class ResidualRefinementCNN(nn.Module):
        """Small residual CNN that predicts a cleanup residual."""

        def __init__(self, channels: int = 32) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            )

        def forward(self, image):
            residual = self.net(image)
            return torch.clamp(image - residual, 0.0, 1.0)

else:

    class ResidualRefinementCNN:  # type: ignore[no-redef]
        """Placeholder shown when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use ResidualRefinementCNN.")


def refine_with_cnn(model, image, device: str = "cpu"):
    """Run inference with a trained residual CNN on a single 2D image."""

    if torch is None:
        raise ImportError("Install torch to use CNN refinement.")

    model = model.to(device)
    model.eval()
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)[None, None]
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze().detach().cpu().numpy()

