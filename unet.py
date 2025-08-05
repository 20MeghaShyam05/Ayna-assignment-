import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer"""

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim

        # Linear layers to generate scale (gamma) and shift (beta) parameters
        self.gamma_linear = nn.Linear(condition_dim, feature_dim)
        self.beta_linear = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, condition):
        """
        Args:
            x: Feature tensor of shape (B, C, H, W)
            condition: Condition tensor of shape (B, condition_dim)

        Returns:
            Modulated features of shape (B, C, H, W)
        """
        # Generate scale and shift parameters
        gamma = self.gamma_linear(condition)  # (B, feature_dim)
        beta = self.beta_linear(condition)    # (B, feature_dim)

        # Reshape for broadcasting with feature maps
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)  # (B, C, 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)      # (B, C, 1, 1)

        # Apply FiLM: scale and shift
        return gamma * x + beta


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for polygon coloring.
    Takes an input image and color condition to generate colored polygon.
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 3,
                 num_colors: int = 8,
                 color_embed_dim: int = 128,
                 bilinear: bool = True):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_colors = num_colors
        self.color_embed_dim = color_embed_dim
        self.bilinear = bilinear

        # Color embedding layer
        self.color_embedding = nn.Sequential(
            nn.Linear(num_colors, color_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(color_embed_dim, color_embed_dim),
            nn.ReLU(inplace=True)
        )

        # UNet encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # FiLM layers for conditioning at different scales
        self.film1 = FiLM(color_embed_dim, 64)
        self.film2 = FiLM(color_embed_dim, 128)
        self.film3 = FiLM(color_embed_dim, 256)
        self.film4 = FiLM(color_embed_dim, 512)
        self.film5 = FiLM(color_embed_dim, 1024 // factor)

        # UNet decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Output activation
        self.output_activation = nn.Tanh()  # Output in [-1, 1] range

    def forward(self, x, color_condition):
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W)
            color_condition: Color condition tensor of shape (B, num_colors) - one-hot encoded

        Returns:
            Generated colored polygon of shape (B, 3, H, W)
        """
        # Embed color condition
        color_embed = self.color_embedding(color_condition)  # (B, color_embed_dim)

        # Encoder path with FiLM conditioning
        x1 = self.inc(x)
        x1 = self.film1(x1, color_embed)

        x2 = self.down1(x1)
        x2 = self.film2(x2, color_embed)

        x3 = self.down2(x2)
        x3 = self.film3(x3, color_embed)

        x4 = self.down3(x3)
        x4 = self.film4(x4, color_embed)

        x5 = self.down4(x4)
        x5 = self.film5(x5, color_embed)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output layer
        logits = self.outc(x)
        output = self.output_activation(logits)

        return output


class AlternativeConditionalUNet(nn.Module):
    """
    Alternative conditioning approach: concatenate color info with input image
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 3,
                 num_colors: int = 8,
                 bilinear: bool = True):
        super(AlternativeConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_colors = num_colors
        self.bilinear = bilinear

        # Input channels = image channels + color channels
        input_channels = n_channels + num_colors

        # UNet architecture
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.output_activation = nn.Tanh()

    def forward(self, x, color_condition):
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W)
            color_condition: Color condition tensor of shape (B, num_colors)

        Returns:
            Generated colored polygon of shape (B, 3, H, W)
        """
        batch_size, _, height, width = x.shape

        # Expand color condition to match spatial dimensions
        color_maps = color_condition.unsqueeze(-1).unsqueeze(-1)  # (B, num_colors, 1, 1)
        color_maps = color_maps.expand(-1, -1, height, width)     # (B, num_colors, H, W)

        # Concatenate image and color information
        x_conditioned = torch.cat([x, color_maps], dim=1)  # (B, 3+num_colors, H, W)

        # Standard UNet forward pass
        x1 = self.inc(x_conditioned)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        output = self.output_activation(logits)

        return output


def create_model(model_type: str = "film", **kwargs):
    """
    Factory function to create different model variants.

    Args:
        model_type: "film" for FiLM-based conditioning, "concat" for concatenation-based
        **kwargs: Additional arguments for model initialization

    Returns:
        Model instance
    """
    if model_type == "film":
        return ConditionalUNet(**kwargs)
    elif model_type == "concat":
        return AlternativeConditionalUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test the models
if __name__ == "__main__":
    # Test FiLM-based model
    model_film = ConditionalUNet(num_colors=8)

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    color_condition = torch.zeros(batch_size, 8)
    color_condition[0, 0] = 1  # First sample: color 0
    color_condition[1, 3] = 1  # Second sample: color 3

    # Forward pass
    output = model_film(x, color_condition)
    print(f"FiLM model output shape: {output.shape}")

    # Test concatenation-based model
    model_concat = AlternativeConditionalUNet(num_colors=8)
    output_concat = model_concat(x, color_condition)
    print(f"Concatenation model output shape: {output_concat.shape}")

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"FiLM model parameters: {count_parameters(model_film):,}")
    print(f"Concatenation model parameters: {count_parameters(model_concat):,}")