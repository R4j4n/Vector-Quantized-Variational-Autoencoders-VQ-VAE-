import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()

        self.commitment_cost = commitment_cost  # weighting factor
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # initialize the codebook vector
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # reset params
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, latents):
        # reshape the latent BCHW -> BHWC
        latents_r = latents.movedim(1, -1)  # similar to permute(0, 2, 3, 1)
        latents = latents_r.contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(distances, dim=1)
        z = self.embedding(min_encoding_indices)
        quantized_latents = z.view(latents_r.shape)

        # vq loss
        codebook_loss = F.mse_loss(latents_r.detach(), quantized_latents)

        # commitement loss
        commitment_loss = F.mse_loss(latents_r, quantized_latents.detach())

        # Compute the VQ Losses
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # preserving the gradients for the backward folw:
        quantized_latents = latents_r + (quantized_latents - latents_r).detach()

        return quantized_latents.movedim(-1, 1), vq_loss


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, activation=True
    ):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list, latent_channels) -> None:
        super().__init__()

        self.input_conv = ConvBlock(in_channels, channels_list[0], kernel_size=3)

        self.downsample = nn.ModuleList()

        for in_channel, out_channel in zip(channels_list[:-1], channels_list[1:]):
            self.downsample.append(nn.MaxPool2d(2))
            self.downsample.append(ConvBlock(in_channel, out_channel, kernel_size=3))

        self.latent_conv = ConvBlock(channels_list[-1], latent_channels, 3)

    def forward(self, x):
        x = self.input_conv(x)
        for f in self.downsample:
            x = f(x)
        x = self.latent_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels, channels_list, out_channels):
        super().__init__()
        self.stem = ConvBlock(latent_channels, channels_list[0], 3)

        self.upsample = nn.ModuleList()
        for in_channel, out_channel in zip(channels_list[:-1], channels_list[1:]):
            self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.upsample.append(ConvBlock(in_channel, out_channel, 3))

        self.to_output = nn.Conv2d(channels_list[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        for f in self.upsample:
            x = f(x)
        x = self.to_output(x)
        x = torch.sigmoid(x)
        return x


class VecotrQuantizerAE(nn.Module):
    def __init__(
        self,
        num_downsamplings,
        latent_channels,
        num_embeddings,
        channels=32,
        in_channels=3,
    ) -> None:
        super().__init__()

        channel_list = [channels * 2**i for i in range(num_downsamplings + 1)]
        channels_list_reverse = channel_list[::-1]

        self.encoder = Encoder(in_channels, channel_list, latent_channels)

        self.vq = VectorQuantizer(num_embeddings, latent_channels)

        self.decoder = Decoder(latent_channels, channels_list_reverse, in_channels)

        self.reduction = 2**num_downsamplings
        self.num_embeddings = num_embeddings

    def forward(self, x):
        latents = self.encoder(x)
        z, vq_loss = self.vq(latents)
        decoded = self.decoder(z)

        return decoded, vq_loss


if __name__ == "__main__":
    embedding_dim = 4
    num_embeddings = 8
    commitment_cost = 0.25

    # input
    # BCHW
    input_tensor = torch.rand(size=(2, embedding_dim, 2, 2))
    vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
    q_l, loss = vq(input_tensor)

    assert q_l.shape == input_tensor.shape
