import torch
from torch import nn
from torchvision import models

from data import VisDemoDataset

"""

How to implement a decoder?

The basic steps in a decoder are:
    1. Upsample the input
    2. Convolutional transpose layer
    3. Normalization layer
    4. ReLU layer
    5. Bottleneck layer

The first thing that I need to do is to upsample the input
Reference: https://github.com/JiahongChen/ResNet-decoder

# There are two things that are new to me here:
# 1. Upsampling the input
# 2. Convolutional transpose layer

"""

from vector_quantize_pytorch import VectorQuantize


class ResNet34Decoder(nn.Module):
    """
    ChatGPT assisted code to build a ResNet34Decoder. No idea what makes it ResNet34 decoder.
    Anyhow, it takes a latent input (batch, embedding_dim=512, 7, 7) input and returns
    a 3-channel RGB image (batch, 3, 224, 224).

    Now, we probably need add a dense linear layer would take a different 1-d latent code to convert it
    to (batch, embedding_dim=512, 7, 7) latent. Maybe we add it outside of this class, idk, lets see.

    """

    def __init__(self):
        super(ResNet34Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(
            64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Output 3 channels

        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.upconv1(x)))
        x = self.relu(self.batchnorm2(self.upconv2(x)))
        x = self.relu(self.batchnorm3(self.upconv3(x)))
        x = self.relu(self.batchnorm4(self.upconv4(x)))
        x = self.sigmoid(self.upconv5(x))  # Output in range [0, 1] (batch, 3 , 224, 224)
        return x


class VQVAE(nn.Module):

    def __init__(
        self,
        num_embeddings=256,
    ) -> None:

        super().__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet34.children())[:-2])  # Remove the fully connected layers
        self.embedding_dim = 512  # fixed because we use resnet34
        self.quantizer = VectorQuantize(
            self.embedding_dim, num_embeddings, decay=0.8, commitment_weight=1.0, accept_image_fmap=True
        )
        self.decoder = ResNet34Decoder()

    def forward(self, x):
        z_e = self.encoder(x)  # (batch, embedding_dim, 7, 7)
        z_q, loss, _ = self.quantizer(z_e)  # (batch, embedding_dim, 7, 7)
        x_recon = self.decoder(z_q)
        return x_recon, loss
        # return x_recon


def test_vq_vae():

    model = VQVAE(
        num_embeddings=256,
        embedding_dim=512,
    )

    img = torch.rand(2, 3, 224, 224)

    model(img)


if __name__ == "__main__":

    test_vq_vae()
