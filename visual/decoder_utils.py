from torch import nn


class ResNet34Decoder(nn.Module):
    """
    ChatGPT assisted code to build a ResNet34Decoder. No idea what makes it ResNet34 decoder.
    Anyhow, it takes a latent input (batch, embedding_dim=512, 7, 7) input and returns
    a 3-channel RGB image (batch, 3, 224, 224).

    Now, we probably need add a dense linear layer would take a different 1-d latent code to convert it
    to (batch, embedding_dim=512, 7, 7) latent. Maybe we add it outside of this class, idk, lets see.

    """

    def __init__(self, embed_dim):
        super(ResNet34Decoder, self).__init__()
        self.l1 = nn.Linear(embed_dim, 512 * 7 * 7)  # dense layer to convert it to (batch, 512, 7, 7)
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
        x = self.l1(x)
        x = x.view(-1, 512, 7, 7)
        x = self.relu(self.batchnorm1(self.upconv1(x)))
        x = self.relu(self.batchnorm2(self.upconv2(x)))
        x = self.relu(self.batchnorm3(self.upconv3(x)))
        x = self.relu(self.batchnorm4(self.upconv4(x)))
        x = self.sigmoid(self.upconv5(x))  # Output in range [0, 1] (batch, 3 , 224, 224)
        return x


def build_visual_decoder(embed_dim, args) -> nn.Module:
    """
    This function can build decoder and return a decoder of choice. Currently, we have only one decoder but
    eventually we will have more options.
    """
    decoder = ResNet34Decoder(embed_dim)
    return decoder
