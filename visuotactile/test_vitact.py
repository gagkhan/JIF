import torch
from visuotactile.visuo_tactile_transformer import VisuoTactileTransformer

if __name__ == "__main__":

    vitact = VisuoTactileTransformer(
        input_sizes=[(3, 224, 224), (3, 224, 224), (2,)],
        patch_sizes=[16, 16, 1],
    )

    vitact = vitact.cuda()

    img1 = img2 = torch.rand((4, 3, 224, 224)).cuda()
    touch = torch.rand(
        (
            4,
            2,
        )
    ).cuda()

    vitact([img1, img2, touch])
