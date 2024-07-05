import wandb
import umap
import umap.plot
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def umap_log():
    #     """
    #     Log the UMAP embedding of the data.

    #     Returns
    #     -------
    #     None
    #     """
    data = np.random.rand(1000, 100)

    #     print(data.shape)
    reducer = umap.UMAP()
    u = reducer.fit_transform(data)

    plt.scatter(u[:, 0], u[:, 1])
    plt.savefig("umap.png")

    wandb.init(project="cpt_umap")
    wandb.log({"umap": wandb.Image("umap.png")})

    umap.plot.points(reducer, labels=None, theme="fire")

    plt.savefig("umap2.png")

    wandb.log({"umap": wandb.Image("umap2.png")})


if __name__ == "__main__":
    umap_log()
