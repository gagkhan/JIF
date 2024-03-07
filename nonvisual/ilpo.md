### ILPO

ILPO model is implemented in nav2d/models. 

```
cd nav2d
python train.py 
```

It can be trained in two modes, one, where we allow action loss gradients to propogate back to the latent action network, and two, where we don't allow this gradient propogation. By default, this gradient propogation is enabled. To disable it pass `--detach_latent`. For more configuration, look at `train.py`


### ILPO-AC

Action Chunking (AC) has been helpful for imitation learning. We will explore this in the conext of observation imitiation learning. This yet to be implemented model is aimed at exactly this.