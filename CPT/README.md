# CPT
Cross-embodiment Pre-training


# Commands

### Jupyter Lab

jupyter lab --no-browser --port=8888

### Conda
conda env export


## Nav2D

Nav2D is a toy example to try out Observation only Imitation Learning (OIL)


### OIL

OIL model is implemented in nav2d/models. 

```
cd nav2d
python train.py 
```

It can be trained in two modes, one, where we allow action loss gradients to propogate back to the latent action network, and two, where we don't allow this gradient propogation. By default, this gradient propogation is enabled. To disable it pass `--detach_latent`. For more configuration, look at `train.py`


### OIL-AC

Action Chunking (AC) has been helpful for imitation learning. We will explore this in the conext of observation imitiation learning. This yet to be implemented model is aimed at exactly this.


### DINO

python main_dino.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v0_frames  --output_dir ../../DINO/<outdir> --epochs 200 --local_crops_number 0

### DINO + CPT

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v0_frames  --output_dir ../../DINO/<outdir>  --epochs 200 --local_crops_number 0
```