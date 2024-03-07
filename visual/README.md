# CPT
Cross-embodiment Pre-training


# Commands

### Jupyter Lab

jupyter lab --no-browser --port=8888

### Conda
conda env export


### DINO

python main_dino.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v0_frames  --output_dir ../../DINO/<outdir> --epochs 200 --local_crops_number 0

### DINO + CPT

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v0_frames  --output_dir ../../DINO/<outdir>  --epochs 200 --local_crops_number 0

```

# Data

The command to use to copy images from subfolders to a new directory and prefix subfolder name.

```
find src -type f -exec bash -c 'cp "$0" "dest/$(basename "$(dirname "$0")")_$(basename "$0")"' {} \;
```

Command to copy videos 