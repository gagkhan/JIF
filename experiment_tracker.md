# Experiments


## Feb 29
### CPT with Ours-v3

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v3_frames  --output_dir ../../DINO/feb29_ours_v3_cpt --epochs 2000 --local_crops_scale 0.4 1.0

```
Notes:
- This is the first run for the new dataset collected by Sarah 
- ours_v3 is a new "two-fingered" bi-manual manipulation with about 300

## Feb 26
### CPT with SSV2-Tiny
```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --output_dir ../../DINO/feb26_ssv2_cpt --epochs 2000 --local_crops_scale 0.4 1.0

```
Notes: We run this to compare with ours-v2

## Feb 22-23
### CPT with ours_v1
```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v1_frames --output_dir ../../DINO/feb23_2000  --epochs 2000  --local_crops_scale 0.4 1.0

```
Notes: We should run for more epochs because the dataset is tiny

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/ours_v2_frames --output_dir ../../DINO/feb22_ours_v1 --epochs 200 --local_crops_scale 0.4 1.0

```
Notes:
(1) Ours-v1 has some issues with start and end frames of the video
(2) These issues are fixed in Ours-v2


## Feb 14
### CPT with SSv2-Tiny (bottlneck dimension)
```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --output_dir ../../DINO/feb14_cpt 
```

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --local_crops_number 0 --output_dir ../../DINO/feb14_nolocal 
```

## Feb 7 to 13
### CPT with SSv2-Tiny (high-dimensional embedding)

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --output_dir ../../DINO/feb7_cpt
```

```
python main_cpt.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --output_dir ../../DINO/feb13_cpt
```


## Feb 4
### DINO with SSv2-Small
```
python main_dino.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-small  --output_dir ../../DINO/feb3_ssv2_small
```


## Feb 3
### DINO with SSv2-Tiny
```
python main_dino.py --arch vit_tiny --data_path /home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny  --output_dir ../../DINO/feb3_ssv2_tiny
```






