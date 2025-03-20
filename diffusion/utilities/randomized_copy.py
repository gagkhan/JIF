import os
import random
import shutil

src_dir = '/ssd01/gagan/cpt_data/ours/01_30_pickrobot_omnibus'

# List all subdirectories in the source directory
subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

# Shuffle
subdirs = sorted(subdirs, key=lambda x: int(x.split('_')[1]))
random.seed(42)
random.shuffle(subdirs)

# Usage
for num_subdirs in [100]:

    dst_dir = src_dir + "_" + str(num_subdirs)

    # Select
    selected_subdirs = subdirs[:num_subdirs]

    print(f"Copying {dst_dir}")
    # Copy selected subdirectories to the destination
    for subdir in selected_subdirs:
        src_path = os.path.join(src_dir, subdir)
        dst_path = os.path.join(dst_dir, subdir)
        shutil.copytree(src_path, dst_path)
    print("Done.")