# CPT
Cross-embodiment Pre-training

# Setting up the environment 

### Create the environment

`conda env create -f environment.yml -n cpt`

If you want to use a pre-existing environment then

`conda env update --file environment.yml` 

to which you can additionally add `--prune` option to remove packages not found in `environment.yaml`

### Set environment variables

Set `PYTHONPATH` to include CPT source files

`conda env config vars set PYTHONPATH=<path/to/CPT>`

Set `PROJDIR` to include CPT source files

`conda env config vars set PROJDIR=<path/to/VideoIL>`. `PROJDIR` is the path to the parent folder containing the CPT source files above. 
By convention, the data is located at `PROJDIR/data/..` and the output files are logged to `PROJDIR/runs`.

Verify by `conda env config vars list`


## Tests

`pytest **/*.py --disable-warnings`


# Data Preparation

Copy images from subfolders to a new directory and prefix subfolder name:

```
find src -type f -exec bash -c 'cp "$0" "dest/$(basename "$(dirname "$0")")_$(basename "$0")"' {} \;
```
