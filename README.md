# CPT
Cross-embodiment Pre-training

# Setting up the environment 

### Create the environment variables

`conda env create -f environment.yml -n cpt`

### Set environment variables

Set `DATA_ROOT` which points to the root of all the data

`conda env config vars set DATA_ROOT=<path/to/data>`

Set `PYTHONPATH` to point the root of all the data

`conda env config vars set DATA_ROOT=<path/to/data>`

Verify by `conda env config vars list`