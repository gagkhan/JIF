# CPT
Cross-embodiment Pre-training

# Setting up the environment 

### Create the environment variables

`conda env create -f environment.yml -n cpt`

### Set environment variables

Set `PYTHONPATH` to point the root of all the data

`conda env config vars set PYTHONPATH=<path/to/CPT>`

Verify by `conda env config vars list`