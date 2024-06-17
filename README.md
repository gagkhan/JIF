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

Verify by `conda env config vars list`


