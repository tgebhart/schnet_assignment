# SchNet Coding Assignment

## Dependencies

This was developed using Python 3.10 on a machine with 32GB RAM and a NVIDIA GeForce 1080 GPU (8GB) with CUDA 11.7. 

The required packages are given in `requirements.txt`. 

## Data

The data is assumed to reside in a subdirectory named `data` which contains within a `labels.txt` file with each 
protein's label and a `proteins` subdirectory with each .pdb file. 

## Output

See `notebooks/test_model.ipynb` for an example of the entire pipeline for loading the data, training, and evaluating the model.   
The model itself is a minorly altered version of the (PyG implementation)[https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html].