# Formal v1 configurations

`mae.yaml`, `data2vec.yaml`, and `jepa.yaml` are the frozen Formal v1, directly runnable pretraining configurations. The methods use the same shared encoder and training protocol where applicable; each YAML encodes its method-specific masking, EMA, predictor, and loss settings.

The Slurm templates override cluster-specific manifest and processed-data paths, so no Python source editing is required.
