## TODO / Project Status

### Current Pipeline

目前已经打通的 PyTorch 主线：

```text
Allen/Billeh SONATA data
-> convert2pkl.py
-> Chen-compatible network_dat.pkl / input_dat.pkl
-> load_sparse_torch.py
-> PyTorch sparse tensors
-> BillehColumnTorch
-> readout classifier
-> single-GPU training
-> DDP multi-GPU training
```

### Completed

- Converted Allen/Billeh SONATA network files into the legacy Chen-compatible pickle format.
- Loaded recurrent and input connectivity as PyTorch sparse tensors.
- Implemented and validated `BillehColumnTorch`.
- Verified CUDA forward pass on:
  - small model: `1000` neurons
  - full core model: `51978` neurons, `14M+` recurrent synapses
- Added readout classifier and single-GPU training loop.
- Added DDP multi-GPU training support.
- Verified training with synthetic toy tasks.

### Missing Original Data Files

The following files from the original TensorFlow project are currently unavailable:

```text
alternate_small_stimuli.pkl
many_small_stimuli.pkl
EA_LGN.h5
lgn_full_col_cells_3.csv
garrett_firing_rates.pkl
additive_noise.mat
```

Because of these missing files, the original task pipelines cannot yet be reproduced directly.

### Remaining Work

- Port the original `lgn.py` module to PyTorch.
- Reproduce the original `multi_train.py` training pipeline in PyTorch.
- Rebuild or replace the missing stimulus / LGN data pipeline.
- Run the original benchmark tasks:
  - Garrett task
  - evidence accumulation
  - VCD grating
  - orientation difference
  - 10-class classification

### Next Recommended Step

Before fully reproducing the original TensorFlow pipeline, add a real accessible dataset, such as MNIST or CIFAR, and convert it into `[batch, time, 17400]` input format to validate end-to-end PyTorch training on real labels.
```

# Code for papers:
- Chen, Guozhang, Franz Scherr, and Wolfgang Maass. “[A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing.](https://www.science.org/doi/full/10.1126/sciadv.abq7592)” Science Advances 8.44 (2022): eabq7592.
- Chen, Guozhang, Franz Scherr, and Wolfgang Maass. “[Data-based large-scale models provide a window into the organization of cortical computations.](https://www.biorxiv.org/content/10.1101/2023.04.28.538662v3.abstract)” bioRxiv (2023): 2023-04.

# To run the code
- Please follow the instruction in "install_a_conda_enviroment" first.
- Download supporting files from [here](https://cloud.tugraz.at/index.php/s/JmDakasAHEqsA9J)
- Make sure the file structure is like:
  dir/parent_folder/GLIF_network/network
  
  parent_folder should contain alternate_small_stimuli.pkl, many_small_stimuli.pkl, EA_LGN.h5 (these three are for dataset)
  
  GLIF_network should contain input_dat.pkl, network_dat.pkl,
  
  network contains ‘v1_node_types.csv’ and ‘v1_nodes.h5’
  
  When you run the code, you should give dir/parent_folder/GLIF_network to flag "data_dir".
  
  For LGN supporting file, you have to give it in /lgn_model/lgn.py line 88-99 where I hard coded there. Or, you can specify the path to lgn_full_col_cells_3.csv as an argument when you call LGN in stim_dataset.py.
- do ```python multi_training.py``` to train the V1 model for 5 tasks together.

# Authorship
The code package was written by [Franz Scheer](https://www.franzscherr.com/) and [Guozhang Chen](https://ifgovh.github.io/) (equally contributed).

# Citation
@article{chen2022data,
  title={A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing},
  author={Chen, Guozhang and Scherr, Franz and Maass, Wolfgang},
  journal={Science Advances},
  volume={8},
  number={44},
  pages={eabq7592},
  year={2022},
  publisher={American Association for the Advancement of Science}
}

@article{chen2023data,
  title={Data-based large-scale models provide a window into the organization of cortical computations},
  author={Chen, Guozhang and Scherr, Franz and Maass, Wolfgang},
  journal={bioRxiv},
  pages={2023--04},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

# The following paper(s) have utilized or derived from the code provided in this repository:
Fraile, J. G., Scherr, F., Sukia, J. R., Arkhipov, A., Maass, W., & Santos, C. R. M. (2023). [Prediction error computation in cortical neurons via competition between bottom-up visual input and recurrent inhibition.](https://www.ibroneuroreports.org/article/S2667-2421(23)01666-4/fulltext) IBRO Neuroscience Reports, 15, S780.
