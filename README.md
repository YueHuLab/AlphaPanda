
AlphaPanda README

Overview
AlphaPanda (AlphaFold2 inspired Protein-specific antibody design in a diffusional manner)is a software package designed for advanced protein design and modeling tasks. This package includes modules for dataset handling, model training, inference, and evaluation, with a focus on flexibility and high performance in computational biology.Inspired by AlphaFold2 and other protein design methods combined with diffusion generation model, we propose AlphaPanda algorithm. We have written AlphaPanda based diffab and 3DCNN, which were written by python and pytorch. Because diffab program is mainly for antibody design, we added the 3DCNN program as a module to diffab. 

Table of Contents
1. Installation
2. Software Structure
3. Training
4. Inference and Design
5. Usage Examples
6. License
7. Contributing
8. Contact

Installation

To install AlphaPanda, follow these steps:

1. Clone the Repository
   Download the AlphaPanda package from the repository:
   ```
   git clone https://github.com/YourUsername/AlphaPanda-main.git
   cd AlphaPanda-main/
   ```

2. Set Up the Environment
   AlphaPanda uses Conda for environment management. Create the necessary environment using the provided AlphaPanda_env.yml file:
   ```
   conda env create -f AlphaPanda_env.yml
   ```

3. Activate the Environment
   Once the environment is created, activate it:
   ```
   conda activate alphapanda
   ```

Software Structure

The directory structure of AlphaPanda is organized as follows:

- AlphaPanda: Core modules and scripts for dataset handling, model training, and inference.
  - datasets/: Dataset processing scripts.
  - models/: Model architecture definitions.
  - modules/: Common utilities and functions.
  - tools/: Tools for docking, evaluation, relaxation, and renumbering.
  - utils/: Utility functions and scripts for training and data processing.
- configs/: Configuration files for training and design tasks.
- data/: Directory for storing datasets and results.
- logs/: Logs generated during training and inference.
- train_AlphaPanda.py: Script for training models.
- design_AlphaPanda.py: Script for performing protein design.

Training

To train a model with AlphaPanda:

1. Prepare the Configuration
   Make sure your configuration file is set up correctly. The default configurations are located in the configs/YueTrain/ directory.

2. Run the Training Script
   Execute the training script with the appropriate parameters:
   ```
   python train_AlphaPanda.py --device cpu --num_workers 0 --logdir /path/to/logs /path/to/config.yml
   ```
   Replace /path/to/logs and /path/to/config.yml with the actual paths.

Inference and Design

To perform inference and design tasks with AlphaPanda:

1. Prepare the Input
   Ensure that the input PDB file and the configuration file for the design are ready.

2. Run the Design Script
   Use the following command to execute the design:
   ```
   python design_AlphaPanda.py /path/to/input.pdb --config /path/to/config.yml --device cpu
   ```
   Replace /path/to/input.pdb and /path/to/config.yml with your file paths.

Usage Examples

- Training Example:
  ```
  python train_AlphaPanda.py --device gpu --num_workers 4 --logdir ./logs ./configs/YueTrain/codesign_single_yue201.yml
  ```
- Design Example:
  ```
  python design_AlphaPanda.py ./data/testPDB/8en0.pdb --config ./configs/YueTest/codesign_single_yueTest200.yml --device gpu
  ```
#For training, download all_structures.zip file of the SAbDab dataset from https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/ , and then extract to data/ directory.

#For inference, checkponits examples can be download from [checkpoints/ directory](https://huggingface.co/datasets/YueHuLab/AlphaPanda_checkpoints).

License

AlphaPanda is licensed under the Apache 2.0 License. For more details, see the LICENSE file.

Contributing

Contributions are welcome! 

Contact

For any questions or issues, please contact the project maintainers at huyue@qlu.edu.cn



Our integration and improvements of the code are reflected in the following tree, and we have also provided an interface for future integration of protein language models.

AlphaPanda-main/
├── AlphaPanda
│   ├── datasets
│   │   ├── _base.py
│   │   ├── custom.py
│   │   ├── __init__.py
│   │   └── sabdab.py
│   ├── models
│   │   ├── _base.py
│   │   ├── diffab.py
│   │   └── __init__.py
│   ├── modules
│   │   ├── common
│   │   │   ├── geometry.py
│   │   │   ├── layers.py
│   │   │   ├── so3.py
│   │   │   ├── structure.py
│   │   │   └── topology.py
│   │   ├──# dcnn
│   │   │   ├── common
│   │   │   │   ├── atoms.py
│   │   │   │   ├── logger.py
│   │   │   │   └── run_manager.py
│   │   │   ├── LICENSE
│   │   │   ├── load_and_save_bb_coords.py
│   │   │   ├── load_and_save_coords.py
│   │   │   ├── README.md
│   │   │   ├── requirements.txt
│   │   │   ├── run.py
│   │   │   ├── seq_des
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models_0716002.py
│   │   │   │   ├── models_bk0716.py
│   │   │   │   ├── models.py
│   │   │   │   ├── sampler.py
│   │   │   │   └── util
│   │   │   │       ├── acc_util.py
│   │   │   │       ├── canonicalize0718_back.py
│   │   │   │       ├── canonicalize_bak0716.py
│   │   │   │       ├── canonicalize_bk0823003.py
│   │   │   │       ├── canonicalize_bk0823.py
│   │   │   │       ├── canonicalize.py
│   │   │   │       ├── data.py
│   │   │   │       ├── __init__.py
│   │   │   │       ├── pyrosetta_util.py
│   │   │   │       ├── sampler_util.py
│   │   │   │       ├── voxelize_bk0822.py
│   │   │   │       └── voxelize.py
│   │   │   ├── seq_des_info.pdf
│   │   │   ├── SETUP.md
│   │   │   ├── train_autoreg_chi_baseline.py
│   │   │   ├── train_autoreg_chi.py
│   │   │   └── txt
│   │   │       ├── test_domains_s95.txt
│   │   │       ├── test_idx.txt
│   │   │       └── train_domains_s95.txt
│   │   ├── diffusion
│   │   │   ├── dpm_full.py
│   │   │   ├── getBackAnlge.py
│   │   │   └── transition.py
│   │   └── encoders
│   │       ├── ga.py
│   │       ├── pair.py
│   │       └── residue.py
│   ├── tools
│   │   ├── dock
│   │   │   ├── base.py
│   │   │   └── hdock.py
│   │   ├── eval
│   │   │   ├── base.py
│   │   │   ├── energy.py
│   │   │   ├── __main__.py
│   │   │   ├── run.py
│   │   │   ├── run_single.py
│   │   │   └── similarity.py
│   │   ├── relax
│   │   │   ├── base.py
│   │   │   ├── __main__.py
│   │   │   ├── openmm_relaxer.py
│   │   │   ├── pyrosetta_relaxer.py
│   │   │   ├── run.py
│   │   │   └── run_single.py
│   │   ├── renumber
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   └── run.py
│   │   └── runner
│   │       ├── design_for_pdb.py
│   │       ├── design_for_testset0801.py
│   │       └── design_for_testset.py
│   └── utils
│       ├── data_bk.py
│       ├── data.py
│       ├── inference.py
│       ├── misc.py
│       ├── protein
│       │   ├── constants.py
│       │   ├── parsers.py
│       │   └── writers.py
│       ├── residue_constants.py
│       ├── train.py
│       ├── transforms
│       │   ├── _base.py
│       │   ├── __init__.py
│       │   ├── mask.py
│       │   ├── merge.py
│       │   ├── patch.py
│       │   └── select_atom.py
│       └── utils_trx.py
├── AlphaPanda_env.yml
├── configs
│   ├── YueTest
│   │   └── codesign_single_yueTest200.yml
│   └── YueTrain
│       └── codesign_single_yue201.yml
├── data
│   ├── examples
│   ├── processed
│   └── sabdab_summary_all.tsv
├── design_AlphaPanda.py
├── diffab_license
│   ├── LICENSE_diffab
│   └── README_diffab.md
├── env.yaml
├── LICENSE_AlphaPanda
├── logs
├── residue_constants.py
├── results
└── train_AlphaPanda.py


It also provides the flexibility needed for potential future enhancements, such as incorporating advanced protein language models, which can further extend the capabilities of the AlphaPanda framework (ESM model here).
https://github.com/YueHuLab/AlphaPanda/tree/AlphaPanda_PLM

AlphaPanda_PLM/
├── datasets
│   ├── _base.py
│   ├── custom.py
│   ├── __init__.py
│   └── sabdab.py
├── models
│   ├── _base.py
│   ├── diffab.py
│   └── __init__.py
├── modules
│   ├── alphafold2_pytorch
│   │   ├── alphafold2.py
│   │   ├── BK__init__.py
│   │   ├── constants.py
│   │   ├── embeds.py
│   │   ├── mlm.py
│   │   ├── reversible.py
│   │   ├── rotary.py
│   │   └── utils.py
│   ├── common
│   │   ├── geometry.py
│   │   ├── layers.py
│   │   ├── so3.py
│   │   ├── structure.py
│   │   └── topology.py
│   ├── dcnn
│   │   ├── common
│   │   │   ├── atoms.py
│   │   │   ├── logger.py
│   │   │   └── run_manager.py
│   │   ├── imgs
│   │   ├── LICENSE
│   │   ├── load_and_save_bb_coords.py
│   │   ├── load_and_save_coords.py
│   │   ├── pdbs
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── seq_des
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── sampler.py
│   │   │   └── util
│   │   │       ├── acc_util.py
│   │   │       ├── canonicalize.py
│   │   │       ├── data.py
│   │   │       ├── __init__.py
│   │   │       ├── pyrosetta_util.py
│   │   │       ├── sampler_util.py
│   │   │       └── voxelize.py
│   │   ├── SETUP.md
│   │   ├── train_autoreg_chi_baseline.py
│   │   ├── train_autoreg_chi.py
│   │   └── txt
│   │       ├── test_domains_s95.txt
│   │       ├── test_idx.txt
│   │       └── train_domains_s95.txt
│   ├── diffusion
│   │   ├── alphafold2.py
│   │   ├── dpm_full_evoformer.py
│   │   ├── dpm_full.py
│   │   ├── evoformer_huyue.py
│   │   ├── getBackAnlge.py
│   │   └── transition.py
│   ├── encoders
│   │   ├── ga.py
│   │   ├── pair.py
│   │   └── residue.py
│   └── GeoTrans
│       ├── augmentation_collate_fun.py
│       ├── GeometricTransformer_0428.py
│       ├── GeometricTransformer.py
│       ├── LICENSE
│       ├── Main.py
│       ├── README.md
│       └── Utils.py
├── tools
│   ├── dock
│   │   ├── base.py
│   │   └── hdock.py
│   ├── eval
│   │   ├── base.py
│   │   ├── energy.py
│   │   ├── __main__.py
│   │   ├── run.py
│   │   ├── run_single.py
│   │   └── similarity.py
│   ├── relax
│   │   ├── base.py
│   │   ├── __main__.py
│   │   ├── openmm_relaxer.py
│   │   ├── pyrosetta_relaxer.py
│   │   ├── run.py
│   │   └── run_single.py
│   ├── renumber
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── run.py
│   └── runner
│       ├── design_for_pdb.py
│       ├── design_for_pdb_Yue.py
│       └── design_for_testset.py
└── utils
    ├── data_bk.py
    ├── data.py
    ├── inference.py
    ├── misc.py
    ├── protein
    │   ├── constants.py
    │   ├── parsers.py
    │   └── writers.py
    ├── residue_constants.py
    ├── train.py
    ├── transforms
    │   ├── _base.py
    │   ├── #esmTrans.py (Protein Language model here)
    │   ├── __init__.py
    │   ├── mask.py
    │   ├── merge.py
    │   ├── patch.py
    │   └── select_atom.py
    └── utils_trx.py

We believe that making the code publicly available will contribute to advancing the entire research field."

