
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

#For inference, checkponits examples can be seen in checkpoints/ directory.

License

AlphaPanda is licensed under the Apache 2.0 License. For more details, see the LICENSE file.

Contributing

Contributions are welcome! 

Contact

For any questions or issues, please contact the project maintainers at huyue@qlu.edu.cn
