# Geometric Transformer for End-to-End Molecule Properties Prediction

Implementation of the Geometric Transformer described in ["Geometric Transformer for End-to-End Molecule Properties Prediction
" (IJCAI 2022)](https://arxiv.org/abs/2110.13721).

<p align="center">
<img src="https://user-images.githubusercontent.com/10303791/159278549-d00ab6ee-6e54-473e-b62a-8acdcd079253.PNG" width="550px">
 <img src="https://user-images.githubusercontent.com/10303791/159279942-3a80030b-e634-46aa-b85f-5fa192ab8f1a.PNG" width="250px">
</p>

## Abstract

Transformers have become methods of choice in many applications thanks to their ability to represent complex interaction between elements. 
However, extending the Transformer architecture to non-sequential data such as molecules and enabling its training on small datasets remain a challenge. 
In this work, we introduce a Transformer-based architecture for molecule property prediction, which is able to capture the geometry of the molecule. 
We modify the classical positional encoder by an initial encoding of the molecule geometry, as well as a learned gated self-attention mechanism. 
We further suggest an augmentation scheme for molecular data capable of avoiding the overfitting induced by the overparameterized architecture. 
The proposed framework outperforms the state-of-the-art methods while being based on pure machine learning solely, i.e. the method does not incorporate domain knowledge from quantum chemistry and does not use extended geometric inputs beside the pairwise atomic distances.


## Install
- Pytorch 1.11.1
- Schnetpack (Data management)

## Script
Use the following command to train, on GPU 0, the proposed Geometric Transformer on the Atomization Energy property (U0):

`python Main.py --gpus=0 --property=U0`
## Reference

        @article{choukroun2021geometric,
        title={Geometric Transformer for End-to-End Molecule Properties Prediction},
        author={Choukroun, Yoni and Wolf, Lior},
        journal={arXiv preprint arXiv:2110.13721},
        year={2021}
        }
## License
This repo is MIT licensed.
