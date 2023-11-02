This repository consists of Python codes used in the paper - "Trade-offs in the latent representation of microstructure evolution"
by Saaketh Desai, Ankit Shrivastava, Marta D'Elia, Habib N. Najm, Remi Dingreville from Sandia National Laboratories.

Repository consists of two main directories:
- models: this directory contains all codes used to train and test models such as autoencoders, pca, and diffusion maps, used in the paper
- analysis: this directory contains all codes used to analyze microstructures, for e.g, analyze sizes of local features in microstructures

The models directory has the following structure:
- Model type (PCA, autoencoders, diffusion maps)
    - Physical phenomenon (Spinodal decomposition, Physical vapor deposition, Dendritic evolution, Grain growth)
        - <Codes> (Python files)

Note that the codes have various structures depending on the specific model. For instance, PCA models have one single Python script that trains for multiple latent dimensions, while autoencoders have multiple directories with each directory containing scripts for one latent dimension.

The analysis directory is structured according the codes using for each figure in the manuscript. For instance, the sub-directory "figure 2_9" consists of codes used for Figures 2 and 9 in the manuscript. 


For any questions, please contact Saaketh Desai (saadesa@sandia.gov) or Ankit Shrivastava (ashriva@sandia.gov)
