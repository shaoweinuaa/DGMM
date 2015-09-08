Identifying Multimodal Intermediate Phenotypes between Genetic Risk Factors and Disease Status in Alzheimer¡¯s Disease

Author: Xiaoke Hao,robinhc@163.com

"DGMM" is a package written in Matlab and the name stands for "Diagnosis-guided Multi-modality Phenotype Associations" . 

This repository contains the following files:

DGMM_main.m is the main function of the DGMM package. 

f_myCV.m Performs cross-validation to choose an appropriate penalty parameter and for the 5 fold test.
f_lapLabelDistMatrix.m calculates the Laplacian matrix.
f_GMTM_APG.m is our proposed algorithm that involves multi-task learning and manifold learning. 
These "f_*.m" files are all sub-routines of the DGMM main function.