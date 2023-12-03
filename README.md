# An Optimization Framework for BraTS 2023

This repository contains the code developed by Kurtlab for the 2023 International Brain Tumor Segmentations (BraTS) Cluster of Challenges (see [here](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282).)
The aim of the BraTS Challenges is to push the state-of-the-art in brain tumor segmentation algorithms. That is, algorithms that can segment tumor subregions from multiparametric MRI scans of the brain.

We submitted to three challenges:
1. Adult gliomas.
2. BraTS-Africa - gliomas from patients in sub-Saharan Africa.
3. Meningiomas.

Our team ranked in the top 10 of all those who submitted, including 7th place in the BraTS-Africa challenge.

## Implementation

Our baseline model is based on the [Optimized U-Net](https://arxiv.org/abs/2110.03352) presented by Futrega et al. from NVIDIA.

We developed a framework of optimization techniques to boost model performance. These include:
1. Preprocessing the MRI data with Z-score normalization and contrast rescaling.
2. Postprocessing our predictions to remove small connected components to minimize the number of false positives predicted, to account for the [new performance metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics) introduced in BraTS 2023.
3. A transfer learning technique in which specific layers of the model are frozen before training is continued on a new dataset.

We found that this transfer learning technique successfully allowed us to transfer knowledge from the large Challenge 1 dataset of adult gliomas to the smaller and lower resolution Challenge 2 dataset for BraTS-Africa.

We performed several ablation studies to fine-tune these techniques.

## Code

The dependencies for our code are provided in the `requirements.txt` file.

We provide a brief explanation of each subpackage:
* `datasets` - the dataset class for loading training and test data for BraTS;
* `losses` - loss functions used in model training;
* `model_routines` - commonly run routines for model training, validation and inference;
* `processing` - preprocessing, postprocessing and plotting code;
* `utils` - many utility functions used in model training and inference.

### Usage

To ensure the code runs with the relative imports working correctly, navigate outside the root folder and use `python -m` to run the Python module as a script. For example, if you wanted to run the training with validation model routine, you would run the following line.
```
python -m brats2023_updated.model_routines.train_with_val
```

## Contributors

This research was conducted by Tianyi Ren, Ethan Honey and Harshitha Rebala under the supervision of Dr Mehmet Kurt. The code was developed by Ethan Honey and Harshitha Rebala. The updated version of the code and this repository was put together by Ethan Honey. The edge loss function code was developed by Agamdeep Chopra.

## Acknowledgements

We would particularly like to acknowledge the work of Futrega et al. and NVIDIA for their Optimized U-Net model that was a crucial part of our foundation, as well as their accompanying [code](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/nnUNet/notebooks/BraTS22.ipynb).

We acknowledge [MONAI](https://monai.io/) for their implementations of the Dice Score and 95% Hausdorff Distance metrics. We used these in our validation code.

We acknowledge the [cc3d Python library](https://pypi.org/project/connected-components-3d/). We used their connected component tools in our postprocessing strategies.

We acknowledge Abderezaei et al. for their development of the edge loss function. We utilized this in training our model.

Finally, we would like to acknowledge the organizers of BraTS 2023, for presenting the challenges, sharing the datasets and supporting us in the development of our algorithms.

## References

- [Futrega et al. 2021. Optimized U-Net for Brain Tumor Segmentation.](https://arxiv.org/abs/2110.03352)
- [Cardoso et al. 2022. MONAI: An open-source framework for deep learning in healthcare](https://arxiv.org/abs/2211.02701)
- [Abderezaei et al. 2022. 3D Inception-Based TransMorph: Pre- and Post-operative Multi-contrast MRI Registration in Brain Tumors](https://arxiv.org/abs/2212.04579)
- [Baid et al. 2021. The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification](https://arxiv.org/abs/2107.02314)
- [Menze et al. 2015. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)," in IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/document/6975210)
- [Bakas et al. 2017. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features](https://www.nature.com/articles/sdata2017117)
- Bakas et al. 2017. Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection. The Cancer Imaging Archive.
- Bakas et al. 2017. Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection. The Cancer Imaging Archive.
- [Adewole et al. 2023. The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa)](https://arxiv.org/abs/2305.19369)
- [LaBella et al. 2023. The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial Meningioma](https://arxiv.org/abs/2305.07642)

## To-do
- Evaluate updated model performance on BraTS-GLI and BraTS-Africa
- Produce plots of predicted segmentations for the updated model
