# Transfer Learning & Latent Representation Clustering Pipeline

[Watch the 10-Minute Video Presentation Here](https://youtu.be/6H6r_K37kAI)

This repository implements an unsupervised machine learning pipeline to classify image data using deep feature extraction and latent space clustering. By leveraging transfer learning, the pipeline extracts expressive feature descriptors from unlabeled images, applies dimensionality reduction with whitening, and clusters the latent representations to achieve near-supervised accuracy. 

The workflow was evaluated on a 1,800-image benchmark dataset with six balanced classes, achieving **99.2% unsupervised clustering accuracy**[cite: 1, 3].

## Repository Structure

* **`steel_defect_clustering.ipynb`**: The core execution notebook containing the pipeline implementation, from image preprocessing to feature extraction, clustering, and evaluation[cite: 2, 4].
* **`Unsupervised_Steel_Defect_Classification_Report.pdf`**: A comprehensive sensitivity analysis detailing the impact of preprocessing choices, network layer selection, and PCA parameters[cite: 3, 4].
* **`Unsupervised_Steel_Defect_Classification_Slides.pdf`**: Executive presentation summarizing the methodology and key findings[cite: 1, 4].

## Pipeline Architecture

1.  **Preprocessing**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) normalizes brightness and enhances local contrast to prevent the feature extractor from indexing irrelevant noise[cite: 1, 2, 3].
2.  **Deep Feature Extraction**: A pre-trained VGG16 convolutional neural network acts as the feature extractor. High-level features are sourced from the deepest fully connected layers (`fc1`, `fc2`) rather than convolutional blocks to maximize discriminative power[cite: 1, 2, 3].
3.  **Dimensionality Reduction & Whitening**: Principal Component Analysis (PCA) reduces the 4096-dimensional features. A critical whitening step scales components to unit variance, amplifying lower-variance signals[cite: 1, 3].
4.  **Clustering & Alignment**: K-Means (`k-means++`) partitions the latent space. The Hungarian algorithm matches the unsupervised cluster IDs to ground-truth labels to allow for rigorous performance evaluation[cite: 2, 3].

## Key Findings & Sensitivity Analysis

*   **Preprocessing is Critical**: Omitting histogram equalization collapses accuracy from 99.2% to 84.7%, indicating that brightness normalization is essential for reducing intra-class variance[cite: 1, 3].
*   **Whitening Trade-offs**: Whitening dramatically boosts peak performance but requires careful component selection. Retaining 20-50 components is optimal; retaining >100 components amplifies noise and destroys cluster separability[cite: 1, 3].
*   **Initialization Robustness**: Across 5,000 initialization trials, selecting the K-Means run with the lowest relative inertia proved to be a highly reliable heuristic for identifying the global optimum[cite: 1, 3].
*   **Generalizability**: 5-fold cross-validation demonstrated a 99.0% validation accuracy using an optimal 35-component representation, proving the pipeline's robustness as a predictive classifier on unseen data[cite: 1, 3].
