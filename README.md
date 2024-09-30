# Breast Cancer Dataset Clustering and Dimensionality Reduction

This project demonstrates the use of **K-Means clustering** and **Singular Value Decomposition (SVD)** to analyze the well-known **Breast Cancer Wisconsin dataset** using **Apache Spark** on Colab. The project includes various stages such as data preprocessing, clustering, dimensionality reduction, and comprehensive data visualization to illustrate the results.

## Steps:

### Data Preprocessing:
- The Breast Cancer dataset is loaded using `scikit-learn`.
- Data is converted into a Pandas DataFrame, adjusted for schema, and then converted into a Spark DataFrame.

### K-Means Clustering:
- The **K-Means algorithm** is applied with **k=2** to classify the dataset into two clusters (potentially corresponding to malignant and benign tumors).
- A **Silhouette Score of 0.834** was achieved, indicating strong cluster cohesion and separation.

### Dimensionality Reduction:
- **Singular Value Decomposition (SVD)** is used to reduce the dataset's dimensionality by a factor of **15x**.
- Clustering is performed again on the reduced dataset, achieving a **Silhouette Score of 0.835**, showing that the dimensionality reduction did not significantly affect clustering performance.

### Results Comparison:
- The clustered results are compared to the ground truth labels.
- Correctly clustered data points are identified before and after dimensionality reduction.

### Dimensionality Reduction Efficiency:
- The project explores how feature reduction optimizes computational performance while preserving model accuracy.

---

## Data Visualization:

Data visualizations are included to provide a clearer understanding of the clustering and dimensionality reduction results.

### 1. PCA Visualization:
- A **PCA plot** shows the Breast Cancer dataset reduced to two components, visually displaying the separation between benign and malignant tumors.

### 2. K-Means Clustering with Centroids:
- A plot visualizes the clusters formed by the K-Means algorithm, highlighting the **cluster centroids** with red markers. This illustrates how the algorithm has grouped the dataset into two clusters.

### 3. SVD Clustering:
- A similar visualization is provided after applying **SVD** for dimensionality reduction, showing that the clusters remain well-separated even after reducing the dataset’s dimensionality by 15x.

### 4. Silhouette Plot:
- A **Silhouette plot** is used to assess the quality of the clusters formed by the K-Means algorithm. The average **Silhouette Score** of **0.834** demonstrates strong intra-cluster cohesion and separation, validating the effectiveness of the clustering algorithm.

---

## Dependencies:
- `pyspark`
- `scikit-learn`
- `pandas`
- `numpy`

## Running the Code:
To run the code in Colab, install the necessary packages using:
```bash
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq


### Result:
- **K-Means clustering** was applied to classify the data into two clusters (Benign and Malignant), achieving a **Silhouette Score of 0.834**, demonstrating strong intra-cluster cohesion.
- **Singular Value Decomposition (SVD)** was used to reduce the dataset's dimensionality by **15x** while maintaining a **Silhouette Score of 0.835**, ensuring the model's accuracy and efficiency post-reduction.
- The results confirmed that dimensionality reduction did not significantly impact clustering performance, while **reducing computational costs**.

