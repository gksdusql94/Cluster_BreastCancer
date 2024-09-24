# Breast Cancer Dataset Clustering and Dimensionality Reduction

This project demonstrates the use of **K-Means clustering** and **Singular Value Decomposition (SVD)** to analyze the well-known **Breast Cancer Wisconsin dataset** using Apache Spark on Colab.

### Steps:
1. **Data Preprocessing:**
   - The Breast Cancer dataset is loaded using scikit-learn.
   - Data is converted into a Pandas DataFrame, tuned for schema, and finally converted into a Spark DataFrame.

2. **K-Means Clustering:**
   - K-Means algorithm is applied with k=2 to classify the dataset into two clusters (potentially corresponding to malignant and benign tumors).
   - A **Silhouette Score** of **0.834** was achieved, indicating strong cluster cohesion and separation.

3. **Dimensionality Reduction:**
   - Singular Value Decomposition (SVD) is used to reduce the dataset's dimensionality by a factor of 15x.
   - Clustering is performed again on the reduced dataset, achieving a **Silhouette Score** of **0.835**, showing that the dimensionality reduction did not significantly affect clustering performance.

4. **Results Comparison:**
   - Clustered results are compared to ground truth labels.
   - Correctly clustered data points were identified before and after dimensionality reduction.

5. **Dimensionality Reduction Efficiency:**
   - The project explores how feature reduction optimizes computational performance while preserving model accuracy.


### Dependencies:
- `pyspark`
- `scikit-learn`
- `pandas`
- `numpy`

### Running the Code:
1. Install the necessary packages in Colab using:
   ```bash
   !pip install pyspark
   !pip install -U -q PyDrive
   !apt install openjdk-8-jdk-headless -qq
   
### Result:
   Reduction Utilized the Breast Cancer Wisconsin dataset to apply unsupervised learning and dimensionality reduction techniques for efficient data analysis.
-	Classified the data into two clusters (Benign and Malignant) using K-Means clustering, achieving a Silhouette score of 0.834, demonstrating strong intra-cluster cohesion. 
-	Applied Singular Value Decomposition (SVD) to reduce the dataset's dimensionality by 15 times while maintaining a Silhouette score of 0.835, ensuring the model's accuracy and efficiency post-reduction. 
-	This confirmed that dimensionality reduction did not significantly impact analysis results while reducing computational costs

