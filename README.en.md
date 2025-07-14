[Versión en Español](README.md)

# Credit Card Fraud Detection with ML 🪪
This project explores the application of Machine Learning for detecting fraudulent credit card transactions. Given the inherent imbalance in this type of data (very few transactions are fraudulent), the project addresses how to handle this imbalance and compares the performance and training efficiency of two fundamental libraries: Scikit-learn and Snap ML. The goal is to quickly and accurately identify suspicious transactions, which is crucial for financial security.

## Technologies Used 🐍
- pandas: For manipulating, cleaning, and analyzing tabular data (structuring the transaction dataset).
- numpy: For efficient numerical operations, especially with data arrays.
- matplotlib: For creating static visualizations (histograms, pie charts).
- seaborn: For creating more attractive statistical visualizations.
- scikit-learn (sklearn): For data preprocessing, class balancing, and evaluation metrics.
- snapml: An optimized IBM library offering GPU/CPU accelerated ML algorithms, used here to compare its training speed.
- warnings: To control the display of warnings during script execution.
- skillsnetwork: Used for programmatic downloading of the dataset from an IBM-provided URL.

## Installation Considerations ⚙️
If you're using pip:

pip install -q 

  pandas==1.3.4 
   
  numpy==1.21.4 \
  
  seaborn==0.9.0 \
  
  matplotlib==3.5.0 \
  
  scikit-learn==0.20.1

pip install snapml

For this project, the code was written in Jupyter Notebook for Python. Additionally, we utilized the SkillNetwork connection (part of IBM) to access free databases.

## Usage Example 📎
The script performs a fraud detection analysis on credit card transactions.
 1. Data Download and Loading: The creditcard.csv dataset is downloaded and loaded into a DataFrame.
 2. Dataset Inflation: To simulate a larger data scenario, the original dataset is replicated 10 times.
 3. Class Distribution Analysis: A pie chart showing the distribution of legitimate (Class 0) and fraudulent (Class 1) transactions is displayed to illustrate the imbalance.
 4. Transaction Amount Analysis: The distribution of transaction amounts is visualized using a histogram, and key statistics (minimum, maximum, 90th percentile) are printed.
 5. Data Preprocessing: The anonymized features V1 to V28 are scaled using StandardScaler.
 6. Data Splitting: The dataset is divided into training and testing sets (70% for training, 30% for testing), maintaining class proportionality (stratified split).
 7. Model Comparison - Decision Trees: A DecisionTreeClassifier from both Scikit-learn and Snap ML is trained, and their training speeds and ROC-AUC scores are compared.
 8. Model Comparison - Support Vector Machines (SVM): A LinearSVC (linear SVM) from Scikit-learn and a SupportVectorMachine from Snap ML are trained to compare their training speeds and ROC-AUC scores.
 9. SVM Model Evaluation with Hinge Loss: The hinge_loss metric for predictions from both SVM models is calculated and printed.

## Contributions 🖨️
If you're interested in contributing to this project or using it independently, consider:
- Forking the repository.
- Creating a new branch (git checkout -b feature/new-feature).
- Making your changes and committing them (git commit -am 'Add new feature').
- Pushing your changes to the branch (git push origin feature/new-feature).
- Opening a 'Pull Request'.

## License 📜
This project is under the MIT License. Refer to the LICENSE file (if applicable) for more details.
