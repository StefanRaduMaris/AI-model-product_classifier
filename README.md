Product Title Classifier
Project Description

This project implements an automated system to classify product titles into predefined categories using Natural Language Processing (NLP) and machine learning. The goal is to streamline product organization and improve search functionality for e-commerce platforms.

Key Features

Data availability: The CSV dataset is included in the repository under the folder data as products.csv.

Exploratory analysis: A detailed analysis of the dataset is provided in the notebook located in the notebooks folder: products_analysis.ipynb.

Source code: The src folder contains:

The script to train the model.

The script to test the model and make predictions on new product titles.

Text preprocessing: Product titles are converted into numerical representations using TF-IDF (TfidfVectorizer).

Classification models: Multiple algorithms were tested, including Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and Support Vector Machine (SVM). LinearSVC (SVM) was selected for production due to its high accuracy and performance.

Pipeline integration: Preprocessing and classification steps are combined in a single pipeline for easy training and prediction.

Model persistence: The trained model is saved with joblib in the model folder and can be reused for new product titles without retraining.

Results

The SVM model achieves approximately 97–98% accuracy, performing consistently across all product categories.

While other models also performed well, SVM provides the best balance between precision, recall, and inference speed.