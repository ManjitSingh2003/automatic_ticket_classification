# Automatic Ticket Classification

This repository contains a project focused on automating the classification of customer support tickets for a financial company. The goal is to categorize these tickets based on the products or services mentioned in the complaints, enabling efficient assignment to the appropriate department for timely resolution.

## Problem Statement

Customer complaints are a critical indicator of a company’s service and product performance. Manually classifying and routing these tickets to the appropriate department is time-consuming and becomes increasingly challenging as the customer base grows. This project aims to automate this process by building a model that can classify unstructured text data into predefined categories, facilitating quicker response and improved customer satisfaction.

## Business Objective

Build a model to classify customer complaints based on products/services using Non-Negative Matrix Factorization (NMF). With this model, complaints can be automatically mapped to the following categories:

1. Credit card / Prepaid card
2. Bank account services
3. Theft/Dispute reporting
4. Mortgages/loans
5. Others

Once the complaints are categorized, a supervised learning model such as Logistic Regression, Decision Tree, or Random Forest can be trained using the resulting clusters to predict the categories for new incoming complaints.

## Dataset

The dataset used for this analysis is too large to be hosted directly in this repository. You can download the dataset from the following Google Drive link:

[**Dataset Download Link**](https://drive.google.com/file/d/1Y4Yzh1uTLIBLnJq1_QvoosFx9giiR1_K/view?usp=sharing)

The dataset used for this project is in JSON format and contains 78,313 customer complaints with 22 features. This data was converted into a Pandas DataFrame for processing and analysis.

## Project Structure

The project is divided into the following tasks:

1. **Data Loading**: Loading and preparing the JSON data.
2. **Text Preprocessing**: Cleaning and transforming the textual data.
3. **Exploratory Data Analysis (EDA)**: Understanding the dataset using visualizations and descriptive statistics.
4. **Feature Extraction**: Using TF-IDF to extract relevant features from the text.
5. **Topic Modelling**: Applying NMF to discover hidden topics and patterns in the complaints.
6. **Supervised Model Building**: Training models such as Logistic Regression, Decision Tree, and Random Forest on the clustered data.
7. **Model Evaluation**: Evaluating the performance of the models using metrics like accuracy, precision, recall, and F1-score.
8. **Model Inference**: Predicting the categories for new, unseen complaints using the trained model.

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, nltk, spacy, matplotlib, seaborn, plotly

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ManjitSingh2003/automatic_ticket_classification.git
   cd automatic_ticket_classification
   ```

2. Open the Jupyter Notebook `Automatic_Ticket_Classification_Assignment.ipynb` to explore the code and results.

## Results

The best-performing model was selected based on evaluation metrics, which efficiently classifies the customer complaints into their respective categories.
