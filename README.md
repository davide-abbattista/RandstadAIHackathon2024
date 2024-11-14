# Randstad AI Hackathon 2024 - Sentiment Analysis Challenge

## Challenge Description

The goal of this challenge is to build an NLP model for **binary sentiment classification** on a large dataset of hotel reviews from Tripadvisor, and to create a **visual representation** of reviews by city. The classification task is focused on labeling each review as either **positive** or **negative**. The visualization task involves analyzing review scores across cities, providing insights into sentiment patterns across different locations.

## Dataset

1. **Review Dataset**: Contains approximately **350,000 hotel reviews** labeled as either **positive** (`1`) or **negative** (`0`).
   
2. **City Dataset**: Includes information on the cities associated with each review and a corresponding sentiment score ranging from 1 to 10. This dataset is used to generate visualizations of review sentiment across cities.

## Solution Overview

This solution is divided into two main tasks, implemented across two Jupyter notebooks:

1. **1st_task.ipynb**: This notebook covers the binary sentiment classification task using a DistilBERT-based model.
2. **2nd_task.ipynb**: This notebook is dedicated to the visual representation of sentiment data by city using Folium for map-based visualization and Matplotlib for histogram plots.

## Task 1: Sentiment Classification (1st_task.ipynb)

This notebook focuses on developing a **DistilBERT-based classifier** for sentiment analysis. The process includes:
- **Data Loading**: Loading the review dataset and preparing it for model training.
- **Preprocessing and Tokenization**: Text reviews are tokenized using the DistilBERT tokenizer.
- **Model Training and Evaluation**: The model is trained on 80% of the dataset, using cross-entropy loss and evaluated with accuracy, ROC-AUC, and other classification metrics.
- **Final Model Training**: The classifier is trained on the full dataset for robust performance on new data.
- **Testing**: The final model, evaluated on a test dataset, achieved an accuracy of **0.9837**.

### Model Architecture

The DistilBERT-based classifier is built with:
- A DistilBERT layer - initialized from a pretrained DistilBERT model (specifically, "distilbert-base-uncased") - to encode tokenized reviews.
- A dropout layer to reduce overfitting.
- A fully connected layer for binary classification (positive vs. negative).

Model weights are saved for use in predictions, making the model ready for deployment or further fine-tuning.

## Task 2: Data Visualization by City (2nd_task.ipynb)

This notebook uses the **City Dataset** to visualize review sentiment geographically. Key steps include:
- **City Data Loading**: Loading the city dataset and exploring data distribution across cities.
- **Histogram Plotting**: For each city, a histogram of review scores is generated using Matplotlib. The histogram provides a visual summary of sentiment distribution for that city.
- **Geolocation and Aggregation**: Using the `geopy` library, city coordinates are obtained to map data points. For each city, average and median review scores are calculated.
- **Interactive Map Creation**: Using **Folium**, an interactive map is created, with each city marked by its location. Each marker displays:
  - Average and median review scores.
  - An embedded histogram of review score distribution.
