# Data Mining - Airbnb Athens

## Overview

This project explores Airbnb listing and review data for Athens, Greece, using data mining and natural language processing techniques. The project is divided into two parts.The first part focuses on exploratory data analysis and the development of a content-based recommendation system for Airbnb listings. The second part focuses on sentiment analysis, text classification, and semantic similarity using customer reviews and word embeddings.

The analysis uses Airbnb data from 2019 and 2023 to investigate changes in short-term rentals, property prices, neighborhood characteristics, and customer reviews over time.

## Project 1: Exploratory Data Analysis & Recommendation System

The first part focuses on cleaning, processing, and exploring Airbnb listing data for Athens in 2019 and 2023, as well as developing a content-based recommendation system using listing names and descriptions.

### Main tasks

- **Data preparation**: Combine monthly CSV files, handle missing values, and investigate outliers.
- **Exploratory Data Analysis**: Analyze room types, property listings, reviews, prices, availability, and neighborhood distributions.
- **Data visualization**: Create histograms, charts, word clouds, and interactive Folium maps.
- **Amenities analysis**: Group property amenities into broader categories and visualize their distribution.
- **Price Analysis**: Calculate average prices for properties accommodating two people and categorize neighborhoods by price level.
- **Host Analysis**: Identify hosts with the highest number of property listings.
- **Year-to-Year Comparison**: Analyze changes in prices, room types, and neighborhood-level price patterns between 2019 and 2023.

### Recommendation System

- **Text preprocessing**: Combine listing names and descriptions and handle missing values.
- **TF-IDF representation**: Create a TF-IDF matrix using unigrams and bigrams, while removing stop words.
- **Cosine Similarity**: Calculate the similarity between Airbnb listings based on their TF-IDF representations.
- **Recommendations**: Store the most similar listings and implement a function that receives a listing ID and the number of recommendations and returns the most similar properties.
- **Collocation Analysis**: Use *BigramCollocationFinder* to identify the 10 word pairs that frequently occur together in the listing text.

## Project 2: Sentiment Analysis & Semantic Similarity

The second project analyzes Airbnb reviews from 2019 and 2023 using TF-IDF, word embeddings, and sentiment analysis, evaluating SVM, Random Forest, and KNN classifiers and exploring semantic similarity between words.

### Sentiment Analysis

- **Review preprocessing**: Convert comments to lowercase, remove punctuation and stop words, lemmatize words, and keep alphabetic tokens.
- **Language filtering**: Detect the language of reviews and retain English comments.
- **Hugging Face sentiment analysis**: Use the `cardiffnlp/twitter-roberta-base-sentiment-latest` model to assign sentiment labels to the reviews.
- **Sentiment dataset**: Store the review `id`, processed `comments`, and predicted `sentiment` in a separate CSV file.
- **Sentiment distribution**: Analyze and visualize the distribution of sentiment labels in the 2019 reviews.

### Sentiment Classification

The second part treats sentiment classification as a supervised machine learning problem. The annotated reviews are divided into training and test datasets and represented using two different types of features.

#### TF-IDF

* Create TF-IDF representations of the review comments.
* Store the trained TF-IDF vectorizer using Python `pickle`.
* Reuse the saved vectorizer when transforming new reviews.

#### Word Embeddings

* Train a **Word2Vec** model using the review comments.
* Represent each review using the average of the word embedding vectors of its words.
* Save the trained Word2Vec model using Python `pickle`.

### Classification Models

The following classifiers are tested with both TF-IDF features and Word2Vec-based features:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

The classifiers are evaluated using validation data and predictions are generated for the held-out test reviews.

### Model Evaluation

The classification methods are evaluated using **10-fold cross-validation** with the following metrics:

* **Precision**
* **Recall**
* **F-Measure**
* **Accuracy**

The results are compared across the different combinations of feature representation and classifier. Based on the experiments in the notebook, **TF-IDF provides better performance than the Word2Vec-based representation across the three classifiers**, with SVM and Random Forest achieving the strongest overall results.

## Semantic Similarity & Neighborhoods

The final part of Project 2 uses Word2Vec embeddings to study semantic relationships between words appearing in the Airbnb reviews.

* **Word embeddings**: Use the trained Word2Vec model to represent words as vectors.
* **Semantic neighborhoods**: Find the most similar words to a given word and construct its semantic neighborhood for a selected value of `N`.
* **Maximum neighborhood similarity**: Calculate similarity based on the maximum similarity between a word and the neighborhood of the other word.
* **Correlation of neighborhood similarities**: Compare the similarity patterns of two words and their respective neighborhoods using correlation.
* **Sum of squared neighborhood similarities**: Calculate similarity based on the sum of squared similarities between the reference words and their neighborhoods.
* **Effect of neighborhood size**: Study how changing `N` affects the three neighborhood-based similarity measures.

The experiments show that increasing the neighborhood size can affect the three similarity measures differently. The maximum similarity tends to remain relatively stable, while the sum of squared similarities increases as more neighboring words are included. The effect on correlation depends on the words added to the semantic neighborhoods.
