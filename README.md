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

## Project 2: