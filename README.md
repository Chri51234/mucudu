The python api is hosted on render.com

If you hit the endpoint and pass in the user_id, the number of offers you want generated and the venue_id you will get your response

https://mucudu-usergenoffers4jun.onrender.com//create_offer/9665?num_offers=3&venue_id=174

# Personalized Offer Generator

## Overview
This Python Flask application uses machine learning and rule mining to generate personalized offers for users based on their past transaction data. The application performs clustering, association rule mining, and collaborative filtering to recommend highly personalized discounts and offers.

## Features
- **Data Processing**: Converts transaction data from CSV and menu items from JSON for analysis.
- **Clustering**: Utilizes K-means to segment users based on their purchasing behavior.
- **Association Rule Mining**: Generates rules from transaction data to identify patterns in item purchases.
- **Collaborative Filtering**: Uses the SVD algorithm for making item recommendations.
- **Personalized Offers**: Generates personalized offers based on user cluster and rule mining.
- **REST API**: Provides endpoints to retrieve and create personalized offers for users.

## Installation

To get started with this application, clone this repository and install the required dependencies.

```bash
git clone [URL]
cd [repository-name]
pip install -r requirements.txt
