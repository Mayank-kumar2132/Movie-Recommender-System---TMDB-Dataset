# Movie Recommender System - TMDB Dataset

## Overview
This project aims to build a **Content-Based Movie Recommender System** using the **TMDB dataset**. The system recommends movies based on the similarity of their content (plot, cast, crew, etc.) to a given movie. It uses **cosine similarity** to find movies with similar features.

## Dataset
The dataset comes from **The Movie Database (TMDb)** API, providing data on movie titles, overviews, genres, production details, and more. It was adapted from the original IMDb version, now providing more accurate and recent data fields, including full cast and crew information.

### New Columns:
- `homepage`
- `id`
- `original_title`
- `overview`
- `popularity`
- `production_companies`
- `production_countries`
- `release_date`
- `spoken_languages`
- `status`
- `tagline`
- `vote_average`

### Removed Columns:
- Facebook likes fields (e.g., `actor_1_facebook_likes`, `movie_facebook_likes`)
- IMDb-specific fields (e.g., `movie_imdb_link`, `num_critic_for_reviews`)

## Project Steps

### 1. Data Preprocessing
First, the dataset is cleaned and processed to extract relevant movie features such as:
- Movie titles
- Plot overviews
- Cast and crew data
- Genres

### 2. Feature Extraction
For a **Content-Based Recommender**, we need to extract features from movie metadata such as:
- **Overview**: We use the movie plot overview to extract keywords.
- **Genres**: Movie genres are also important to determine similarities.

### 3. Similarity Calculation
We use **Cosine Similarity** to compute the similarity between two movies based on the extracted features. It measures the cosine of the angle between two vectors, where the vectors are based on the movie metadata (e.g., keywords, genres).

### 4. Recommending Movies
Once the similarity matrix is built, we can recommend movies based on the highest cosine similarity scores. When a user inputs a movie title, the system finds and recommends similar movies.

### Libraries Used:
- `pandas`: For data manipulation.
- `scikit-learn`: For computing the cosine similarity.
- `nltk`: For natural language processing tasks like tokenization.
- `numpy`: For efficient mathematical operations.

#Contributing
Feel free to open a pull request or issue if you have any suggestions for improvement!

License
This project is licensed under the MIT License.
#
This file covers the project overview, dataset, steps for building the system, installation instructions, and other key details. You can upload it to your GitHub repository to guide others through your movie recommender system project.


## Code Example

```python
# Importing the required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies_df = pd.read_csv('tmdb_dataset.csv')

# Fill missing overviews with an empty string
movies_df['overview'] = movies_df['overview'].fillna('')

# Convert the overview text into a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['original_title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Return the titles of the recommended movies
    return movies_df['original_title'].iloc[movie_indices]

# Example usage:
print(get_recommendations('Avatar'))

