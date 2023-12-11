import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from kalliope.core.NeuronModule import NeuronModule, MissingParameterException

class MovieRecommender(NeuronModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Required parameters
        self.movie_id = kwargs.get('movie_id', None)
        
        if self.movie_id is None:
            raise MissingParameterException("movie_id is required")

        # Load data
        self.movies = pd.read_csv("movies.csv")
        self.ratings = pd.read_csv("ratings.csv")

        # Create the user-item matrix
        self.X, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = self.create_X(self.ratings)

        # Get similar movies
        self.similar_movies = self.find_similar_movies(self.movie_id, self.X, k=10)

        # Prepare the response message
        self.response = self.prepare_response(self.movie_id, self.similar_movies)

        # Return the response
        self.say(self.response)

    def create_X(self, df):
        N = df['userId'].nunique()
        M = df['movieId'].nunique()
        user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
        user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
        user_index = [user_mapper[i] for i in df['userId']]
        movie_index = [movie_mapper[i] for i in df['movieId']]
        X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def find_similar_movies(self, movie_id, X, k, metric='cosine'):
        neighbour_ids = []
        movie_ind = self.movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        k += 1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(X)
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1, -1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(self.movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    def prepare_response(self, movie_id, similar_ids):
        movie_titles = dict(zip(self.movies['movieId'], self.movies['title']))
        response = f"Because you watched {movie_titles[movie_id]}, you might also like:\n"
        for i in similar_ids:
            response += f"- {movie_titles[i]}\n"
        return response

# Example usage in a Kalliope brain file:
# - name: "MovieRecommender"
#   signals:
#     - order: "recommend a movie similar to movie number {{ movie_id }}"
#   neurons:
#     - MovieRecommender:
#         movie_id: "{{ movie_id }}"
