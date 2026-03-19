import pandas as pd
from gensim.models import Word2Vec
import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import train_test_split, precision_at_k


class MusicRecommend:
    def __init__(self, path_triplets, path_tracks, path_majority, path_track_words):
        self.path_triplets = path_triplets
        self.path_tracks = path_tracks
        self.path_majority = path_majority
        self.path_track_words = path_track_words

        self.triplets = None
        self.tracks = None
        self.majority = None
        self.vocab_dict = None
        self.track_words = None
        self.model_w2v = None

        self.triplets_codes = None
        self.user_item_matrix = None
        self.train_matrix = None
        self.test_matrix = None
        self.model_als = None

        self.load_data()
        self.create_triplets_codes()

    def load_data(self):
        # Загружаем основные таблицы
        self.triplets = pd.read_csv(
            self.path_triplets, sep='\t', header=None,
            names=['user_id', 'song_id', 'play_count']
        )

        self.tracks = pd.read_csv(
            self.path_tracks, sep='<SEP>', header=None,
            names=['track_id', 'song_id', 'artist', 'title'], engine='python'
        )

        self.majority = pd.read_csv(
            self.path_majority, sep='\t', header=None, comment='#',
            names=['track_id', 'majority_genre', 'minority_genre']
        )

        # Загружаем словарь и данные для content-based
        vocab = self.load_vocab(self.path_track_words)
        self.vocab_dict = {i + 1: word for i, word in enumerate(vocab)}
        self.track_words = self.load_mxm_dataset(self.path_track_words)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('%'):
                    return line[1:].strip().split(',')
        return []

    @staticmethod
    def load_mxm_dataset(file_path):
        data = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or line.startswith('%'):
                    continue
                parts = line.strip().split(',')
                track_id, mxm_id = parts[0], parts[1]
                counts = {int(idx): int(cnt) for idx_cnt in parts[2:] for idx, cnt in [idx_cnt.split(':')]}
                data.append({'track_id': track_id, 'mxm_track_id': mxm_id, 'word_counts': counts})
        return pd.DataFrame(data)

    def top_250_tracks(self):
        top_tracks = self.triplets.groupby('song_id')['play_count'].sum().sort_values(ascending=False).head(
            250).reset_index()
        top_tracks = top_tracks.merge(self.tracks.drop_duplicates('song_id'), left_on='song_id', right_on='song_id',
                                      how='left')
        return top_tracks[['song_id', 'artist', 'title', 'play_count']]

    def top_100_genre(self, majority_genre='Rock'):
        tracks_with_genre = self.tracks.merge(self.majority, left_on='track_id', right_on='track_id', how='left')
        top_tracks = self.triplets.groupby('song_id')['play_count'].sum().sort_values(ascending=False).reset_index()
        top_tracks = top_tracks.merge(tracks_with_genre.drop_duplicates('song_id'), left_on='song_id',
                                      right_on='song_id', how='left')
        top_100_with_genre = top_tracks[top_tracks['majority_genre'] == majority_genre].head(100)
        return top_100_with_genre[['song_id', 'artist', 'title', 'play_count']].reset_index(drop=True)

    def get_word_index(self, word):
        for idx, w in self.vocab_dict.items():
            if w.lower() == word.lower():
                return idx
        return None

    def score_track_by_word(self, word_idx):
        return self.track_words['word_counts'].get(word_idx, 0)

    def top_50_by_word(self, word, use_w2v=False, coefficient=0.2):
        word_idx = self.get_word_index(word)
        if word_idx is None:
            raise ValueError("Word not found in vocabulary")

        if use_w2v:
            try:
                similar_words_with_scores = self.model_w2v.wv.most_similar(word, topn=10)
                similar_words = [w for w, _ in similar_words_with_scores]
                print(f'Similar_words: {similar_words}')
                similar_words_idx = [self.get_word_index(w) for w in similar_words if
                                     self.get_word_index(w) is not None]
                self.track_words['score'] = self.track_words['word_counts'].map(
                    lambda wc: wc.get(word_idx, 0) + coefficient * sum(wc.get(idx, 0) for idx in similar_words_idx))
            except NameError:
                print("Word2Vec model is not available. Scoring based on the original word only.")
                self.track_words['score'] = self.track_words['word_counts'].map(lambda wc: wc.get(word_idx, 0))
        else:
            self.track_words['score'] = self.track_words['word_counts'].map(lambda wc: wc.get(word_idx, 0))

        top_tracks = self.triplets.groupby('song_id')['play_count'].sum().sort_values(ascending=False).reset_index()
        top_tracks = top_tracks.merge(self.tracks.drop_duplicates('song_id'), left_on='song_id', right_on='song_id',
                                      how='left')

        top_tracks_by_word = top_tracks.merge(self.track_words, left_on='track_id', right_on='track_id', how='left')
        top_tracks_by_word = top_tracks_by_word[top_tracks_by_word['score'] > 0].sort_values('play_count',
                                                                                             ascending=False).head(50)

        return top_tracks_by_word[['song_id', 'artist', 'title', 'play_count']].reset_index(drop=True)

    def train_w2v_model(self):
        corpus = []
        for wc in self.track_words['word_counts']:
            seq = []
            for idx, cnt in wc.items():
                seq.extend([self.vocab_dict[idx]] * cnt)
            corpus.append(seq)

        model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
        self.model_w2v = model

    def train_als_model(self):
        rows = self.triplets['user_id'].astype('category').cat.codes
        cols = self.triplets['song_id'].astype('category').cat.codes
        data = self.triplets['play_count']
        user_item_matrix = coo_matrix((data, (rows, cols))).tocsr()

        train_matrix, test_matrix = train_test_split(user_item_matrix, train_percentage=0.8, random_state=21)

        model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, iterations=10)

        model.fit(train_matrix)

        self.user_item_matrix = user_item_matrix
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.model_als = model

    def precision_at_k(self, K=10):
        if self.model_als is None or self.train_matrix is None or self.test_matrix is None:
            raise RuntimeError("ALS model or train/test matrices are not available. Run train_als_model() first.")
        # Вызов с корректным параметром K (по умолчанию 10)
        precision = precision_at_k(self.model_als, self.train_matrix, self.test_matrix, K=K)
        print(f'Precision@K: {precision:.4f}')
        return precision

    def create_triplets_codes(self):
        triplets_codes = self.triplets.copy()
        triplets_codes['user_id_code'] = self.triplets['user_id'].astype('category').cat.codes
        triplets_codes['song_id_code'] = self.triplets['song_id'].astype('category').cat.codes
        self.triplets_codes = triplets_codes

    def get_recommendations(self, user_id, N=100, max_per_artist=2):
        if self.model_als is None or self.user_item_matrix is None:
            raise RuntimeError("ALS model is not trained. Call train_als_model() before requesting recommendations.")

        user_row = self.triplets_codes.loc[self.triplets_codes['user_id'] == user_id, 'user_id_code']
        if user_row.empty:
            raise ValueError("user_id not found")
        user_code = user_row.iloc[0]

        songs, scores = self.model_als.recommend(user_code, self.user_item_matrix[user_code], N=N)

        top_10_recommendations = pd.DataFrame({
            'song_id_code': songs,
            'score': scores
        })

        song_map = self.triplets_codes[['song_id_code', 'song_id']].drop_duplicates()

        top_10_recommendations = top_10_recommendations.merge(song_map, on='song_id_code', how='left')
        top_10_recommendations = top_10_recommendations.merge(self.tracks, on='song_id', how='left')

        top_10_recommendations = top_10_recommendations.drop_duplicates(subset=['song_id']) \
            .sort_values(by='score', ascending=False)

        listened_songs = self.triplets_codes.loc[self.triplets_codes['user_id'] == user_id, 'song_id'].unique()
        top_10_recommendations = top_10_recommendations[~top_10_recommendations['song_id'].isin(listened_songs)]

        top_10_recommendations['artist_count'] = top_10_recommendations.groupby('artist').cumcount()

        top_10_recommendations = top_10_recommendations[top_10_recommendations['artist_count'] < max_per_artist]

        return top_10_recommendations[['song_id', 'artist', 'title']].head(10)

    def get_similar_items(self, item_id, N=100, max_per_artist=2):
        if self.model_als is None:
            raise RuntimeError("ALS model is not trained. Call train_als_model() before requesting similar items.")

        item_row = self.triplets_codes.loc[self.triplets_codes['song_id'] == item_id, 'song_id_code']
        if item_row.empty:
            raise ValueError("song_id not found")
        item_code = item_row.iloc[0]

        songs, scores = self.model_als.similar_items(item_code, N=N)

        top_10_similar = pd.DataFrame({
            'song_id_code': songs,
            'score': scores
        })

        song_map = self.triplets_codes[['song_id_code', 'song_id']].drop_duplicates()

        top_10_similar = top_10_similar.merge(song_map, on='song_id_code', how='left')
        top_10_similar = top_10_similar.merge(self.tracks, on='song_id', how='left')

        top_10_similar = top_10_similar.drop_duplicates(subset=['song_id']) \
            .sort_values(by='score', ascending=False)

        exclude_song = self.triplets_codes.loc[self.triplets_codes['song_id'] == item_id, 'song_id'].unique()
        top_10_similar = top_10_similar[~top_10_similar['song_id'].isin(exclude_song)]

        top_10_similar['artist_count'] = top_10_similar.groupby('artist').cumcount()

        top_10_similar = top_10_similar[top_10_similar['artist_count'] < max_per_artist]

        return top_10_similar[['song_id', 'artist', 'title']].head(10)
