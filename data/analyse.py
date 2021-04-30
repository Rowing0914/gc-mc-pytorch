import pandas as pd

# """
# users.dat: UserID::Gender::Age::Occupation::Zip-code
# movies.dat: MovieID::Title::Genres
# ratings.dat: UserID::MovieID::Rating::Timestamp (5-star scale)
# """
#
# ratings_headers = ['userId', 'movieId', 'rating', 'timestamp']
# df_ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_headers, encoding='latin-1',
#                          engine='python')
# users_headers = ['userId', 'gender', 'age', 'occupation', 'zipCode']
# df_users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_headers, encoding='latin-1',
#                        engine='python')
# movie_headers = ['movieId', 'movieTitle', 'genre']
# df_movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=movie_headers, encoding='latin-1',
#                         engine='python')
#
# print(len(df_movies["movieId"].unique()), len(df_users["userId"].unique()), len(df_ratings["rating"].unique()))

headers = ["userId", "itemId", "rating", "timestamp"]
for i in range(5):
    train = pd.read_csv('./ml-100k/u{}.base'.format(i+1), sep='\t', header=None, names=headers, encoding='latin-1')
    test = pd.read_csv('./ml-100k/u{}.test'.format(i+1), sep='\t', header=None, names=headers, encoding='latin-1')
    print("=== u{} ===".format(i))
    print(train.shape, test.shape)
    print(train.columns)
    train_itemIds, test_itemIds = train["itemId"].unique().tolist(), test["itemId"].unique().tolist()
    intersection = list(set(train_itemIds) & set(test_itemIds))
    print("Unique movies| train: {} test: {}".format(len(train_itemIds), len(test_itemIds)))
    print("Only movies in test: {}".format(len(test_itemIds) - len(intersection)))
