import pickle
import string
import pandas as pd
import sklearn

news_df = pd.read_csv("input/uci-news-aggregator.csv", sep=",")
news_df['CATEGORY'] = news_df.CATEGORY.map({'b': 1, 't': 2, 'e': 3, 'm': 4})
news_df['TITLE'] = news_df.TITLE.map(
    lambda x: x.lower().translate(str.maketrans('', '', string.punctuation))
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    news_df['TITLE'],
    news_df['CATEGORY'],
    random_state=1
)
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words='english')
training_data = count_vector.fit_transform(X_train)
print(training_data)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
filename = 'News_finalized_model.sav'
pickle.dump(naive_bayes, open(filename, 'wb'))
