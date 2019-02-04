import warnings

warnings.filterwarnings("ignore")  # Ignoring unnecessory warnings
import sys
import string
import pickle
import numpy as np  # for large and multi-dimensional arrays
import pandas as pd  # for data manipulation and analysis
import nltk  # Natural language processing tool-kit
from sklearn.feature_extraction.text import CountVectorizer  # For Bag of words


def classify(ranked_text):
    """
    Used for the classification of the sentences
    :parameter ranked_text (list)
    """
    print(ranked_text)
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
    testing_data = count_vector.transform(ranked_text)

    from sklearn.naive_bayes import MultinomialNB
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    filename = 'News_finalized_model.sav'
    pickle.dump(naive_bayes, open(filename, 'wb'))
    filename = 'News_finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    predictions_nb = model.predict(testing_data)
    print(predictions_nb)


# function to read CSV dataset
def read_csv(FileName):
    """
    This Function will read the CSV input file
    :param str FileName : The path to the input file
    :rtype dataframe : return the dataframe of the pandas library
    :return : return the pandas dataframe of the input file

    """
    data_threads = pd.read_csv(FileName, encoding='latin-1')
    return data_threads


# Function to remove duplicates
def remove_duplicate(data_threads):
    """
        This Function will remove duplicates of the dataframe.
        :param dataframe data_threads: pandas dataframe having columns thread_number and text
        :rtype dataframe : return the dataframe of the pandas library
        :return : return the pandas dataframe after removed the duplicates

    """

    final_data = data_threads.drop_duplicates(subset={"thread_number", "text", "retweets", "likes", "replies"})
    return final_data


def get_needed_data(final_data):
    final_thread_number = final_data['thread_number']
    final_text = final_data['text']
    return final_thread_number, final_text


def unwanted_text_removal(final_text):
    """
           This Function will remove duplicates of the dataframe.
           :param final_text : pandas dataframe having columns thread_number and text
           :rtype dataframe : return the dataframe of the pandas library
           :return : return the pandas dataframe after removed HTML tags, Removing Punctuations
       """
    import re
    temp = []
    for sentence in final_text:
        sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', sentence)
        sentence = sentence.lower()
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)  # Removing HTML tags
        sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations
        words = [word for word in sentence.split()]
        temp.append(words)

    return temp


def combine_words_to_sentence(final_text):
    """
    make sentences from words
    :param final_text: list of words
    :return: merged sentences
    """
    temp = []
    for row in final_text:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        temp.append(sequ)
    return temp


def vect_conversion(final_text):
    """
    Convert the Text into the Vectorized form using Counter Vector max_features=5000
    :param final_text: list of words or sentences
    :return: vectorized form of the text
    """
    count_vect = CountVectorizer(max_features=5000)
    vect_data = count_vect.fit_transform(final_text)
    return vect_data


def text_rank(final_text, vect_data, no_of_line=5):
    """
    Implemented Text Rank algorithm, does the ranking of the statements as given input
    :rtype: dataframe
    :param final_text: preprocessed text data
    :param vect_data: vectored data of the final_text
    :param no_of_line: no of summarized line, default value is 5
    :return: ranked sentences as per text rank algorithm
    """
    sim_mat = np.zeros([len(final_text), len(final_text)])
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(final_text)):
        for j in range(len(final_text)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(vect_data[i], vect_data[j])[0, 0]

    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(final_text)), reverse=True)

    return ranked_sentences[:no_of_line]


# summarize all complete thread
def summarization(text,No_of_sentences=5):
    """
    takes the dataframe and return top N sentences (Summarized Text)
    :param text:  pandas dataframe having columns thread_number and text
    :return: N ranked sentences
    """
    final_text = unwanted_text_removal(text)
    final_text = combine_words_to_sentence(final_text)
    vect_data = vect_conversion(final_text)
    summarized_data = text_rank(final_text, vect_data)
    ranked_text = []
    for i in range(0, No_of_sentences):
        ranked_text.append(summarized_data[i][1])

    return ranked_text


def text_rank_output(FilePath):
    """
    Need specific type of dataset which contains

    Thread_number - No of that conversation

    Text - Conversation of that Thread

    :param str FilePath: path of the input File
    :return:
    """
    data_thread = read_csv(FilePath)
    data = remove_duplicate(data_thread)
    thread_number, text = get_needed_data(data)

    count = 0
    g_count = 0
    redundent = thread_number[0]
    one_complete_thred = []

    for thread_iterator in thread_number:
        if thread_iterator == redundent:
            one_complete_thred.insert(count, text[g_count])
            count += 1
            g_count += 1

        else:
            # print(one_complete_thred)
            top_senteces=summarization(one_complete_thred)
            classify(top_senteces)
            one_complete_thred.clear()
            count = 0
            one_complete_thred.insert(count, text[g_count])
            count += 1;
            g_count += 1
            redundent = thread_iterator