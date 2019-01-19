import warnings

warnings.filterwarnings("ignore")  # Ignoring unnecessory warnings
import sys
import string
import numpy as np  # for large and multi-dimensional arrays
import pandas as pd  # for data manipulation and analysis
import nltk  # Natural language processing tool-kit
from sklearn.feature_extraction.text import CountVectorizer  # For Bag of words


def classify(ranked_text):
    news_df = pd.read_csv("input/uci-news-aggregator.csv", sep = ",")
    # news_df.CATEGORY.unique()

    news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
    news_df['TITLE'] = news_df.TITLE.map(
        lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
	)
    news_df.head()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
	    news_df['TITLE'],
	    news_df['CATEGORY'],
	    random_state = 1	
	)
    from sklearn.feature_extraction.text import CountVectorizer
    count_vector = CountVectorizer(stop_words = 'english')
    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(ranked_text)
    tree(training_data,y_train,testing_data)




def NVB(training_data,y_train,testing_data):
    from sklearn.naive_bayes import MultinomialNB
    print("NB")
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    predictions_nb = naive_bayes.predict(testing_data)
    print(predictions_nb)

def SVM(training_data,y_train,testing_data):
    print("SVM")
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(training_data, y_train)
    predictions_sv = svclassifier.predict(testing_data)
    print(predictions_sv)

def RF(training_data,y_train,testing_data):
    print("rf")
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=2, random_state=0)
    regressor.fit(training_data, y_train)
    y_pred = regressor.predict(testing_data)
    print(y_pred)

def tree(training_data,y_train,testing_data):

    from sklearn.tree import tree
    regressor = tree.DecisionTreeClassifier()
    regressor.fit(training_data, y_train)
    print("tree")
    y_pred = regressor.predict(testing_data)
    print(y_pred)


#from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

	#print("Naive Bayes Analysis")
	#print("Accuracy score: ", accuracy_score(y_test, predictions_nb))
	#print("Recall score: ", recall_score(y_test, predictions_nb, average = 'weighted'))
	#print("Precision score: ", precision_score(y_test, predictions_nb, average = 'weighted'))
	#print("F1 score: ", f1_score(y_test, predictions_nb, average = 'weighted'))

# function to read CSV dataset
def read_csv():
    #print("input/twitter/five_ten.csv")
    #data_path = input("Enter Dataset Path")
    data_path="input/twitter/twentyfive_thirty.csv"
    data_threads = pd.read_csv(data_path, encoding='latin-1')
    return data_threads


# Function to remove duplicates
def remove_duplicate(data_threads):
    final_data = data_threads.drop_duplicates(subset={"thread_number", "text", "retweets", "likes", "replies"})
    return final_data


def get_needed_data(final_data):
    final_thread_number = final_data['thread_number']
    final_text = final_data['text']
    return final_thread_number, final_text


def unwanted_text_removal(final_text):
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
    temp = []
    for row in final_text:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        temp.append(sequ)
    return temp

def vect_conversion(final_text):
    count_vect = CountVectorizer(max_features=5000)
    vect_data = count_vect.fit_transform(final_text)
    return vect_data

def summerized_text(final_text,vect_data):
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
 
    no_of_line = 5
    #print(final)
    return ranked_sentences[:no_of_line]

#summarize all complete thread
def summarization(text):
    final_text = unwanted_text_removal(text)
    final_text = combine_words_to_sentence(final_text)
    vect_data=vect_conversion(final_text)
    summarized_data = summerized_text(final_text,vect_data)
    ranked_text = []
    for i in range(0,5):
    	ranked_text.append(summarized_data[i][1])
    classify(ranked_text)
    #print(summarized_data)
# calling of functions
data_thread = read_csv()
data = remove_duplicate(data_thread)
thread_number,text = get_needed_data(data)

count=0
g_count=0
redundent=thread_number[0]
one_complete_thred=[]

for thread_iterator in thread_number:
    if thread_iterator == redundent:
        one_complete_thred.insert(count, text[g_count])
        count+=1
        g_count+=1

    else:
        # print(one_complete_thred)
        summarization(one_complete_thred)
        one_complete_thred.clear()
        count=0
        one_complete_thred.insert(count, text[g_count])
        count+=1;
        g_count+=1
        redundent=thread_iterator



# for i in range(no_threads):
#     merger_list.append(text[i])


#
# final_text = unwanted_text_removal(text)
# final_text = combine_words_to_sentence(final_text)
# vect_data=vect_conversion(final_text)
# summerized_text(final_text,vect_data)
#
#
