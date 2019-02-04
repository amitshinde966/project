def naive_bayes(training_data, y_train, input_data):
    from sklearn.naive_bayes import MultinomialNB
    # print("NB")
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    predictions_nb = naive_bayes.predict(input_data)
    return predictions_nb


def svm(training_data, y_train, input_data):
    # print("SVM")
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(training_data, y_train)
    predictions_sv = svclassifier.predict(input_data)
    return predictions_sv


def random_forest(training_data, y_train, input_data):
    # print("rf")
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=2, random_state=0)
    regressor.fit(training_data, y_train)
    y_pred = regressor.predict(input_data)
    return y_pred


def decisoin_tree(training_data, y_train, input_data):
    from sklearn.tree import tree
    regressor = tree.DecisionTreeClassifier()
    regressor.fit(training_data, y_train)
    print("tree")
    y_pred = regressor.predict(input_data)
    print(y_pred)
