from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import re
import operator
import collections
import numpy as np
import preprocessing
from scipy.stats import entropy
import pandas as pd


def map_to_representation(string_, mapping, size_voc):
    string_ = list(set(string_.split()))
    representation_ = np.zeros(size_voc)
    for i in string_:
        if i in mapping:
            representation_[mapping.keys().index(i)] = mapping[i]
    return representation_


def main():
    # importing and cleaning data
    file = 'Data/data.train.test'
    stopwords_ = 'Data/stopwords_fr'
    with open(file) as f, open(stopwords_) as s:
        stopwords = [line.rstrip('\n') for line in s] 
        data = [line.rstrip('\n') for line in f]

    y = [x[x.rfind('/'):] for x in data]
    y = [x.rstrip(' ') for x in y]
    sentences = [preprocessing.clean_string(x[:x.rfind('/'):]) for x in data]
    # tf-idf
    tfidf = TfidfVectorizer(min_df=3, stop_words=stopwords, ngram_range=(1, 4), analyzer='word', binary=True)
    _ = tfidf.fit_transform(sentences)
    idf_ = tfidf.idf_
    representation =  collections.OrderedDict(dict(zip(tfidf.get_feature_names(), idf_)))
    #sorted_representation = sorted(representation.items(), key=operator.itemgetter(1), reverse=True) 
    size_voc = len(representation.keys())
    print(representation.keys())
    #print(representation)
    print('Length of vocabulary %i' % (size_voc))

    print(set(y))
    acc_mean_test = []
    acc_mean_train = []
    misclas_sentences = []
    y_pred = []
    y_true = []
    for i in xrange(1000):
        X_train_, X_test_, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=None)
        y_true.extend(y_test)
        X_train = [map_to_representation(sentence, representation, size_voc) for sentence in X_train_]
        X_test = [map_to_representation(sentence, representation, size_voc) for sentence in X_test_]
        #clf = OneVsRestClassifier(SVC(random_state=0, probability=True, kernel='linear'))
        clf = LogisticRegression()
        #clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10))
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred.extend(y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test) 
        proba_pred = clf.predict_proba(X_test)
        misclas_sentences_ = [(X_test_[k], y_test[k], y_pred_test[k], proba_pred[k]) for k in xrange(len(y_test)) if y_test[k] != y_pred_test[k]]
        if i % 100 == 0:
            print('iter ', i)
            print(len(X_train), len(X_test))
            print('Accuracy train is: ', accuracy_train)
            print('Accuracy test is: ', accuracy_test)
        acc_mean_train.append(accuracy_train)
        acc_mean_test.append(accuracy_test)
    print('Mean accuracy train: %.3f' % np.mean(acc_mean_train))
    print('Mean accuracy test: %.3f' % np.mean(acc_mean_test))
    print('There are %i misclassified sentences:' % len(misclas_sentences_))
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
    print('Confusion Matrix (pandas)\n', pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
    
    
    
    #for i in range(len(misclas_sentences_)):
    #    print(misclas_sentences_[i], entropy(misclas_sentences_[i][3]))
    
    #print('Test set is:')
    #for i in range(len(X_test_)):
    #    print(X_test_[i], proba_pred[i], entropy(proba_pred[i]))


if __name__ == '__main__':
    main()
