from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import re
import operator
import collections
import numpy as np
import preprocessing


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
    print(representation)
    print('Length of vocabulary %i' % (size_voc))

    #X = [map_to_representation(sentence, representation, size_voc) for sentence in sentences]

    acc_mean_test = []
    acc_mean_train = []
    misclas_sentences = []
    for i in xrange(1000):
        X_train_, X_test_, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=None)
        X_train = [map_to_representation(sentence, representation, size_voc) for sentence in X_train_]
        X_test = [map_to_representation(sentence, representation, size_voc) for sentence in X_test_]
        #clf = OneVsRestClassifier(SVC(random_state=0, probability=True, kernel='linear'))
        clf = LogisticRegression()
        #clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10))
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test) 
        #print('XXXXXXXXXXX')
        #print(clf.predict_proba(X_train))
        #print('XXXXXXXXXXX')

        #misclas_sentences_ = [(X_test_[i], y_test[i], y_pred_test[i]) for i in xrange(len(y_test)) if y_test[i] != y_pred_test[i]]
        misclas_sentences_ = [X_test_[i_] for i_ in xrange(len(y_test)) if y_test[i_] != y_pred_test[i_]]
        misclas_sentences.extend(misclas_sentences_)
        if i % 100 == 0:
            print('iter ', i)
            print(len(X_train), len(X_test))
            print('Accuracy train is: ', accuracy_train)
            print('Accuracy test is: ', accuracy_test)
        acc_mean_train.append(accuracy_train)
        acc_mean_test.append(accuracy_test)
    print('Mean accuracy train: %.3f' % np.mean(acc_mean_train))
    print('Mean accuracy test: %.3f' % np.mean(acc_mean_test))
    #print('List of sentences that were misclassified at least once')
    #print(len(list(set(misclas_sentences))))
    #print(list(set(misclas_sentences)))


if __name__ == '__main__':
    main()
