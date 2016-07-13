from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import re
import operator
import collections
import numpy as np

RGX_HOUR = '([0-9]{1,2}[h|H|:][0-9]{0,2})'
RPL_HOUR = 'XXhXX'
CURRENCIES = 'eur|e|usd|gbp|euro|euros|eu|dollar|dollars'
RGX_PRICE = '([0-9]{1,}\s*)(%s)(\W|\s|$)|(\s)(%s)(\s*[0-9]{1,})' % (CURRENCIES, CURRENCIES)
MONTHS = 'janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|jan|fev|mar|avr|sep|sept|oct|nov|dec'
RGX_DATA = '([0-9]{1,}\s*)(%s)(\s*[0-9]{2,4})|([0-9]{1,}\s*)(%s)' % (MONTHS, MONTHS)


def replace_time(string_):
    times_ = re.findall(RGX_HOUR, string_)
    for i in times_:
        string_ = string_.replace(i, 'XXTIMEXX')
    return string_


def replace_date(string_):
    dates_ = re.findall(RGX_DATA, string_)
    dates_ = [''.join(x) for x in dates_]
    for i in dates_:
        string_ = string_.replace(i, 'XXDATEXX')
    return string_


def replace_price(string_):
    prices_ = re.findall(RGX_PRICE, string_)
    prices_ = [''.join(x) for x in prices_]
    for i in prices_:
        string_ = string_.replace(i, 'XXPRICEXX')
    return string_


def clean_string(string_, stopwords):
    return replace_date(
                replace_time(
                    replace_price((string_.lower()))))


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
    sentences = [clean_string(x[:x.rfind('/'):], stopwords) for x in data]
    # tf-idf
    tfidf = TfidfVectorizer(min_df=3, stop_words=stopwords, ngram_range=(1, 4))
    _ = tfidf.fit_transform(sentences)
    idf_ = tfidf.idf_
    representation =  collections.OrderedDict(dict(zip(tfidf.get_feature_names(), idf_)))
    #sorted_representation = sorted(representation.items(), key=operator.itemgetter(1), reverse=True) 
    size_voc = len(representation.keys())
    print 'Length of vocabulary %i' % (size_voc)

    X = [map_to_representation(sentence, representation, size_voc) for sentence in sentences]

    acc_mean_test = []
    acc_mean_train = []
    for i in xrange(1000):
        print i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
        print len(X_train), len(X_test)
        clf = OneVsRestClassifier(SVC(random_state=0))
        #clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10))
        clf.fit(X_train, y_train)
        accuracy_train = accuracy_score(y_train ,clf.predict(X_train))
        accuracy_test = accuracy_score(y_test ,clf.predict(X_test))
        print 'Accuracy train is: ', accuracy_train
        print 'Accuracy test is: ', accuracy_test
        acc_mean_train.append(accuracy_train)
        acc_mean_test.append(accuracy_test)
    print 'Mean accuracy train: ', np.mean(acc_mean_train)
    print 'Mean accuracy test: ', np.mean(acc_mean_test)


if __name__ == '__main__':
    main()
