import sys
import os
import ast
import numpy as np
import re
import argparse
import unicodedata
import urllib2
import preprocessing
from tqdm import tqdm

def remove_diacritics(word):
    word = unicode(word, 'utf-8')
    return unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')


def get_synonyms(word):
    """
    Returns the synonyms of a word.
    
    Args:
        word: a string.

    Returns:
        A list of tuples of the following form `[('synonym1', similarity1), ..., ('synonymn', similarityn)]`.
    """
    #print('Word being processed: %s' % word)
    response = urllib2.urlopen('http://www.cnrtl.fr/synonymie/%s' % word).read()
    pattern = """(?<=\<a href="\/synonymie\/)(?:.+?\>)(.+?)(?=\<\/a\>).+?(?:width=")([0-9]+)"""
    response = re.findall(pattern, response)
    #print(response)
    synonyms = []
    for synonym, score in response:
        if synonym.rfind('/') >= 0:
            synonym = synonym[:synonym.rfind('/')]
        synonyms.append((remove_diacritics(synonym), score))
    return synonyms


def sample_synonym(word, synonyms):
    word_synonym = [x for x, _ in synonyms]
    word_prob = [float(x) for _, x in synonyms]
    normalization_factor = np.sum(word_prob)
    word_prob = [x / normalization_factor for x in word_prob]
    return np.random.choice(word_synonym, p=word_prob)


def sentence_augmentation(sentence, synonyms):
    """
    Args:
        sentence: a `string` representing the sentence to process.
        synonyms: a `dictionnary` where keys are words and values are
            a `list` of `tuples` as returned by `get_synonyms`.

    Returns:
        A string with some words replaced by their synonyms.
    """
    sentence_words = sentence.split()
    eligible_words = [x for x in sentence_words if x in synonyms.keys()]
    # we fix a distribution to chose the number of words to interchange
    # the words to be interchanged could be chosen, instead of uniformly,
    # depending on the number of synonyms or the mean of the scores.
    try:
        interchange_n = np.random.randint(1, np.min([3, len(eligible_words) + 1]))
    except:
        #print('No eligible words in the sentence: %s' % sentence)
        return sentence

    words_interchange = np.random.choice(eligible_words, interchange_n, replace=False)
    for word in words_interchange:
        sentence_words[sentence_words.index(word)] = sample_synonym(word, synonyms[word])
    return ' '.join(sentence_words)
    

def parse_flags():
    parser = argparse.ArgumentParser(prog="Data Augmentation module")
    parser.add_argument('data', type=str, help='path to the data')
    parser.add_argument('stopwords', type=str, help='path to the stopwords')
    parser.add_argument('--augmentation', '-a', type=int, help='augmentation factor per sentence')
    parser.add_argument('--crawl', '-c', action='store_true', help='augmentation factor per sentence')
    return parser.parse_args()


def main():
    flags = parse_flags()
    data_path = flags.data
    stopwords_path = flags.stopwords

    with open(data_path) as f, open(stopwords_path) as g:
        data = [line.rstrip('\n') for line in f]
        stopwords = [line.rstrip('\n') for line in g]
        labels = [x[x.index('/'):] for x in data]
        data = [x[:x.index('/')] for x in data]
        data = [preprocessing.clean_string(x.rstrip(' ')).lower() for x in data]

    vocabulary = [x.split() for x in data]
    vocabulary = list(set([x for y in vocabulary for x in y if x[:2] != 'xx']))
    vocabulary = [x for x in vocabulary]

    synonyms = {}
    print os.path.join(flags.data[:flags.data.rfind('/')], 'synonyms')
    if not os.path.exists(os.path.join(flags.data[:flags.data.rfind('/')], 'synonyms')) or flags.crawl:
        print('Fetchin synonyms for %i words' % len(vocabulary))
        for word in tqdm(vocabulary):
            synonyms[word] = get_synonyms(word)
            if len(synonyms[word]) == 0:
                synonyms.pop(word, None)
        with open(os.path.join(flags.data[:flags.data.rfind('/')], 'synonyms'), 'w') as f:
            f.write(str(synonyms))
    else:
        print('Retrieving synonyms from %s' % (os.path.join(flags.data[:flags.data.rfind('/')], 'synonyms')))
        with open(os.path.join(flags.data[:flags.data.rfind('/')], 'synonyms')) as f:
            synonyms = f.read()
            synonyms = ast.literal_eval(synonyms)

    data_augmented = []
    print len(data)
    for _ in tqdm(range(flags.augmentation)):
        for i in range(len(data)):
            #print sentence
            sentence_augmented = sentence_augmentation(data[i], synonyms) + labels[i]
            data_augmented.extend([data[i] + labels[i], sentence_augmented])
    print len(data_augmented)
    data_augmented = list(set(data_augmented))
    print len(data_augmented)
    data_augmented_output = ''
    for sentence in data_augmented:
        data_augmented_output += sentence + '\n'
    with open(os.path.join(flags.data[:flags.data.rfind('/')], 'data_augmented'), 'w') as f:
        f.write(data_augmented_output)

if __name__ == '__main__':
    main()
