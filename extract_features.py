import re
import numpy as np
import nltk
import csv
import operator
import sklearn
from sklearn.naive_bayes import MultinomialNB
nltk.download('cmudict')


sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def fvsLexical(data):
    """
    Compute feature vectors for word and punctuation features
    :param data:
    :return:
    """
    fvs_lexical = np.zeros((len(data), 3))
    fvs_punct = np.zeros((len(data), 3))
    labels_lexical = [''] * len(data)
    labels_punct = [''] * len(data)
    for e, (id, text, author) in enumerate(data):
        # print id, text
        tokens = nltk.word_tokenize(text.lower())
        words = word_tokenizer.tokenize(text.lower())
        sentences = sentence_tokenizer.tokenize(text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s)) for s in sentences])

        # update fvs_lexical and labels_lexical
        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))
        # put author label
        labels_lexical[e] = author

        # update fvs_punct and labels_punct
        # commas per sentence
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
        # put author label
        labels_punct[e] = author

    return (fvs_lexical, labels_lexical), (fvs_punct, labels_punct)

def lexicalFeatures(training_data, test_data):
    # create lexical and punctuational feature vectors
    print('processing lexical and punctuation features...')
    lexical_train, punct_train = fvsLexical(training_data)
    lexical_test, punct_test = fvsLexical(test_data)
    return (lexical_train[0], lexical_train[1], lexical_test[0]), (punct_train[0], punct_train[1], punct_test[0])

 

def syntacticFeatures(all_text_dict, test_data):
    print('processing syntactic features...')
    # make all_text_dict into list of tuple (id, text, author) so that it can be treated as a data type here
    all_data = list((None, sentences, author) for author, sentences in all_text_dict.items())
    train_fvs, train_labels = fvsSyntax(all_data)
    test_fvs, test_labels = fvsSyntax(test_data)

    return train_fvs, train_labels, test_fvs


def fvsSyntax(data):
    """
    Extract feature vector for part of speech frequencies
    """

    def token_to_pos(text):
        tokens = nltk.word_tokenize(text)
        return [p[1] for p in nltk.pos_tag(tokens)]

    texts_pos = [token_to_pos(text) for id, text, author in data]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[text.count(pos) for pos in pos_list]
                           for text in texts_pos]).astype(np.float64)
    labels_syntax = [author for id, text, author in data]

    return fvs_syntax, labels_syntax


def bagOfWordsFeatures(all_text_dict, test_data):
    print('processing bag of words features...')
    # create all of the word set
    wordset = set()
    # make all_text_dict into list of tuple (id, text, author) so that it can be treated as a data type here
    all_data = []
    for author, sentences in all_text_dict.items():
        words = word_tokenizer.tokenize(sentences.lower())
        for word in words:
            wordset.add(word)
        all_data.append((None, sentences, author))

    # Return a dictionary that maps each word from wordset to a unique index starting at 0
    # and going up to N-1, where N is the len(wordset).
    windex = {}
    sort_words = sorted(list(wordset))
    for i in range(len(sort_words)):
        word = sort_words[i]
        windex[word] = i

    # Compute the bag of words in the whole text by each author
    train_fvs, train_labels = fvsBagOfWords(all_data, windex)
    test_fvs, test_labels = fvsBagOfWords(test_data, windex)

    return train_fvs, train_labels, test_fvs


def fvsBagOfWords(data, windex):
    fvs_bow = np.zeros((len(data), len(windex)))
    labels_bow = [''] * len(data)
    for e, (id, text, author) in enumerate(data):
        all_tokens = nltk.word_tokenize(text.lower())
        fdist = nltk.FreqDist(all_tokens)
        sorted_fdist = reversed(sorted(fdist.items(), key=operator.itemgetter(1)))
        for (word, count) in sorted_fdist:
            if word not in windex:
                continue
            index = windex[word]
            fvs_bow[e, index] = count / float(len(all_tokens))

        labels_bow[e] = author
    return fvs_bow, labels_bow


def extract_mean_syllables_per_word(text):
    """
    Extracts the mean number of syllables per word from the given text.
    """
    print("tokenizing")
    words = nltk.word_tokenize(text.lower())
    print("extracting syllables")
    syllables_per_word = [len(list(nltk.corpus.cmudict.dict().get(word, [''])[0])) for word in words]
    if len(syllables_per_word) == 0:
        return 0
    else:
        return sum(syllables_per_word) / len(syllables_per_word)

def extract_mean_syllables_per_word_features(all_text_dict, test_data):
    """
    Extracts the mean number of syllables per word for each text in all_text_dict and test_data.
    Returns the feature vectors and labels for the training and test data.
    """
    print('processing mean syllables per word features...')
    
    # Extract mean syllables per word for all texts
    all_data = []
    for author, sentence in all_text_dict.items():
        print("beginning loop")
        print(sentence)
        mean_syllables_per_word = extract_mean_syllables_per_word(sentence)
        all_data.append((None, mean_syllables_per_word, author))
    for i, (text, author) in enumerate(test_data):
        print("second loop")
        mean_syllables_per_word = extract_mean_syllables_per_word(text)
        test_data[i] = (mean_syllables_per_word, author)
    
    # Extract feature vectors and labels
    train_fvs, train_labels = fvsSyntax(all_data)
    test_fvs, test_labels = fvsSyntax(test_data)
    
    return train_fvs, train_labels, test_fvs


def count_unique_words(text):
    """
    Counts the number of unique words in the given text.
    """
    #print(text)
    words = nltk.word_tokenize(text)
    return len(set(words))

def extract_unique_word_features(all_text_dict, test_data):
    """
    Extracts the number of unique words for each text in all_text_dict and test_data.
    Returns the feature vectors for the training and test data.
    """
    print('processing unique word features...')
    
    # Extract unique word count for all texts
    all_data = []
    for author, sentences in all_text_dict.items():
        for sentence in sentences:
            unique_word_count = count_unique_words(sentence)
            all_data.append((None, unique_word_count, author))
    
    test_fvs = []
    for id, text, author in test_data:
        print(type(text))
        unique_word_count = count_unique_words(text)
        test_fvs.append((None, unique_word_count))
    
    # Extract feature vectors and labels
    train_fvs, train_labels = fvsSyntax(all_data)
    
    return train_fvs, train_labels, test_fvs