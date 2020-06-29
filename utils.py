import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()

def tokenize(sen):
    return nltk.word_tokenize()

def stem(word):
    return stemmer.stem(word.lower())


def bow(tokenized_sen, all_words):
    tokenized_sen = [stem(w) for w in tokenized_sen]
    
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sen:
            bag[idx] = 1.0
        # else:
        #     bag[idx] = 0.0
        # we don;t need else as we have already array of zeros
    return bag


a = "asd dsd ? dsdsd das homeeer simp somn"

a = tokenize(a)

print(a)

words = ['extensively','responsibly','organization']

op = map(stem,words)

print(op)