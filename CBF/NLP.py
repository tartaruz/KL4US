import pandas as pd
import pickle
import spacy
from spacy.lang.nb import Norwegian
from spacy.tokenizer import Tokenizer
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class NLP():
    def __init__(self, data):
        self.data = data
        self.nlp = Norwegian()


    def get_titles(self):
        titles = self.data["title"]
        titles = [title for title in titles]
        return titles[:10]

    def remove_noice(self, titles):
        PUNCT = string.punctuation+"«»"

        for index, title in enumerate(titles):
            #Remove punctutations
            titles[index] = title.translate(str.maketrans('', '', PUNCT))

        return titles

    def normalize(self, titles):
        #lemmatazion or port stemmer?
        
        
        return titles

    def remove_stopwords(self, titles):
        for index, title in enumerate(titles):
            titles[index] = [word for word in title if not word in stopwords.words("norwegian")]
        return titles

    
    def tokenize(self, titles):
        for index in range(len(titles)):
            titles[index] =  nltk.word_tokenize(titles[index].lower())
        return titles

    def do_magic(self):
        titles = self.get_titles()

        titles = self.remove_noice(titles)

        titles = self.tokenize(titles)
        
        titles = self.remove_stopwords(titles)
        
        print(titles)
        # title = self.normalize(titles)

    def compare(self):
        pass




path = "CBF/data/df_pickled"
df = pickle.load(open(path, "rb"))
print([col for col in df])

nlp = NLP(df)

nlp.do_magic()






