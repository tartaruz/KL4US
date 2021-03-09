import pandas as pd
import pickle
import spacy
from spacy.lang.nb import Norwegian
# from spacy.tokenizer import Tokenizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import NorwegianStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


stemmer = NorwegianStemmer() 

nltk.download('stopwords')
# print([f+"\n" for f in dir(Norwegian())])

class NLP():
    def __init__(self, data):
        self.data = data
        self.nlp = spacy.load('nb_core_news_sm')
        self.vectors = None


    def preprosses_df(self):
        self.info("Removing None and duplicates in data")
        self.data = self.data.drop_duplicates(["title"])
        self.data = self.data.dropna(subset=('title',))

  
    def get_titles(self):
        self.info("Getting the titles")
        titles = self.data["title"]
        titles = [title for title in titles]
        return titles

    def remove_noice(self, titles):
        self.info("Removing the noice in the titles")
        PUNCT = string.punctuation+"«»"
        for index, title in enumerate(titles):
            titles[index] = title.translate(str.maketrans('', '', PUNCT))
        return titles

    def normalize(self, titles):
        self.info("Stemming the words")
        for index, title in enumerate(titles):
            titles[index] = [stemmer.stem(word) for word in title]
        return titles

    def remove_stopwords(self, titles):
        self.info("Removing the stopwords in the titles")

        for index, title in enumerate(titles):
            titles[index] = [word for word in title if not word in stopwords.words("norwegian")]
        return titles

    
    def tokenize(self, titles):
        self.info("Tokenizing and lowering the word in titles")
        for index in range(len(titles)):
            titles[index] =  nltk.word_tokenize(titles[index].lower())
        return titles

    def do_magic(self):
        self.info("Init prosses")
        self.preprosses_df()
        titles = self.get_titles()

        titles = self.remove_noice(titles)

        titles = self.tokenize(titles)
        
        titles = self.remove_stopwords(titles)
        
        titles = self.normalize(titles)

        self.data["new_title"] = titles
        self.TFIDF()

    def TFIDF(self):
        self.info("Vectorizing the corpus with TFIDF")
        vectorizer = TfidfVectorizer()
        corpus = [" ".join(sentenc) for sentenc in self.data["new_title"].tolist()]
        self.vectors = vectorizer.fit_transform(corpus)
        # denselist = vectors.todense().tolist()

    def compare(self, title, k=10):
        self.info("Comparing title with all other titles")
        q_index = self.data[self.data["title"]==title].index
        
        if len(q_index)<=0:
            return None
        
        similarity = cosine_similarity(self.vectors)[q_index].tolist()[0]
        similarities = [(i,v) for i,v in enumerate(similarity)]
        similarities.sort(key=lambda x:x[1], reverse=True)
        
        #Remove itself since it compares to all enetries in df
        similarities.pop(0)
        results = [self.data.iloc[res[0]] for res in similarities]
        return similarities[:k+1], results

    def info(self, text):
        line = "="*(75-len(text))
        print(f"{line}=> {text}")



    

path = "CBF/data/df_pickled"
df = pickle.load(open(path, "rb"))
title = df["title"][69]
nlp = NLP(df)

nlp.do_magic()

res, res_array = nlp.compare(title)
print(res_array)

# print(nlp.data["title"])




