import pandas as pd
import pickle
import spacy
from spacy.lang.nb import Norwegian
# from spacy.tokenizer import Tokenizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import NorwegianStemmer 
  


stemmer = NorwegianStemmer() 

nltk.download('stopwords')
# print([f+"\n" for f in dir(Norwegian())])

class NLP():
    def __init__(self, data):
        self.data = data
        self.nlp = spacy.load('nb_core_news_sm')


    def fix_none_titles(self):
        doc_id2title = {}
        cols = ["documentId", "title"]
        titles = []
        nr = 0
        for index, row in self.data[cols].iterrows():
            if row["title"] == None:
                nr += 1
            key = row[cols[0]]
            if key not in doc_id2title:
                doc_id2title[key] = row[cols[1]]
        print(nr)
        
        for index, row in self.data[cols].iterrows():
            if row["title"] == None:
                nr-=1
                titles.append(doc_id2title[row["documentId"]])
            else:
                titles.append(row["title"])
        
        print(nr)
        self.data["title"] = titles

    
    def get_titles(self):
        titles = self.data["title"]
        titles = [title for title in titles]
        return titles

    def remove_noice(self, titles):
        PUNCT = string.punctuation+"«»"

        for index, title in enumerate(titles):
            #Remove punctutations
            titles[index] = title.translate(str.maketrans('', '', PUNCT))

        return titles

    def normalize(self, titles):
        #lemmatazion or portstemmer?
        # lemmatizer = self.nlp.lemmatizer()

        for index, title in enumerate(titles):
            # doc = self.nlp(" ".join(title))
            # titles[index] = doc[0].token
            titles[index] = [stemmer.stem(word) for word in title]
        
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
        self.fix_none_titles()
        titles = self.get_titles()

        titles = self.remove_noice(titles)

        titles = self.tokenize(titles)
        
        titles = self.remove_stopwords(titles)
        
        title = self.normalize(titles)
        self.data["title"] = titles

    def TFIDF(self):
        pass
    


    

path = "CBF/data/df_pickled"
df = pickle.load(open(path, "rb"))
print([col for col in df])

nlp = NLP(df)

nlp.fix_none_titles()
nlp.do_magic()





