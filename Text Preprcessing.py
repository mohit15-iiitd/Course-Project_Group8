#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os
import nltk
from sklearn.model_selection import GridSearchCV
import numpy as np
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from transformers import TFBertTokenizer
# from nlp import nlp
import preprocessor as p
import spacy
from torch.utils.data import DataLoader
import re
nlp = spacy.load('en_core_web_sm')
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()


from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample


import torch 
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import fastText
import warnings
from tqdm import tqdm
import torch.nn as nn
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = pd.read_csv("uncleaned_combined_txt.csv")


# In[6]:


data


# In[ ]:





# In[7]:


# preprocessing funciton created
def preprocessing(df):
    df1 = df.copy()
    df1['white_space_removed'] = 0
    df1['emoji_removed'] = 0
    df1['tokenized_data'] = 0
    df1['stopword_removed_data'] = 0
    df1['punct_removed_data'] = 0
    df1['url_removed_data'] = 0
    df1['lemma_data'] = 0
    df1['spelling_checked_data'] = 0
    stop_words = set(stopwords.words("english"))

#   lemmatizer = WordNetLemmatizer()
    spell = Speller(lang='en')


  # iterate over each row of dataset and preprocess data
    for i in range(df1.shape[0]):

        # white space removel
        df1['white_space_removed'][i] = re.sub("\s+", " ", df1.text[i])
        emoji_removed = re.sub(r'\W+', ' ', df1['white_space_removed'][i].encode('ascii', 'ignore').decode('utf-8'), flags=re.UNICODE).strip()
        df1['emoji_removed'] = emoji_removed

        # lower casing and tokenization
        lower = df1['emoji_removed'][i].lower()
        tokenized_data = word_tokenize(lower)
        df1['tokenized_data'][i] = tokenized_data
        # print(tokenized_data)


        # remove stopwords
        stopword_removed_data = [x for x in tokenized_data if x not in stop_words]
        df1['stopword_removed_data'][i] = stopword_removed_data 
        # print(stopword_removed_data)


        # remove urls and html tags
        urls = re.findall("https?://[a-zA-Z0-9_\?=\@\/#=.~-]+", " ".join(stopword_removed_data))
        url_removed_data = [x for x in stopword_removed_data if x not in urls]
        df1['url_removed_data'][i] = url_removed_data
        # print(url_removed_data)
        # print(type(url_removed_data))
        
        
        # punctuation removel
        punct_removed_data = [x for x in url_removed_data if x.isalnum()]
        df1['punct_removed_data'][i] = punct_removed_data
        # print(punct_removed_data)
        

        spelling_checked_data = [spell(x) for x in punct_removed_data]
        df1['spelling_checked_data'][i] = " ".join(spelling_checked_data)
#         print(" ".join(spelling_checked_data))

        lemma_data = [lemma.lemmatize(x) for x in spelling_checked_data]
        df1['lemma_data'][i] = lemma_data
        # spelling checking

    return df1


# In[8]:


data.iloc[0]


# In[ ]:


preprocessed_data = preprocessing(data)


# In[ ]:


preprocessed_data


# In[ ]:


df = pd.DataFrame(columns = ['text', 'target'],index = None)
df['text'] = preprocessed_data['spelling_checked_data']
df['target'] = data['label']


# In[ ]:


df.to_csv("cleaned_data_all.csv", index = False)
cleaned_df = pd.read_csv("cleaned_data_all.csv")


# In[ ]:





# In[ ]:




