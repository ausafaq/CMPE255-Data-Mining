
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import *


# In[53]:


train_data_df = pd.read_csv('new_train.txt',delimiter='\t',engine='python', encoding='utf-8')


# In[54]:


test_data_df = pd.read_csv('new_test.txt',header = None ,delimiter="\n",encoding='utf-8')


# In[55]:


train_data_df.columns = ["Sentiment","Text"]


# In[56]:


test_data_df.columns = ["Text"]


# In[57]:


stemmer = PorterStemmer()


# In[48]:


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# In[49]:


def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = text.split(" ")
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[50]:


vectorizer = TfidfVectorizer(analyzer='word',tokenizer=tokenize,lowercase=True,stop_words ='english',max_features =1100)


# In[51]:


vectorizer


# In[101]:


total_data = []
total_text = []
total_data = train_data_df.Text.tolist() + test_data_df.Text.tolist()


# In[102]:


for item in total_data:
    item = str(item)
    item.encode('utf-8')
    total_text.append(item)


# In[104]:


vectorizer = TfidfVectorizer(analyzer='word',tokenizer=tokenize,lowercase=True,stop_words ='english',max_features =1100)


# In[105]:


corpus_data_features = vectorizer.fit_transform(total_text)


# In[106]:


corpus_data_features_nd = (corpus_data_features.toarray())


# In[108]:


corpus_data_features_nd.shape


# In[109]:


my_model = LinearSVC(penalty = 'l2',dual = True,C=0.7,loss='hinge')


# In[111]:


my_model = my_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)


# In[112]:


test_pred = my_model.predict(corpus_data_features_nd[len(train_data_df):])


# In[113]:


spl = []
for i in range(len(test_pred)) :
    spl.append(i)
results = []
actual = []


# In[114]:


for text, Sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print(Sentiment)

