#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('stopwords')


# In[2]:


from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
import pandas as pd
from nltk.corpus import stopwords
import string

# Download the stopwords resource
nltk.download('stopwords')

# Load stopwords
stopword = set(stopwords.words('english'))

# Your remaining code
data = pd.read_csv("C:/Machine learning project/twitter.csv")
print(data.head())


# In[3]:


data["labels"] = data["class"].map({0: "Hate Speech", 
                                    1: "Offensive Language", 
                                    2: "No Hate and Offensive"})
print(data.head())


# In[4]:


data = data[["tweet", "labels"]]
print(data.head())


# In[5]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)


# In[6]:


x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[13]:


sample = "americans are stupid"
data = cv.transform([sample]).toarray()
print(clf.predict(data))


# In[ ]:





# In[ ]:




