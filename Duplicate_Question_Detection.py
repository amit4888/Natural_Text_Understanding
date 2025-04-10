#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Library

# In[ ]:


get_ipython().system('pip install xgboost lightgbm catboost')


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# # 1. Data Acquisition

# In[3]:


df = pd.read_csv("duplicate_questions_pairs.xls")


# In[4]:


# Display First 5 rows of the dataset
df.head()


# In[5]:


# Shape of the dataset
df.shape


# In[6]:


# Check Dtype and Non-Null values
df.info()


# In[7]:


# Check Missing Values
df.isnull().sum()


# In[8]:


# Check Duplicated Values
df.duplicated().sum()


# In[9]:


# Distribution of duplicate and non-duplicate questions

print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')


# In[10]:


# Combine both question IDs into a single list
all_qids = df['qid1'].tolist() + df['qid2'].tolist()

# Convert list into a pandas Series
qid_series = pd.Series(all_qids)

# Total unique questions
unique_questions = qid_series.nunique()
print("Total unique questions:", unique_questions)

#  Repeated Question
repeated_questions = (qid_series.value_counts() > 1).sum()
print("Number of questions getting repeated:", repeated_questions)


# In[11]:


# Drop rows with missing questions
df.dropna(subset=['question1', 'question2'], inplace=True)


# In[12]:


# Now, again I am checking Missing Values
df.isnull().sum()


# # 2.Text Preprocessing

# # Basic Preprocessing

# In[13]:


import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
import string


# In[14]:


print(df['question1'][0])
print(df['question2'][0])


# In[15]:


# Apply lowercase transformation to the 'Job Description' column

def clean_text(text):
    text = text.lower()    
    return text


# In[16]:


# Apply cleaning to both questions
df['question1'] = df['question1'].apply(clean_text)
df['question2'] = df['question2'].apply(clean_text)


# In[17]:


print(df['question1'][0])
print(df['question2'][0])


# # Remove Punctuation

# In[18]:


string.punctuation
punctuation = string.punctuation
exclude = punctuation


# In[19]:


# Create a function 
def remove_punc(text):
    for char in exclude:
        text = text.replace(char,'')
    return text


# In[20]:


# Apply remove_punc to both questions
df['question1'] = df['question1'].apply(remove_punc)
df['question2'] = df['question2'].apply(remove_punc)


# In[21]:


# Here, we can see our punctuation has removed
df['question1'][0]


# # Stopwords Removing

# In[22]:


english_stopwods = stopwords.words('english')
print(english_stopwods)


# In[23]:


def remove_stopwords(text):
    new_text = []
    
    for word in text.split():
        if word not in stopwords.words('english'):
            new_text.append(word)
            
    return " ".join(new_text)


# In[24]:


df['question1'] = df['question1'].apply(remove_stopwords)
df['question2'] = df['question2'].apply(remove_stopwords)


# # Word Tokenizations

# In[25]:


def tokenization(text):
    tokens = word_tokenize(text)
    return tokens


# In[26]:


df['question1'] = df['question1'].apply(tokenization)
df['question2'] = df['question2'].apply(tokenization)


# In[27]:


print(df['question1'][4222])
print(df['question2'][123])


# In[28]:


# X = Input Features
X = df['question1'].astype(str) + " " + df['question2'].astype(str)

# Target variable
y = df['is_duplicate']


# # Text Vectorization

# # Bag of word

# In[29]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size = 0.33,random_state = 42)


# # Build the models

# # Logistic Regression

# In[32]:


lr = LogisticRegression()
lr


# In[33]:


# Train the model
lr.fit(X_train, y_train)
# Predict the model
lr_preds = lr.predict(X_test)
# Check the accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))


# # XGBClassifier 

# In[34]:


xgb = XGBClassifier()
xgb


# In[35]:


# Train the model
xgb.fit(X_train,y_train)
# Predict the model
y_pred = xgb.predict(X_test)
# Check Accuracy
print("XGB Classifier Accuracy",accuracy_score(y_test,y_pred))


# # TF-IDF

# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[37]:


X = X = df['question1'].astype(str) + " " + df['question2'].astype(str)

# Step 3: TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)


# In[38]:


lr = LogisticRegression()
lr


# In[39]:


# Train the model
lr.fit(X_train, y_train)
# Predict the model
lr_preds = lr.predict(X_test)
# Check the accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))


# # XGB Classifier

# In[40]:


xgb = XGBClassifier()
xgb


# In[41]:


# Train the model
xgb.fit(X_train,y_train)
# Predict the model
y_pred = xgb.predict(X_test)
# Check Accuracy
print("XGB Classifier Accuracy",accuracy_score(y_test,y_pred))


# In[45]:


q1 = input("Enter the first question: ")
q2 = input("Enter the second question: ")
combined = [q1 + " " + q2]
combined_tfidf = tfidf.transform(combined)
prediction = model.predict(combined_tfidf)

if prediction[0] == 1:
    print("Duplicate")
else:
    print("Not Duplicate")


# In[ ]:




