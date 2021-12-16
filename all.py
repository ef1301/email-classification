#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the data into a pandas dataframe called emails
emails=pd.read_csv('input/spam_ham_dataset.csv')

print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))
emails.head()


# I have noticed that these emails are all lowercase; I've looked at some other data sets as well (least the ones that have spam/ham labeles and they also seem to be lowercase)
# 
# I honestly think having caps would be amazing because I'm sure spam emails include a ton more caps - but uh yea

# In[3]:


def get_email_subject(email):
    subject = email[0:email.find('\r\n')]
    subject = subject. replace('Subject: ', '')
    return subject

def get_email_body(email):
    body = email[email.find('\r\n')+2:]
    return body


# In[4]:


# cleaning of columns
email_df = emails.drop(['Unnamed: 0', "label_num"], axis = 1)

# get the subject and body of email
email_df["subject"] = email_df["text"].apply(lambda x: get_email_subject(x))
email_df["body"] = email_df["text"].apply(lambda x: get_email_body(x))

# ridding of the text column (unless we need it)
email_df = email_df.drop(["text"], axis = 1)

email_df

# expand default pandas display options to make emails more clearly visible when printed
pd.set_option('display.max_colwidth', 200)

# from here email_df is our dataframe
email_df.head() # you could do print(bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames


jobs1_df=pd.read_csv("input/jobs-1.csv")

email_df = email_df.append(jobs1_df, ignore_index=True)


# # Text/Data Pre-processing

# In[5]:


# hyperparameters 
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token


# **Tokenization** (Maybe we will have multiple tokenization methods; you can put how you wana tokenize down here)

# In[6]:


# Tokenization method 1
# this is tokenization split by white sapce
def tokenize_1(row):
    if row is None or row is '':
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]
    return tokens


# In[7]:


from nltk.tokenize import word_tokenize, wordpunct_tokenize


# In[8]:


# Tokenization method 2
# split of white space AND punctuation $3.88 --> '3', '.', '88'
def tokenize_2(row):
    return wordpunct_tokenize(str(row))[:maxtokens]


# In[9]:


# Tokenization method 3
def tokenize_3(row):
    return word_tokenize(str(row))[:maxtokens]


# **Regular Expression to remove  unnecessary characters** (removing \n new lines, symbols?, this could also include links)

# In[10]:


import re
import string 

def reg_expressions(row):
    row = re.sub(r'[\r\n]', "", row)
    return row


# **Stop-word removal** (removing unimportant words)
# 

# In[11]:


import nltk
stopwords = nltk.corpus.stopwords.words('english')

def stop_word_removal1(row):
    token = [token for token in row if token not in stopwords]
    return token

def stop_word_removal2(row):
    names=["connie", "deng", "emily", "fang"]
    stopwords.extend(names)
    token = [token for token in row if token not in stopwords]
    return token


# **Stemming** (removing endings of words, -ing, -ly...)

# In[12]:


def stemming(row):
    port_stemmer = nltk.stem.porter.PorterStemmer()
    token = [port_stemmer.stem(token) for token in row]
    return token


# **Lemmatization** (convert into root word)

# In[13]:


def lemmatization(row):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    token = [lem.lemmatize(token) for token in row]
    return token


# **Remove puntuation**

# In[14]:


import string 

def remove_punct(row):
    punctuation = string.punctuation.replace("!", "")
    token = [token for token in row if token not in punctuation]
    return token


# **Final utility in preprocessing data connecting all these preprocessing techniques**

# In[15]:


'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_tokenize=1, flg_stopwords=1, flg_stemm=False, flg_lemm=True, flg_punct=True):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = text.lower()
    text = reg_expressions(text)
    ## Tokenize (convert from string to list)
    if flg_tokenize == 1:
        text = tokenize_1(text)
    elif flg_tokenize == 2:
        text = tokenize_2(text)
    elif flg_tokenize == 3:
        text = tokenize_3(text)
    # remove Stopwords
    if flg_stopwords == 1:
        text = stop_word_removal1(text)
    if flg_stopwords == 2:
        text = stop_word_removal2(text)
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        text = stemming(text)
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        text = lemmatization(text)
    if flg_punct == True:
        text = remove_punct(text)
    ## back to string from list
    text = " ".join(text)
    return text


# In[16]:


email_df["text_clean"] = email_df["body"].apply(lambda x: utils_preprocess_text(x, flg_tokenize=2, flg_stemm=False, flg_lemm=False, flg_stopwords=True, flg_punct=True))
email_df


# # Getting Training and Test Set

# In[17]:


import seaborn as sns
sns.countplot(x="label",data=email_df,order=['spam','ham'])


# In[18]:


email_df["label"].value_counts()


# The ratio between spam and ham is **1499:3672** in the complete dataset. We will maintain this ratio between spam and ham for the training and test dataset.
# 
# We will also split the dataset into a 80%:20% where the training set will be 80% and the test set will be 20%

# In[19]:


from sklearn.model_selection import train_test_split

# random_state 0 makes sure that the data split is consistently the same (so the random sampling does not keep changing)
# train, test = train_test_split(email_df, test_size=0.20, stratify=email_df["label"], random_state=0)

x_train, x_test, y_train, y_test = train_test_split(email_df["text_clean"], email_df["label"], test_size=0.2, stratify=email_df["label"], random_state=0)


# **Training data set**

# In[20]:


sns.countplot(x=y_train,data=x_train, order=["spam", "ham"])


# In[21]:


sns.countplot(x=y_test,data=x_test, order=['spam','ham'])


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

# training
# train_count_wm = countvectorizer.fit_transform(email_train_df["text_clean"])
# train_tfidf_wm = tfidfvectorizer.fit_transform(email_train_df["text_clean"])

# testing
# test_count_wm = countvectorizer.fit_transform(email_train_df["text_clean"])
# test_tfidf_wm = tfidfvectorizer.fit_transform(email_train_df["text_clean"])

# features
# train_count_tokens = countvectorizer.get_feature_names()
# train_tfidf_tokens = tfidfvectorizer.get_feature_names()

# df

# train_df_countvect = pd.DataFrame(data = train_count_wm.toarray(),columns = count_tokens)
# train_df_tfidfvect = pd.DataFrame(data = train_tfidf_wm.toarray(),columns = tfidf_tokens)

# test_df_countvect = pd.DataFrame(data = test_count_wm.toarray(),columns = count_tokens)
# test_df_tfidfvect = pd.DataFrame(data = test_tfidf_wm.toarray(),columns = tfidf_tokens)

x_train_tf = tfidfvectorizer.fit_transform(x_train)
x_test_tf = tfidfvectorizer.transform(x_test)


# In[24]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X = x_train_tf.todense()

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1])
plt.show()             


# In[25]:


from sklearn.naive_bayes import MultinomialNB

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(x_train_tf.toarray(), y_train)


# **Constructing email_train_df**

# In[26]:


x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)

email_train_df = x_train_df.join(y_train_df, lsuffix='_caller', rsuffix='_other')
email_train_df = email_train_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_train_df=email_train_df[["subject","body", "text_clean", "label"]]
email_train_df


# **Constructing email_test_df**

# In[27]:


x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

email_test_df = x_test_df.join(y_test_df, lsuffix='_caller', rsuffix='_other')
email_test_df = email_test_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_test_df=email_test_df[["subject","body", "text_clean", "label"]]
email_test_df


# In[28]:


y_pred = naive_bayes_classifier.predict(x_test_tf.toarray())

email_test_df["prediction"] = y_pred.tolist()
email_test_df


# In[29]:


print(sum(email_test_df["label"] == email_test_df["prediction"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction"])/len(email_test_df))


# # K-Nearest Neighbors

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train_tf, y_train)
y_pred_knn = classifier.predict(x_test_tf)

email_test_df["prediction_knn"] = y_pred_knn.tolist()
email_test_df


# In[45]:


print(sum(email_test_df["label"] == email_test_df["prediction_knn"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_knn"])/len(email_test_df))


# In[51]:


email_test_df.loc[email_test_df["prediction_knn"] != email_test_df["label"] ].head()


# # SVM

# In[52]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train_tf, y_train)
y_pred_svm = svclassifier.predict(x_test_tf)

email_test_df["prediction_svm"] = y_pred_svm.tolist()
email_test_df.head()


# In[53]:


print(sum(email_test_df["label"] == email_test_df["prediction_svm"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_svm"])/len(email_test_df))


# # Decision Tree

# In[55]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train_tf, y_train)
y_pred_dtree = classifier.predict(x_test_tf)

email_test_df["prediction_dtree"] = y_pred_dtree.tolist()
email_test_df.head()


# In[56]:


print(sum(email_test_df["label"] == email_test_df["prediction_dtree"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_dtree"])/len(email_test_df))
