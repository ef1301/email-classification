import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the data into a pandas dataframe called emails
emails=pd.read_csv('input/spam_ham_dataset.csv')

print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))
emails.head()


# I have noticed that these emails are all lowercase; I've looked at some other data sets as well (least the ones that have spam/ham labeles and they also seem to be lowercase)
# 
# I honestly think having caps would be amazing because I'm sure spam emails include a ton more caps - but uh yea

def get_email_subject(email):
    subject = email[0:email.find('\r\n')]
    subject = subject. replace('Subject: ', '')
    return subject

def get_email_body(email):
    body = email[email.find('\r\n')+2:]
    return body

# cleaning of columns
email_df = emails.drop(['Unnamed: 0', "label_num"], axis = 1)

# get the subject and body of email
email_df["subject"] = email_df["text"].apply(lambda x: get_email_subject(x))
email_df["body"] = email_df["text"].apply(lambda x: get_email_body(x))

# ridding of the text column (unless we need it)
email_df = email_df.drop(["text"], axis = 1)

# expand default pandas display options to make emails more clearly visible when printed
pd.set_option('display.max_colwidth', 200)

# from here email_df is our dataframe
email_df.head() # you could do print(bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames


jobs1_df=pd.read_csv("input/jobs-1.csv")

email_df = email_df.append(jobs1_df, ignore_index=True)

# hyperparameters 
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token

# Tokenization method 1
# this is tokenization split by white sapce
def tokenize_1(row):
    if row is None or row == '':
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]
    return tokens

from nltk.tokenize import word_tokenize, wordpunct_tokenize

# Tokenization method 2
# split of white space AND punctuation $3.88 --> '3', '.', '88'
def tokenize_2(row):
    return wordpunct_tokenize(str(row))[:maxtokens]

# Tokenization method 3
def tokenize_3(row):
    return word_tokenize(str(row))[:maxtokens]

import re
import string 

def reg_expressions(row):
    row = re.sub(r'[\r\n]', "", row)
    return row

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

def stemming(row):
    port_stemmer = nltk.stem.porter.PorterStemmer()
    token = [port_stemmer.stem(token) for token in row]
    return token

def lemmatization(row):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    token = [lem.lemmatize(token) for token in row]
    return token

import string 

def remove_punct(row):
    punctuation = string.punctuation.replace("!", "")
    token = [token for token in row if token not in punctuation]
    return token

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

email_df["text_clean"] = email_df["body"].apply(lambda x: utils_preprocess_text(x, flg_tokenize=2, flg_stemm=False, flg_lemm=False, flg_stopwords=True, flg_punct=True))

import seaborn as sns
sns.countplot(x="label",data=email_df,order=['spam','ham'])

email_df["label"].value_counts()

from sklearn.model_selection import train_test_split

# random_state 0 makes sure that the data split is consistently the same (so the random sampling does not keep changing)
# train, test = train_test_split(email_df, test_size=0.20, stratify=email_df["label"], random_state=0)

x_train, x_test, y_train, y_test = train_test_split(email_df["text_clean"], email_df["label"], test_size=0.2, stratify=email_df["label"], random_state=0)

sns.countplot(x=y_train,data=x_train, order=["spam", "ham"])

sns.countplot(x=y_test,data=x_test, order=['spam','ham'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X = x_train_tf.todense()

# pca = PCA(n_components=2).fit(X)
# data2D = pca.transform(X)
# plt.scatter(data2D[:,0], data2D[:,1])
# plt.show()             

from sklearn.naive_bayes import MultinomialNB

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(x_train_tf.toarray(), y_train)

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)

email_train_df = x_train_df.join(y_train_df, lsuffix='_caller', rsuffix='_other')
email_train_df = email_train_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_train_df=email_train_df[["subject","body", "text_clean", "label"]]
email_train_df

x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

email_test_df = x_test_df.join(y_test_df, lsuffix='_caller', rsuffix='_other')
email_test_df = email_test_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_test_df=email_test_df[["subject","body", "text_clean", "label"]]
email_test_df

y_pred = naive_bayes_classifier.predict(x_test_tf.toarray())

email_test_df["prediction"] = y_pred.tolist()
email_test_df

print(sum(email_test_df["label"] == email_test_df["prediction"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction"])/len(email_test_df))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train_tf, y_train)
y_pred_knn = classifier.predict(x_test_tf)

email_test_df["prediction_knn"] = y_pred_knn.tolist()
email_test_df

print(sum(email_test_df["label"] == email_test_df["prediction_knn"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_knn"])/len(email_test_df))

email_test_df.loc[email_test_df["prediction_knn"] != email_test_df["label"] ].head()

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', random_state=0)
svclassifier.fit(x_train_tf, y_train)
y_pred_svm = svclassifier.predict(x_test_tf)

email_test_df["prediction_svm"] = y_pred_svm.tolist()
email_test_df.head()

print(sum(email_test_df["label"] == email_test_df["prediction_svm"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_svm"])/len(email_test_df))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train_tf, y_train)
y_pred_dtree = classifier.predict(x_test_tf)

email_test_df["prediction_dtree"] = y_pred_dtree.tolist()
email_test_df.head()

print(sum(email_test_df["label"] == email_test_df["prediction_dtree"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_dtree"])/len(email_test_df))



# LogisticRegression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class='multinomial', solver = "newton-cg", class_weight = 'balanced')
classifier.fit(x_train_tf.toarray(), y_train)
y_pred_lr = classifier.predict(x_test_tf)

email_test_df["prediction_lr"] = y_pred_lr.tolist()
email_test_df.head()

print(sum(email_test_df["label"] == email_test_df["prediction_lr"]))
print(len(email_test_df))

print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction_lr"])/len(email_test_df))




print("\n\n\n--------------------------------------------\n\n\n")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


for x in ["prediction", "prediction_knn", "prediction_svm", "prediction_dtree", "prediction_lr"]:
    print(x)
    print("confusion matrix")
    print(confusion_matrix(email_test_df["label"], email_test_df[x], labels=["spam", "ham", "jobs"]))
    print("precision score")
    print(precision_score(email_test_df["label"], email_test_df[x], labels=["spam", "ham", "jobs"], average=None))
    print("recall score")
    print(recall_score(email_test_df["label"], email_test_df[x], labels=["spam", "ham", "jobs"], average=None))
    print("f1 score")
    print(f1_score(email_test_df["label"], email_test_df[x], labels=["spam", "ham", "jobs"], average=None))
    print("accuracy score")
    print(accuracy_score(email_test_df["label"], email_test_df[x]))
    print()
