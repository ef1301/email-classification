import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import re
import string 
import nltk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Input data files are available in the "../input/" directory.
filepath = "input/spam_ham_dataset.csv"

# Read the data into a pandas dataframe called emails
emails=pd.read_csv('input/spam_ham_dataset.csv')
#print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))

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
print(email_df)

# ----------------------------------------------------------------------------------- PREPROCESSING

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

# Tokenization method 2
# split of white space AND punctuation $3.88 --> '3', '.', '88'
def tokenize_2(row):
    return wordpunct_tokenize(str(row))[:maxtokens]

# Tokenization method 3
def tokenize_3(row):
    return word_tokenize(str(row))[:maxtokens]

def reg_expressions(row):
    row = re.sub(r'[\r\n]', "", row)
    return row

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

# ----------------------------------------------------------------------------------- PREPROCESSING

email_df["text_clean"] = email_df["body"].apply(lambda x: utils_preprocess_text(x, flg_tokenize=2, flg_stemm=False, flg_lemm=False, flg_stopwords=True, flg_punct=True))
email_df

sns.countplot(x="label",data=email_df,order=['spam','ham'])

email_df["label"].value_counts()


# random_state 0 makes sure that the data split is consistently the same (so the random sampling does not keep changing)
# train, test = train_test_split(email_df, test_size=0.20, stratify=email_df["label"], random_state=0)

x_train, x_test, y_train, y_test = train_test_split(email_df["text_clean"], email_df["label"], test_size=0.2, stratify=email_df["label"], random_state=0)

#sns.countplot(x=y_train,data=x_train, order=["spam", "ham"])
#sns.countplot(x=y_test,data=x_test, order=['spam','ham'])


countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

x_train_tf = tfidfvectorizer.fit_transform(x_train)
x_test_tf = tfidfvectorizer.transform(x_test)


classifier = LogisticRegression(multi_class='multinomial', solver = "newton-cg", class_weight = 'balanced')
classifier.fit(x_train_tf.toarray(), y_train)

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)

email_train_df = x_train_df.join(y_train_df, lsuffix='_caller', rsuffix='_other')
email_train_df = email_train_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_train_df=email_train_df[["subject","body", "text_clean", "label"]]

x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

email_test_df = x_test_df.join(y_test_df, lsuffix='_caller', rsuffix='_other')
email_test_df = email_test_df.join(email_df[["subject","body"]], lsuffix='_caller', rsuffix='_other', how='left')
email_test_df = email_test_df[["subject","body", "text_clean", "label"]]

y_pred = classifier.predict(x_test_tf.toarray())

email_test_df["prediction"] = y_pred.tolist()

print(sum(email_test_df["label"] == email_test_df["prediction"]))
print(len(email_test_df))
print("accuracy:", sum(email_test_df["label"] == email_test_df["prediction"])/len(email_test_df))
