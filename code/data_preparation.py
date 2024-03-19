import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem
from sklearn.feature_extraction.text import CountVectorizer
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


#extract only comments that contain dictionary terms
def extract_pop(df):
    df = df.loc[df['Score']>0]
    return df

#get a weighted value for every comment that weighs populist terms against the text length (word count)
def weighted_pop_words(df):
    df['Weight Words'] = df['Score']/df['n_tokens']
    return df
#weigh populist words per sentence
def weighted_pop_sentences(df):
    df['Weight Sentences'] = df['Score']/df['n_sentences']
    return df

#reproduce the preprocessing of the paper of Hawkins and Castanho Silva (2016)
def lowercase(comment):
    return comment.str.lower()

def remove_punct(comment):
    return comment.str.replace('[^\w\s]',' ', regex = True).replace('_',' ',regex =True)

def remove_num(comment):
    return comment.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

def stop_word_removal(comment):
    stop_words = stopwords.words('german')
    words = comment.split()
    return  ' '.join([x for x in words if not x in stop_words])

def remove_stop(comment):
    return comment.apply(stop_word_removal)
#remove URLs
pat1 = r'http[s]?://(?:[a-z]|[0-9]|[$-_#@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
pat2 = r'www.[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
split_pattern = re.compile(r'\b('  + r')\b')
def rem_url(comment):
    soup = BeautifulSoup(comment, 'html.parser') # HTML
    souped = soup.get_text()
    stripped = re.sub(combined_pat, ' ', souped)
    return stripped

def remove_url(comment):
    comment = pd.DataFrame({'Comment':[rem_url(x) for x in comment]})
    return comment['Comment']

def stem(comment):
    stemmer = Cistem()
    words = comment.str.split()
    words = words.apply(lambda x: ' '.join([stemmer.stem(i) for i in x]))
    return words

def preprocess_hawkins(comment):
    return stem(remove_stop(remove_num(remove_punct(lowercase(remove_url(comment))))))

def doc_term_matrix(comment):
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(comment)
    matrix = pd.DataFrame(matrix.toarray(), columns = vectorizer.get_feature_names_out())
    return matrix

#draw a stratified training, validation and test sample (ratio: 2/3, 1/6, 1/6)
def strat_train_val_test(df):
    id_posting = df['ID_Posting']
    X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['Label'], test_size=1/3, random_state=1337, stratify =df['Label'])
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1337, stratify = y_test)

    train = pd.concat([X_train, id_posting.iloc[X_train.index], y_train], axis=1)
    val = pd.concat([X_val, id_posting.iloc[X_val.index], y_val], axis=1)
    test = pd.concat([X_test, id_posting.iloc[X_test.index], y_test], axis=1)

    train.columns = ['Comment', 'ID_Posting', 'Label']
    val.columns = ['Comment', 'ID_Posting', 'Label']
    test.columns = ['Comment', 'ID_Posting', 'Label']
    train = train[['ID_Posting', 'Comment', 'Label']]
    val = val[['ID_Posting', 'Comment', 'Label']]
    test = test[['ID_Posting', 'Comment', 'Label']]
    print(len(train))
    print(len(test))
    return train, test, val

