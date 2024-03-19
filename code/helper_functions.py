import pandas as pd
import statistics

#load different data sets
def read_2020():
    return pd.read_csv('../data/posts_2020.csv')
def read_2019():
    return pd.read_csv('../data/posts_2019.csv')
def read_covid():
    return pd.read_csv('../data/pop_covid.csv')
def read_non_covid():
    return pd.read_csv('../data/pop_non_covid.csv')
def read_reference():
    return pd.read_csv('../data/pop_reference.csv')

#returns the number of comments and the number of empty comments
def total(df):
    print("Total amount of comments:",len(df))
    print("Amount of NaN-comments:",len(df.loc[df["Comment"].isna()==True]))
    
#returns stats about the number of comments and articles
def stats(df,user,article):
    print("Number of unique users:",len(pd.unique(df['ID_CommunityIdentity'])))
    print("Mean comments per user:",len(df)/len(pd.unique(df['ID_CommunityIdentity'])))
    print("With a standard deviation of:", statistics.stdev(user))
    print("Number of unique articles:",len(pd.unique(df['ID_GodotObject'])))
    print("Mean comments per article:", len(df)/len(pd.unique(df['ID_GodotObject'])))
    print("With a standard deviation of:", statistics.stdev(article))
    
#returns stats about the content of the comments (length etc.)
def text_stats(df):
    chars = df['Comment'].str.len()
    words = df['Comment'].str.count(' ').add(1)
    print("Maximum number of characters:", chars.max())
    print("Minimum number of characters:", chars.min())
    print("Mean number of characters:", chars.mean())
    print("Maximum number of words:", words.max())
    print("Minimum number of words:", words.min())
    print("Mean number of words:", words.mean())
    print("Amount of comments with exactly one word:", len(df.loc[df['Comment'].str.count(' ')<1]))

def count_labels(df):
    print(f"There are {sum(df['Label'])} populist comments.")