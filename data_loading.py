import pandas as pd

df = pd.read_csv('g_lomidze25_63947.csv')

X = df[['words', 'links', 'capital_words', 'spam_word_count']]
y = df['is_spam']
