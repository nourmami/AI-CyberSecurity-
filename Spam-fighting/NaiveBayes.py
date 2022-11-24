import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import numpy as np
#import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split 

from defs import get_tokens
from defs import get_lemmas

#read in the data
sms = pandas.read_csv('../datasets/sms_spam_no_header.csv', sep=',', names=["type", "text"])

#seperate the data into training and testing sets
text_train, text_test, type_train, type_test = train_test_split(sms['text'], sms['type'], test_size=0.3)

#convert the text into a matrix of token counts
bow = CountVectorizer(analyzer=get_lemmas).fit(text_train)
print("###########################bow###########################",bow.get_feature_names())
#transform the counts into a matrix of TF-IDF features
sms_bow = bow.transform(text_train)
print("###########################sms_bow###########################",sms_bow)

#train the model
tfidf = TfidfTransformer().fit(sms_bow)
print("###########################tfidf###########################",tfidf)

#transform the counts into a matrix of TF-IDF features
sms_tfidf = tfidf.transform(sms_bow)
print("###########################sms_tfidf###########################",sms_tfidf)

#train the model
spam_detector = MultinomialNB().fit(sms_tfidf, type_train)
msg = sms['text'][25]
msg_bow = bow.transform([msg])
msg_tfidf = tfidf.transform(msg_bow)

print ('predicted:', spam_detector.predict(msg_tfidf)[0])
print ('expected:', sms.type[25])

predictions = spam_detector.predict(sms_tfidf)
print ('accuracy', accuracy_score(sms['type'][:len(predictions)], predictions))
print (classification_report(sms['type'][:len(predictions)], predictions))
