import chardet
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
import pickle

with open("spam 2.csv", "rb") as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(result)  # Displays the detected encoding

msg = pd.read_csv("spam 2.csv", encoding=result['encoding'])

msg.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace= True)
msg.rename(columns={'v1' : 'target', 'v2' : 'text'}, inplace= True)

encoder = LabelEncoder()
msg['target'] = encoder.fit_transform(msg['target'])
msg = msg.drop_duplicates(keep='first')

ps = PorterStemmer()

print(msg.duplicated().sum())
print(msg['target'].value_counts())

msg['characters'] = msg['text'].apply(len)
msg['words'] = msg['text'].apply(lambda x:len(nltk.word_tokenize(x)))
msg['sentences'] = msg['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

def preprocess_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

msg['preprocessed'] = msg['text'].apply(preprocess_data)

wc = WordCloud(width= 500, height= 500, min_font_size= 10, background_color= 'white')
spam_wc = wc.generate(msg[msg['target'] == 1]['preprocessed'].str.cat(sep = " "))
plt.imshow(spam_wc)


ham_wc = wc.generate(msg[msg['target'] == 0]['preprocessed'].str.cat(sep = " "))
plt.imshow(ham_wc)

spam_words = []
ham_words = []

for message in msg[msg['target'] == 1]['preprocessed'].tolist():
    for word in message.split():
        spam_words.append(word)

for message in msg[msg['target'] == 0]['preprocessed'].tolist():
    for word in message.split():
        ham_words.append(word)

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

x = tfidf.fit_transform(msg['preprocessed']).toarray()
y = msg['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)

mnb = MultinomialNB()

mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

pickle.dump(tfidf, open('../../../spam-sms-detector/vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('../../../spam-sms-detector/model.pkl', 'wb'))
