import pandas as pd
import xml.etree.ElementTree as ET
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from keras.models import Sequential
from keras.utils import to_categorical
import pickle
from keras.layers import Dense, Activation

token = RegexpTokenizer(r'[a-zA-Z]+')

dict_emotion = {"joy": 0, "fear": 1, "anger": 2, "sadness": 3, "disgust": 4, "shame": 5, "guilt": 6}

df = pd.read_csv("isear.csv", sep='|', error_bad_lines=False)
x_list = df.SIT.tolist()
y_list_emotion = df.Field1.tolist()
for i in range(len(y_list_emotion)):
    y_list_emotion[i] = dict_emotion[y_list_emotion[i]]


#print(emotion)
#print(text)



cv = TfidfVectorizer(encoding='latin-1',lowercase=True,stop_words='english',tokenizer = token.tokenize)
text_counts= cv.fit_transform(x_list)
input_vector_size = len(cv.get_feature_names())
# print(cv.get_feature_names())
# print(input_vector_size)


X_train, X_test, y_train, y_test = train_test_split(text_counts, y_list_emotion , test_size=0.2, random_state=1)

# SVM classifier for Aspect Analysis
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print( 'TF-IDF' + " Accuracy(SVM classifier):",metrics.accuracy_score(y_test, predicted))
filename = 'emotion_model_text.sav'
pickle.dump(clf, open(filename, 'wb'))
