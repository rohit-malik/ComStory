import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import xml.etree.ElementTree as ET
from keras.models import load_model
import pandas as pd

token = RegexpTokenizer(r'[a-zA-Z]+')

dict_emotion = {"joy": 0, "fear": 1, "anger": 2, "sadness": 3, "disgust": 4, "shame": 5, "guilt": 6}

dict_emotion_predict = {0: "joy", 1: "fear", 2: "anger", 3: "sadness", 4: "disgust", 5: "shame", 6: "guilt"}


df = pd.read_csv("isear.csv", sep='|', error_bad_lines=False)
x_list = df.SIT.tolist()
y_list_emotion = df.Field1.tolist()
for i in range(len(y_list_emotion)):
    y_list_emotion[i] = dict_emotion[y_list_emotion[i]]

X_test = ["The food was lousy - too sweet or too salty and the portions tiny."]
for line in open("short_story.txt","r"):
    X_test.append(line)
cv = TfidfVectorizer(encoding='latin-1',lowercase=True,stop_words='english',tokenizer = token.tokenize)
vectorizer = cv.fit(x_list)
X_test_vectorized =  vectorizer.transform(X_test)
input_vector_size = len(vectorizer.get_feature_names())


emotion_model = pickle.load(open('emotion_model_text.sav', 'rb'))
result_emotion = emotion_model.predict(X_test_vectorized)


# sentiment_model = load_model('sentiment_model.h5')
# result_sentiment = sentiment_model.predict(X_test_vectorized)
# result_sentiment_final = []
# for value in result_sentiment:
#     result_sentiment_final.append(list(value).index(max(value)))
i = 0

for result in result_emotion:
    print(X_test[i] + "------" + str(dict_emotion_predict[result]))
    i = i + 1

#print("The emotion in the text is : " + str(dict_emotion_predict[result_emotion[0]]) )
