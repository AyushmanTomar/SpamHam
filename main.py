import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Spam Detection- Ayushman Tomar")
msg = st.text_input("Paste the message to check for spam")

ps = PorterStemmer()
def transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    x=[]
    for i in text:
        if i.isalnum():
            x.append(i)
    text = x[:]
    x.clear()
    stopwd=stopwords.words('english')
    
    for i in text:
        if i not in stopwd and i not in string.punctuation:
            x.append(i)
    text = x[:]
    x.clear()
    for i in text:
        x.append(ps.stem(i))
    return " ".join(x)
if st.button('Check Spam'):
    transformed_msg = transformer(msg)
    vector_input = cv.transform([transformed_msg])

    result = model.predict(vector_input)[0]
    if result==1:
        st.header("Your meaage is a Spam")
    else:
        st.header("Your message is Not Spam")
