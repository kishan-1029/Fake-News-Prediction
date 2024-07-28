import streamlit as st
import pickle
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()


vector_form = pickle.load(open('newvector.pkl', 'rb'))
load_model = joblib.load(open('Decisionmodel.pkl', 'rb'))

def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction



if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class[0] == [0]:
            st.success('The News is Real')
        if prediction_class[0] == [1]:
            st.warning('The News is Fake')