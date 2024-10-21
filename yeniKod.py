import streamlit as st

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tag import pos_tag


st.title("Your sentences are toxic or not!!")

st.image('identify-toxic-people.jpg',width=250)

from joblib import dump, load

cv = load('cv.pkl')
model = load('model.pkl')

lower_n = lambda x: str(x.lower()).replace('\n',' ')

def stop_words(text):
    text = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text  
    
def alphabetic(text):
    text = [word for word in word_tokenize(text) if word.isalpha()]
    text = ' '.join(text)
    return text
    
def stemmer(text):
    text = [SnowballStemmer('english').stem(word) for word in word_tokenize(text)]
    text = ' '.join(text)
    return text
    
text_input = st.text_area("Please enter english words or sentences")

text_input = lower_n(text_input)


if st.button("Predict"):
    vektor = cv.transform([text_input])
    prediction = model.predict(vektor)[0]
    
    st.success(f"Predicted Output : {prediction}")
