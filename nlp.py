import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def nlp_ready(string):
    sentence = nltk.word_tokenize(string)
    lemma = nltk.WordNetLemmatizer()
    lem_output = ' '.join([lemma.lemmatize(w) for w in sentence])
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(lem_output)
    wordIndex = len(tokenizer.word_index) + 1
    clean_text = tokenizer.texts_to_sequences(lem_output)
    clean_text = pad_sequences(clean_text, maxlen=30)
    return clean_text
