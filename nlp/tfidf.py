import nltk

paragraph = """
There are many problems that may require general intelligence to solve the problems as well as humans do. For example, even specific straightforward tasks, like machine translation, require that a machine read and write in both languages (NLP), follow the author's argument (reason), know what is being talked about (knowledge), and faithfully reproduce the author's original intent (social intelligence). All of these problems need to be solved simultaneously in order to reach human-level machine performance. It is possible that some of these problems will be solved by using narrow AI methods in isolation, but it is more likely that solving one problem will require systems that can integrate approaches from all these fields. Some simple examples of narrow AI include image recognition, search engines, and recommendation algorithms.
"""

# Tokenization the raw data

nltk.download('punkt')

sentences = nltk.sent_tokenize(paragraph)

# print("sentences: ", sentences)

nltk.download('stopwords')
# Stemming with Stopwords
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

import re

corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    
    review = [
        stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))
    ]
    review = ' '.join(review)
    
    corpus.append(review)

# Converting Bag of Words
            
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(binary=True,ngram_range=(2, 2))

X = tfidf.fit_transform(corpus).toarray()

print(X[0])