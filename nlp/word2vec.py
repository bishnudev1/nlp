import gensim

from gensim.models import Word2Vec


import gensim.downloader as api

# Load the model from the gensim API

model = api.load('word2vec-google-news-300')

# getting 'king' word vector

word_vector = model['king']

# printing the array of the vector 'king'

print(word_vector.toArray())

# printing the shape of the vector 'king'

print(word_vector.shape)

# King and Queen similarity

print(model.similarity('king', 'queen'))

# All similarity of King

print(model.most_similar('king'))

# King - Men + Women = ?

vec = model['king'] - model['men'] + model['women']

# printing the array of the vector 'vec'

print(vec.toArray())

print(model.most_similar([vec], topn=3))