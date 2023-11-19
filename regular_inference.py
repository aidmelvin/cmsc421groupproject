from transformers import pipeline

generator = pipeline('text-generation', model='facebook/opt-1.3b')
print(generator('what is the answer/purpose of the universe?'))
