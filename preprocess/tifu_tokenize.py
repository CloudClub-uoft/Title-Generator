import numpy as np
import re
import json

def build_list_of_sentences(file):
    with open(file) as f:
        # example run with the first few entries:
        lines = f.readlines()[:10]
        paragraphs = []
        for line in lines:
            line = json.loads(line)
            if line["tldr"] is not None:
                paragraphs.append(line["tldr"])
        return paragraphs

def tokenizer(words: str):
    '''
    words is a string
    '''
    return re.findall(r"[\w']+|[.,!?:;]", words)
def generate_tokens(list_of_words):
    '''
    list_of_words is a list of strings
    '''
    tokens = [tokenizer(words) for words in list_of_words]
    return tokens

def build_one_hot_from_tokens(tokens, max_length):
    '''
    tokens should be a list of list of words
    max_length is maximum length of all passages, for our project this will be 60
    '''
    # build an index of all tokens in the data
    token_index = {}
    i = 1
    for sample in tokens:

        for word in sample:
            if word not in token_index:
                # Assign a unique index to each unique word
                token_index[word] = i
                i += 1

    # vectorize our tokens
    results = np.zeros((len(tokens), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(tokens):
        for j, word in enumerate(sample[:max_length]):
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results
    
list_of_sentences = build_list_of_sentences("tifu_all_tokenized_and_filtered.json")
tokens = generate_tokens(list_of_sentences)
one_hot = build_one_hot_from_tokens(tokens, 60)

print(one_hot)
print(one_hot.shape) # (4x60x51)