import numpy as np
import re
import json
import gzip

def build_list_of_sentences(file):
    """ 
    opens the file, returns the valid lines in a list of dictionaries 
    """
    with gzip.open(file, "rb") as gzip_file:
        # example run with the first few entries:
        lines = gzip_file.readlines()[:10] 
        paragraphs = []
        for line in lines: 
            line = json.loads(line) 
            paragraphs.append(line["body"]) 
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
    
list_of_sentences = build_list_of_sentences(r"C:\Users\ssara\OneDrive\Documents\SCHOOL\programming\CloudAI\Title-Generator\preprocess-hugging-face\reddit_title_text_2011.jsonl.gz")
tokens = generate_tokens(list_of_sentences)
one_hot = build_one_hot_from_tokens(tokens, 60)

print(list_of_sentences)
print(one_hot)
print(one_hot.shape) # (4x60x51)