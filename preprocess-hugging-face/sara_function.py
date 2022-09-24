from hugging_face_tokenize import *

def split_one_hot_title(title, cutoff):
    '''
    title is a 1x60xm array. m is the number of unique tokens. 
    cutoff is an integer index of title. cutoff < len(title)
    The first n words are kep. The (n+1)th word on are cut off 
    '''

    # better error catching later
    if cutoff >= title.shape[1]:
        print('invalid string indexing')
        return 'didn\'t', 'work'

    #? why the extra layers
    #! i can't explain the second 0 yet
    before_cutoff = title[:][0][0:cutoff] 
    next_word = title[:][0][cutoff]


    return before_cutoff, next_word

# list_of_sentences = build_list_of_sentences(r"C:\Users\ssara\OneDrive\Documents\SCHOOL\programming\CloudAI\Title-Generator\preprocess-hugging-face\reddit_title_text_2011.jsonl.gz")
list_of_sentences = ["I am testing this function. Yes I am."]
tokens = generate_tokens(list_of_sentences)
one_hot_title = build_one_hot_from_tokens(tokens, 60)

[before_cutoff, next_word] = split_one_hot_title(one_hot_title, 0)
print(before_cutoff)
print(next_word)
