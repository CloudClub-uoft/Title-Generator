### Preprocessing, Tokenizing, One-Hot
1. Pull the `preprocess-hugging-face` branch from remote
    * https://github.com/CloudClub-uoft/Title-Generator/tree/preprocess-hugging-face
2. look at `tifu_tokenize.py` under `/preprocess` or `/preprocess-tifu` directory
    * this is the code I used to tokenize and one-hot encode the tifu dataset, which we're probably not using (since we found the hugging face one which is better)
3. Copy most of the code to tokenize and one-hot encode the `reddit_title_text_2011.jsonl.gz` dataset
    * first download [here](https://huggingface.co/datasets/sentence-transformers/reddit-title-body/tree/main)
    * confirm that all words are lowercase first before you tokenize! Might have to do `.lower()`
