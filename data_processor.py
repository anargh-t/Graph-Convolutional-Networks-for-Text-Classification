"""
TextGCN Data Preprocessing Script

This script handles text preprocessing and cleaning for TextGCN datasets.
It performs various text normalization tasks including:
- Character filtering and normalization
- Stop word removal
- Number replacement
- URL replacement
- Low-frequency word filtering

The preprocessing pipeline:
1. Load raw text corpus
2. Apply text cleaning and normalization
3. Remove stop words (except for MR dataset)
4. Filter out low-frequency words
5. Save cleaned corpus for graph construction

Key classes:
- StringProcess: Handles individual text cleaning operations
- CorpusProcess: Orchestrates the complete preprocessing pipeline
"""

import os
import re
from collections import Counter
from collections import defaultdict
import numpy as np

from tqdm import tqdm


class StringProcess(object):
    """
    Text preprocessing utility class for cleaning and normalizing text.
    
    This class provides various methods for text cleaning including:
    - Character filtering and normalization
    - Contraction handling
    - Punctuation spacing
    - Stop word removal
    - Number and URL replacement
    """
    
    def __init__(self):
        """
        Initialize regex patterns for text processing.
        """
        # Regex pattern to match non-alphanumeric characters (except basic punctuation)
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        # Regex pattern to match numbers (including decimals and signs)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # Regex pattern to match URLs
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        
        # Initialize lazy-loaded resources
        self.stop_words = None  # Will be loaded when needed
        self.nlp = None         # Will be loaded when needed

    def clean_str(self, string):
        """
        Clean and normalize a text string.
        
        This method performs comprehensive text cleaning:
        1. Remove non-alphanumeric characters (except basic punctuation)
        2. Add spaces around contractions
        3. Add spaces around punctuation marks
        4. Normalize whitespace
        5. Convert to lowercase
        
        Args:
            string (str): Input text string
            
        Returns:
            str: Cleaned and normalized text string
        """
        # Remove non-alphanumeric characters, replace with spaces
        string = re.sub(self.other_char, " ", string)
        
        # Add spaces around contractions to separate them
        string = re.sub(r"\'s", " \'s", string)      # it's -> it 's
        string = re.sub(r"\'ve", " \'ve", string)    # I've -> I 've
        string = re.sub(r"n\'t", " n\'t", string)    # don't -> do n't
        string = re.sub(r"\'re", " \'re", string)    # you're -> you 're
        string = re.sub(r"\'d", " \'d", string)       # I'd -> I 'd
        string = re.sub(r"\'ll", " \'ll", string)    # I'll -> I 'll
        
        # Add spaces around punctuation for better tokenization
        string = re.sub(r",", " , ", string)         # comma spacing
        string = re.sub(r"!", " ! ", string)         # exclamation spacing
        string = re.sub(r"\(", " \( ", string)       # opening parenthesis spacing
        string = re.sub(r"\)", " \) ", string)       # closing parenthesis spacing
        string = re.sub(r"\?", " \? ", string)       # question mark spacing
        
        # Normalize multiple whitespace characters to single spaces
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()  # Strip whitespace and convert to lowercase

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result


def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])


class CorpusProcess:
    def __init__(self, dataset, encoding=None):
        corpus_path = "data/text_dataset/corpus"
        clean_corpus_path = "data/text_dataset/clean_corpus"
        if not os.path.exists(clean_corpus_path):
            os.makedirs(clean_corpus_path)

        self.dataset = dataset
        self.corpus_name = f"{corpus_path}/{dataset}.txt"
        self.save_name = f"{clean_corpus_path}/{dataset}.txt"
        self.context_dct = defaultdict(dict)

        self.encoding = encoding
        self.clean_text()

    def clean_text(self):
        sp = StringProcess()
        word_lst = list()
        with open(self.corpus_name, mode="rb", encoding=self.encoding) as fin:
            for indx, item in tqdm(enumerate(fin), desc="clean the text"):
                data = item.strip().decode('latin1')
                data = sp.clean_str(data)
                if self.dataset not in {"mr"}:
                    data = sp.remove_stopword(data)
                word_lst.extend(data.split())

        word_st = set()
        if self.dataset not in {"mr"}:
            for word, value in Counter(word_lst).items():
                if value < 5:
                    continue
                word_st.add(word)
        else:
            word_st = set(word_lst)

        doc_len_lst = list()
        with open(self.save_name, mode='w') as fout:
            with open(self.corpus_name, mode="rb", encoding=self.encoding) as fin:
                for line in tqdm(fin):
                    lines_str = line.strip().decode('latin1')
                    lines_str = sp.clean_str(lines_str)
                    if self.dataset not in {"mr"}:
                        lines_str = sp.remove_stopword(lines_str)
                        lines_str = remove_less_word(lines_str, word_st)

                    fout.write(lines_str)
                    fout.write(" \n")

                    doc_len_lst.append(len(lines_str.split()))

        print("Average length:", np.mean(doc_len_lst))
        print("doc count:", len(doc_len_lst))
        print("Total number of words:", len(word_st))


def main():
    # CorpusProcess("R52")
    CorpusProcess("20ng")
    # CorpusProcess("mr")
    # CorpusProcess("ohsumed")
    # CorpusProcess("R8")
    # pass


if __name__ == '__main__':
    main()
