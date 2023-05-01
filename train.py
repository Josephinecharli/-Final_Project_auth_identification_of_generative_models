import re
import nltk
import csv

# use the most common word tokenizers
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# write a fn that preprocesses text segments and removes weird characters

def preprocess_text(text):
    pattern = re.compile(r'[a-zA-Z0-9_:;,."?\' ]+')
    clean_text = ''.join(pattern.findall(text))
    return clean_text

def train(file_path, is_supervised=False):
    data = []
    all_text = {}

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            text_id, text, author = row.get('id'), row.get('text'), row.get('author') if is_supervised else None
            # remove special characters
            text = preprocess_text(text)

            # tokenize text
            if is_supervised:
                if author not in all_text.keys():
                    all_text[author] = ''
                sentences = text.split('.')
                all_text[author] += ' '.join(sentences) + ' '

            data.append((text_id, text, author))
    return (data, all_text) if is_supervised else data