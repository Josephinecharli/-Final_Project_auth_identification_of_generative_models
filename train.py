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

def train1(file_path, is_supervised=False):
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

def train(filename, supervised=False):
    """
    read the file into data
    :param filename:
    :param supervised:
    :return data:
    """
    p = re.compile(r'[a-zA-Z0-9_:;,."?\' ]+')
    data = []
    all_text = {}
    infile = csv.DictReader(open(filename), delimiter=',', quotechar='"')
    for row in infile:
        text_id = row['id']
        text = row['text']
        author = row['author'] if supervised else None

        # remove special characters
        new_text = ''
        for word in text:
            for letter in word:
                reg = p.match(letter)
                if reg is not None:
                    new_text += reg.group()
        print(new_text)
        data.append((text_id, new_text, author))
        if supervised:
            if author not in all_text.keys():
                all_text[author] = ''
            else:
                sentences = sentence_tokenizer.tokenize(new_text)
                all_text[author] += ' '.join(sentences) + ' '
                # print("{} {} {}".format(text_id, text, author))
    print(all_text.keys())
    print("end train\n")
    if supervised:
        return data, all_text
    else:
        return data