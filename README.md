# Author Identification

This is a machine learning program to perform authorship identification on sample sentences from three horror authors HP Lovecraft (HPL), Mary Wollstonecraft Shelley (MWS), and Edgar Allen Poe (EAP).

<!-- repository overview -->
## Repository Overview

### Mileston Assignments

- [Project Proposal](https://docs.google.com/document/d/1UjZSUKIN5-pZVDwkh30WvPxeJ4GcDklhxFT54oqhlvU/edit?usp=sharing)
- [Project Progress Report](https://docs.google.com/document/d/1nQ4TuBLxJSoydQmPv5aiRY7_rXya0itOSrlbphC7_KU/edit?usp=sharing)
- [Demo Slides](https://docs.google.com/presentation/d/1HsItXIxG8BvZHMXEjezesqXytGEJaSBK0bOph0_HXe0/edit?usp=sharing)

### Scripts
- classify.py: Contains the main functions to run the experiment. Make sure to specify the training and testing files as needed.
- extract_features.py: Contains the functions to extrace lexical and semantic feature vectors from text
- train.py: contains the preprocessing and training functions
- training_data/process_data.py: process a .txt file into the appropriate format in a csv file. Make sure to specify the input .txt file, output .csv file and author signifier.
- training_data/generate_data.py: create a single training dataset from a set of .csv files (for each author)

### Data
- training_data/train_data.csv: Contains dataset of 5 poets, 75 sequences of 512 tokens for each author.
- training_data/test_gpt.csv: Contains dataset of GPT generated sequences for each author
- training_data/test_control.csv: Contains validation dataset of real sequences for each author, not present in the training set.
- *.txt files: raw text data scraped for each author
- *.csv files: processed data for each author

## How to Run
All of the necessary data needed to run and evaluate this project is provided in the repository. It can be evaluated by simply running `python3 classify.py`. One thing to not is that `extract_mean_syllables_per_word_features` adds roughly 2 hours of runtime to the code. For evaluation purposes I will leave it commented out, however in the main function, this feature can be toggled.
To process additional data or test your own generated text from chatGPT, the files `training_data/process_data.py` and `training_data/generate_data.py` can be directly edited and run from the `python3` shell command with no additional commandline arguments. Descriptions of these scripts are above.

<!-- implementation details -->
## Implementation Details

The classifier is based on Naive Bayes, and will feed the training data and predict each of the unknown sentences.

The project includes the training and testing data files. 
The training data is labeled with the author of each sentence, while the test data is not labeled.

The followings are the feature vectors that the program uses for prediction.
1. bag of words (I put all of the training texts into lists labeled with author and create bag of words based on it. Then read each test text and classify it with bag of words)
2. parts of speech (syntax features)
3. lexical features (average number of words per a sentence, sentence length variation, and lexical diversity)
4. punctuation features  (commas, semicolons, and colons per a sentence)



<!-- Prerequisites -->
## Prerequisites

To run the code, make sure that you install all packages that the project is using. The project is using the following packages: 
- [numpy][numpy-url]
- [nltk][nltk-url]
- [sklearn][sklearn-url]

To ensure that you install the packages above, run the following command on your console: 

```python -m pip install --user numpy nltk sklearn```


<!-- Acknowledgements -->
## Acknowledgements
This project is directly based off the code from:

Shogo Akiyama - shogo.a.0504@gmail.com
Project Link: [https://github.com/shogo54/author-identification][project-url]