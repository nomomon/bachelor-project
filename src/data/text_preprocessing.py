import re

import nltk
from nltk.corpus import stopwords


def clean_text(text):
    """
        Clean text by removing unnecessary characters (non-aphabetic) and multiplying spaces

        Args:
            text (str): Text to clean
        Returns:
            clean_text (str): Cleaned text
    """

    # Remove non-alphabetic characters
    text = re.sub('[^A-Za-z.?!,]', ' ', text)

    # Remove multiple spaces
    text = re.sub('\s+', ' ', text)

    return text

def clean_token(token):
    """
        Clean token by removing unnecessary characters (non-aphabetic) and multiplying spaces

        Args:
            token (str): Token to clean
        Returns:
            clean_token (str): Cleaned token
    """

    # Remove non-alphabetic characters
    token = re.sub('[^A-Za-z]', '', token)

    return token

def remove_contractions(text):
    """
        Remove contractions from text

        Args:
            text (str): Text to remove contractions from
        Returns:
            text (str): Text without contractions
    """

    # Contractions mapping
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        " u ": " you ",
        " ur ": " your ",
        " n ": " and ",
        "won't": "would not",
        'dis': 'this',
        'bak': 'back',
        'brng': 'bring',
        "'s ": " is ",
        "'r ": " are ",
        "'ve ": " have ",
        "thats": "that is",
        "wont": "will not",
        "cant": "can not",
        "cannot": "can not",
        "youre": "you are",
        "loveu": "love you",
        "im": "i am",
        "ill": "i will",
        "cuz": "because",
        "wanna": "want to",
        "gonna": "going to",
        "gotta": "got to",
    }

    # Replace contractions
    text = text.replace("’", "'").replace("‘", "'").replace("´", "'") \
        .replace("`", "'").replace("“", '"').replace("”", '"') \
        .replace("„", '"').replace("«", '"').replace("»", '"') \
        .replace("–", "-").replace("—", "-").replace("…", "...") \
        .replace(" ", " ").replace("•", "*").replace("·", "*") \
        .replace("‹", "<").replace("›", ">").replace("™", "TM") \
        .replace("©", "(c)").replace("®", "(r)").replace("°", " degrees") \
        .replace("€", " euros").replace("$", " dollars").replace("£", " pounds") \
        .replace("₤", " pounds").replace("×", "x").replace("²", "2")
        

    for contraction in contractions:
        text = text.replace(contraction, contractions[contraction])

    return text


def preprocess_text(text):
    """
        Preprocess text by cleaning, removing stopwords and lemmatizing

        Args:
            text (str): Text to preprocess
        Returns:
            text (str): Preprocessed text
    """

    # Lowercase text
    text = text.lower()

    # Handle contractions
    text = remove_contractions(text)

    # Clean text
    text = clean_text(text)

    return text