import string

EXTRA_PUNCTUATION = '—«»'

def filter_text(text):
    text = to_lower_case(text)
    text = remove_punctuation(text)
    return text

def to_lower_case(text):
    return text.lower()

def remove_punctuation(text):
    all_punct = string.punctuation + EXTRA_PUNCTUATION
    return text.translate(str.maketrans('', '', all_punct))
