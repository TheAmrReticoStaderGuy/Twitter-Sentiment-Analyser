import re
import nltk
nltk.download('stopwords') # Download the stopwords dataset
from nltk.corpus import stopwords# Import stopwords after downloading
stop_words = set(stopwords.words('english'))
import string 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert POS tag to WordNet format
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Perform cleaning and preprocessing on the 'text' column
def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Remove newlines
    text = re.sub(r'\n', '', text)
    # Remove alphanumeric words (words containing digits)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove remaining non-alphabetic characters (except spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Normalize repeated characters (e.g.,  "soooo" -> "so")
    text = re.sub(r'(.)\1+', r'\1\1', text)
  
    tokens = nltk.word_tokenize(text)

# Get POS tags
    pos_tags = nltk.pos_tag(tokens)

# Lemmatize with POS
    lemmatized_words = [
    lemmatizer.lemmatize(token, get_wordnet_pos(pos)) 
    for token, pos in pos_tags 
    if token.lower() not in stop_words
    ]
    # Join words back into a single string
    text = ' '.join(lemmatized_words)
    
    return text
