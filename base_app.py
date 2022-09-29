"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# Loading Data
import pandas as pd
import numpy as np
import nltk
import string
import re
import time

# Preprocessing & Model Building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer

# Model Evaulation
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Visualistion
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Display
sns.set(font_scale=1)
sns.set_style("white")
from PIL import Image

# Vectorizer
tweet_vectorizer = open("resources/models/TfidfVectorizer.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) 

# Load your raw data
raw = pd.read_csv("resources/train.csv")
train = raw.copy()

# Data cleaning
# Data cleaning
def clean(tweet_text):
	token = (TweetTokenizer().tokenize(tweet_text)) ## first we tokenize
	punc = [i for i in token if i not in list(string.punctuation)] ## remove punctuations
	dig = [i for i in punc if i not in list(string.digits)] ## remove digits
	final = [i for i in dig if len(i) > 1] ## since we not removing stopwords, remove all words with only 1 character
	return final

# Lemmatisation
def get_part_of_speech(word):
	probable_part_of_speech = wordnet.synsets(word) ## finding word that is most similar (synonyms) for semantic reasoning
	pos_counts = Counter() # instantiating our counter class
	## finding part of speech of word if part of speech is either noun, verb, adjective etc and add it up in a list
	pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
	pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
	pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
	pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
	most_likely_part_of_speech = pos_counts.most_common(1)[0][0] ## will extract the most likely part of speech from the list
	return most_likely_part_of_speech

normalizer = WordNetLemmatizer()
def lemmatise_words(final):
	lemma = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in final] ## lemmatize by way of applying part of speech
	return ' '.join(lemma)

def highlight(val):
	return'background-color: yellow'

labels = {'News': 2,'Pro': 1,'Neutral': 0,'Anti': -1}

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
