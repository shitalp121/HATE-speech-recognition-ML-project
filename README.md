Hate Speech Recognition on Twitter Data Using ML
Overview
This project classifies tweets as Hate Speech, Offensive Language, or Non-Offensive using Machine Learning. A Decision Tree Classifier is trained on a dataset to predict whether a given tweet contains offensive language.

Dataset
The project uses a Twitter dataset containing labeled tweets with three categories:

0: Hate Speech
1: Offensive Language
2: No Hate or Offensive
Features
Text Cleaning: Removing URLs, punctuation, and stopwords
Stemming: Using NLTK SnowballStemmer to normalize words
Vectorization: CountVectorizer for text feature extraction
Model: Decision Tree Classifier
Installation

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Download necessary NLTK resources:
python
Copy
Edit
import nltk
nltk.download('stopwords')
Usage
Run the script to train the model and test predictions:

bash
Copy
Edit
python hate_speech_detection.py
Example Prediction:

python
Copy
Edit
sample = "americans are stupid"
data = cv.transform([sample]).toarray()
print(clf.predict(data))
Expected Output:

css
Copy
Edit
['Offensive Language']
Dependencies
Python
Pandas
Scikit-learn
NLTK
Future Improvements
Implement deep learning models (LSTMs, Transformers)
Use advanced NLP techniques for better accuracy
Deploy as a web API
