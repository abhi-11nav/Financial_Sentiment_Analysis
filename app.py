#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:38:58 2022

@author: abhinav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 23:16:24 2021

@author: abhinav
"""

from flask import Flask, request, render_template
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import xgboost

lemmatizer = WordNetLemmatizer()

app = Flask(__name__, template_folder='template')

model = pickle.load(open('nlp_model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('/index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
        
        financial_news = request.form["news_"]
        
        financial_news = re.sub('[^a-zA-Z]',' ',financial_news)
        
        financial_news = " ".join(x for x in financial_news.split()).lower()
        
        for word in financial_news.split():
            if word not in stopwords.words("english"):
                financial_news = financial_news.replace(word, lemmatizer.lemmatize(word))
            
        
        for words in financial_news.split():
            if words in stopwords.words("english"):
                financial_news = financial_news.replace(words, "")
                
                
        final_news = vectorizer.transform([financial_news])
        
        prediction = model.predict(final_news)
        
        
        if prediction == "neutral":
            return render_template('/index.html',prediction_text = "NEUTRAL TEXT")
        elif prediction == "postive":
            return render_template('/index.html',prediction_text = "POSITIVE TEXT")
        else:
            return render_template('/index.html',prediction_text = "NEGATIVE TEXT")

        
if __name__ == '__main__':
    app.run(debug=True)

        
        
        