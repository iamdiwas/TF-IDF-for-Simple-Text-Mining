# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:17:43 2019

@author: e67967
"""

import nltk
import pandas as pd
import math
import numpy as np
#### Reda the input text files ###
## Note**:- Here I have given my path for reading the files from my local system ##
dec_texts = {
             "text_1": open('W:/Diwas/Python_projects/NLP_Codes/tf_idf_data/text1.txt', "rU").read(),
             "text_2": open('W:/Diwas/Python_projects/NLP_Codes/tf_idf_data/text2.txt', "rU").read(),
             }

from nltk.tokenize import word_tokenize
## term-Frequency calculator ##
def term_frequency(doc,text):


    word_tokens = word_tokenize(doc[text])
    words_freq = nltk.FreqDist(word_tokens)
    
    return words_freq  
## inverse-document-frequency calculator ##
def inverse_document_frquency(doc,word):
    
    word_cnt = [word in doc[file] for file in doc]
    print(word_cnt)
    idf = math.log(len(word_cnt) / sum(word_cnt))
    
    return idf
## tf-idf calculator ##
def tf_idf(doc,text):
    
    tfidf_scores = {}
    tf = term_frequency(doc,text)
    for term in tf:
        
        if term.isalpha(): ## pre-process to remove anything other than text ##
            
            idf = inverse_document_frquency(doc,term)
            tf_term = term_frequency(doc,text)[term]
            tf_idf = idf*tf_term
            tfidf_scores[term] = round(tf_idf,3)
            
    return tfidf_scores    
           
##################################################### EOL ######################################################    