from typing import Optional
import argparse
import pickle
import spacy
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import JSONResponse,FileResponse
import uvicorn
from typing import List
#import fitz
import os
from base64 import b64encode
import os
import shutil
#import fitz
import PyPDF2 
import pandas as pd
import gensim
from natsort import natsort
import uuid
import traceback
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from os.path import dirname, join, realpath
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
import zipfile
import re
from pydantic import BaseModel
from fastapi import Request
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import csv

class file(BaseModel):
    File
        



app = FastAPI()





        

model = pickle.load(open('D:/Tattvamasi_Software/CODEMANTRA_PROJECT/CODE/CODE_6/xgboost_clf.pkl', 'rb'))


@app.post("/document-classify")
async def classifier_predict(file: UploadFile = File(...)):
    with open("D:/DocumentClassifier/input/input.pdf", "wb") as pdf:
        content = await file.read()
        pdf.write(content)
        pdf.close()
    #return JSONResponse(content={"filename": file.filename},status_code=200)
    
    
    
    #output= []
    
    #model  = pickle.load(open('D:/Tattvamasi_Software/CODEMANTRA_PROJECT/CODE/code_2/document_classifier_model_pipeline.pkl', 'rb'))

    dataset_path='input/'
    data=[]

    pdf_name=[]
   
    
    
    for pdfs in os.listdir(dataset_path):

        if pdfs.endswith('.pdf'):

            pdfFileObj = open(dataset_path+'/'+pdfs, 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)  
            pageObj = pdfReader.getPage(0) 
            text = pageObj.extractText()
            data.append([text])
            pdf_name.append(pdfs)
				
    new_df = pd.DataFrame(data, columns = ['text'])
    new_df['clean_text'] = new_df['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
    tagged_docs_test = [gensim.models.doc2vec.TaggedDocument(word_list, [i]) for i,word_list in enumerate(new_df['clean_text'])]
    d2v_model = gensim.models.Doc2Vec(tagged_docs_test, vector_size=100, window=5, min_count=2)
    test_vectors = [d2v_model.infer_vector(i.words) for i in tagged_docs_test]
            

    results = model.predict(np.array(test_vectors))
    results = results.tolist()
    print('type of result.....',type(results))
    #results = 4
    print('results....',results[0])
    
    
        
        
    
                    
    #output.append([results, pdf_name])
    #docType = ""
    if results[0] == 0:
        output = "message : The document is Invoice"
    elif results[0] == 1:  
        output = "message : The document is Purchase Order"
    elif results[0] == 2:
        output = "message :The document is Productorder"
    elif results[0] == 3:
        output = "message : The document is Statement of Work"
    else:
        output = "message : the document type is not valid"
        
        
    #results = "hello"
        
    print('output...',output)    
                
        
  
    return output
    
    
    
    


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    


