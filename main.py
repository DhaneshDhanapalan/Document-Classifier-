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

"""


with open(
        join(dirname(realpath(__file__)), "D:/Tattvamasi_Software/CODEMANTRA_PROJECT/CODE/CODE_6/xgboost_clf.pkl"), "rb"
    ) as f:
        model = load_model(f)
        
"""
        
        
#model = load_model('D:/Tattvamasi_Software/CODEMANTRA_PROJECT/CODE/CODE_6/model.H5', compile=False)
model = pickle.load(open('D:/Tattvamasi_Software/CODEMANTRA_PROJECT/CODE/CODE_6/xgboost_clf.pkl', 'rb'))

"""

@app.get("/")
def read_root():
    return {"Processing Document classifier !"}
"""
    
"""
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with open("D:/DocumentClassifier/input/input.pdf", "wb") as pdf:
        content = await file.read()
        pdf.write(content)
        pdf.close()
    return JSONResponse(content={"filename": file.filename},status_code=200)
"""

"""
zip_file = "D:/DocumentClassifier/input/INVOICE.zip"
 
try:
    with zipfile.ZipFile(zip_file) as z:
        z.extractall("D:/DocumentClassifier/docs/")
        print("Extracted all")
except:
    print("Invalid file")  
    
"""
"""

with zipfile.ZipFile("D:/DocumentClassifier/input/INVOICE.zip","r") as zip_ref:
    zip_ref.extractall("D:/DocumentClassifier/docs ")
"""
#textfile=open("D:/DocumentClassifier/zip/readme.txt","r")
csvfile = open('D:/DocumentClassifier/pdf_label.csv')

#print(type(csvfile))
data= pd.read_csv(csvfile)
print(data)
#print('cells based on indexing****',data.iloc[0,2])
#print('rows.....',data.shape[0])
#print( 'columns.............',data.shape[1])
"""

for i in data.itertuples():
    tuple1=i
    fileName=tuple1[1]
    print('filename',fileName)
    fileLabel=tuple1[2]
    print('filelabel',fileLabel)
    if not os.path.exists('D:/DocumentClassifier/DOCFOLDERS/'+fileLabel):
        print('directory ',fileLabel,' does not exist')
        os.mkdir('D:/DocumentClassifier/DOCFOLDERS/'+fileLabel)
    print('directory ',fileLabel,' created')
    source ='D:/DocumentClassifier/zip/'+fileName
    print('source...........................',source)
    destination = 'D:/DocumentClassifier/DOCFOLDERS/'+fileLabel+'/'
    print('destination......................',destination)
            
    shutil.move('D:/DocumentClassifier/zip/'+fileName,destination)
    print('file ',fileName,' moved to folder ',fileLabel)
"""

"""

for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.values)
    
    
     
    if(columnName == 'label') :   
        arr2 = columnData.values

for x in arr2:
    print('??????',x)
    if x == 'PO' and  not os.path.exists('D:/DocumentClassifier/DOCFOLDERS/PO'):
        os.mkdir('D:/DocumentClassifier/DOCFOLDERS/PO')
    if x == 'INVOICE' and  not os.path.exists('D:/DocumentClassifier/DOCFOLDERS/INVOICE') :   
   
        os.mkdir('D:/DocumentClassifier/DOCFOLDERS/INVOICE')    
        
    if x == 'SOW' and  not os.path.exists('D:/DocumentClassifier/DOCFOLDERS/SOW') :   
   
        os.mkdir('D:/DocumentClassifier/DOCFOLDERS/SOW') 
        
        #shutil.move()
    

print('?????????..............',columnData.values)
#print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',os.listdir('D:/DocumentClassifier/zip')[3])

"""
"""
path_of_the_directory= 'D:/DocumentClassifier/zip'
print("Files and directories in a specified path:")
for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        print('#####################',f)
        
"""  
"""       

for i in range(len(columnData.values)):
    #print('?????????..............',columnData.values)
    #print('******',i,columnData.values[i])
    source = os.listdir('D:/DocumentClassifier/zip')[i]
    print('^^^^^^^^^^',source)
    #print('%%%%%%%%%%%%',len(columnData.values))
   
    
    if arr2[i] == 'SOW':
        print('sow............',source)
        shutil.move('D:/DocumentClassifier/zip/'+source,'D:/DocumentClassifier/DOCFOLDERS/SOW')
    print('******',i,arr2[i])
    print('^^^^^^^^^^',source)
   
   
        
      
     
    elif arr2[i] == 'PO':  
        print('po.........',source)
        shutil.move('D:/DocumentClassifier/zip/'+source,'D:/DocumentClassifier/DOCFOLDERS/PO')
    elif arr2[i] == 'INVOICE':
        print('invoice.......',source)
        shutil.move('D:/DocumentClassifier/zip/'+source,'D:/DocumentClassifier/DOCFOLDERS/INVOICE')
    
                    
    else:
         print( "message : the document type is not valid")
        
    
"""

"""

@app.post("/training")
def classifier_training():

    dataset_path = 'TRAINING/'
    model = pickle.load(open('D:/DocumentClassifier/pickle/xgboost_clf.pkl', 'rb'))

    data=[]

    for topics in os.listdir(dataset_path):
    
        for pdfs in os.listdir(dataset_path+topics):
    
            if pdfs.endswith('.pdf'):
                #print("----pdf name and text is", dataset_path+topics+'/'+pdfs)
            
                pdfFileObj = open(dataset_path+topics+'/'+pdfs, 'rb')
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)  
                pageObj = pdfReader.getPage(0) 
                text = pageObj.extractText()
                #print("----pdf name and text is", dataset_path+topics+'/'+pdfs,'--------' ,text)
                #print('\n')
            
                data.append([text, topics])
    new_df = pd.DataFrame(data, columns = ['text', 'label'])
    #print(new_df.head(20))
                
    import gensim
    new_df['clean_text'] = new_df['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
    for index, row in new_df.iterrows():
        if len(row['clean_text']) < 5:
            new_df.drop(index, inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    new_df['label'] = le.fit_transform(new_df['label'])
    new_df.head(30)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(new_df['clean_text'],new_df['label'], test_size=0.2,random_state=42)
    tagged_docs_train = [gensim.models.doc2vec.TaggedDocument(word_list, [i]) for i,word_list in enumerate(X_train)]
    tagged_docs_test = [gensim.models.doc2vec.TaggedDocument(word_list, [i]) for i,word_list in enumerate(X_test)]
    d2v_model = gensim.models.Doc2Vec(tagged_docs_train, vector_size=160, window=5, min_count=30)
    train_vectors = [d2v_model.infer_vector(i.words) for i in tagged_docs_train]
    test_vectors = [d2v_model.infer_vector(i.words) for i in tagged_docs_test]        
    from numpy import loadtxt
    cwd = os.getcwd()
    #print('current directory',cwd)
    #loaded_model = pickle.load(open('/modellename/xgboost_clf.pkl', 'rb'))
    #with open('/modellename/xgboost_clf.pkl' , 'rb') as f:
     #   lr = pickle.load(f)

    xgb = XGBClassifier()
    xgb_model = model.fit(np.array(train_vectors), np.array(y_train))
    y_pred = xgb_model.predict(np.array(test_vectors))

    # make predictions for test data
#    y_pred = xgb_model.predict(test_vectors)
    #print(y_pred)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy is", score*100)
    
"""



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
    


