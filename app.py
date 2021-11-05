from flask import Flask, render_template,request
import os
from werkzeug.utils import secure_filename
import numpy
import pickle
import sqlite3
import joblib 
from Logger.logger import Logs
import pandas as pd
app=Flask(__name__)
log = Logs("test_logs.log")
log.addLog("INFO", "Execution started Successfully !")
model = joblib.load('Best_Model_Random_forest_Max_depth_72.pkl')
log.addLog("INFO", "Best_Model_Random_forest_Max_depth_72.pkl loaded Successfully !")
print(model)

with open("test.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)


@app.route('/')
def home():
    log.addLog("INFO", "Rendering template <index.html> !")
    return render_template('index.html')

@app.route('/success', methods = ['POST']) 
def predict():
    if request.method == 'POST': 
                message=request.files["file"]
                message.save(message.filename)
                log.addLog("INFO", "Information collected Successfully !")
                o=output(message.filename)
                print(message.filename)
                print(o[0])
                os.remove(message.filename)
                r=[]
                for j in range(len(o)):
                    r.append('Predicted Class '+str(o[j])+' for Sample '+str(j))
                print(r)
                log.addLog("INFO", "Predicted Successfully !")
                log.addLog("INFO", "Rendering template <index2.html> !")
                return render_template('index2.html' , data=r)
def output(file):
    data=pd.read_csv(file)
    print( data.shape)
    #print(data)
    temp = data.copy()
    #print(temp)
    log.addLog("INFO", "Connected to database successfully !")
    conn = sqlite3.connect('my_data.db')
    c = conn.cursor()
    
    log.addLog("INFO", "Prediction started Successfully !")
    j=1
    for co in data.columns:
        data[co]=b[j].transform(data[co])
        j=j+1
    prediction=model.predict(data)
    t=[]
    for j in prediction:
        if j==0:
            t.append('p')
        else:
            t.append('e')
    temp.columns = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20','col21','col22']
    temp.insert(0,'sample_class',t)
    temp.to_sql(name='mushroom', con=conn, if_exists='append', index=False)
    log.addLog("INFO", "Results appended to database succesfully !")
    #s=c.execute('SELECT * FROM mushroom').fetchall()
    #print(s)
    conn.close()
    return prediction 

if __name__=="__main__":
    app.run(host="localhost", port=8000, debug=True)
