from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from SPP.settings import BASE_DIR


def index(request):

    context = {"result":None}
    return render(request, 'index.html', context)


def runModel(request):
    
 
    if request.method=="POST":

        tenth_score = request.POST["10thMarks"]
        twlve_score = request.POST["12thMarks"]
        GradMarks = request.POST["GradMarks"]
        pgMarks = request.POST["pgMarks"]
        noBacklogs = request.POST["noBacklogs"]
        quants = request.POST["quants"]
        logical = request.POST["logical"]
        verbal = request.POST["verbal"]
        communication = request.POST["communication"]
        gd = request.POST["gd"]

        inputs =  [tenth_score,twlve_score,GradMarks,pgMarks,noBacklogs,quants,logical,verbal,communication,gd]
        answer = predict(inputs)
        context = {"tenscore":tenth_score,"twlscore":twlve_score,"grdscore":GradMarks,"pgScore":pgMarks,
                    "bklog":noBacklogs,"qnt":quants,"lgcl":logical,"vrbl":verbal,"cmnc":communication,"gd":gd,"result":answer}
        
        return render(request, 'index.html', context)


def predict(inputs):
    conf = train_model()
    
    sc = conf["StdSc"]
    classifier = conf["model"]

    inpts = np.array(inputs,dtype="int").reshape(1,10)
    out = classifier.predict(sc.transform(inpts))
    
    return out[0]



def train_model():
    
    file_path = os.path.join(BASE_DIR, "placement_analysis\\assets\Placement_Data - Original.csv")
    dataset = pd.read_csv(file_path)
    dataset['10th_Score'] = dataset['10th_Score'].fillna(dataset['10th_Score'].mean())

    X = dataset.iloc[:, 1:-1].values    
    y = dataset.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return {"StdSc":sc,"model":classifier}
