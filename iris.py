## Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#########################

#setting the title and text
st.title("🌼Iris Flower Classification")
st.write("*Made with ❤️‍🔥 by Shakeb Shamsi👨🏻‍💻*")


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

print(model)  # Display the loaded data

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

print(encoder)  # Display the loaded data

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

print(scaler)  # Display the loaded data

#taking the input from user
newSL = st.number_input("Enter sepalLength (cm):", min_value=0.0)
newSW = st.number_input("Enter sepalWidth (cm):", min_value=0.0)
newPL = st.number_input("Enter petalLength (cm):", min_value=0.0)
newPW = st.number_input("Enter petalWidth (cm):", min_value=0.0)

#button to trigger the classification
if st.button("Classify"):
    newValue = pd.DataFrame([[newSL, newSW, newPL, newPW]])
    newValue = scaler.transform(newValue)
    prediction = model.predict(newValue)
    finalAns = encoder.inverse_transform(prediction)
    st.markdown(f"Prediction result: **{finalAns[0]}**")


###################################


##Importing DataSets/CSV File
file_path = "iris.csv"
df = pd.read_csv(file_path)


## Data Processing

# 01:Encoding the data variable
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# 02:Splitting the data into features (X) and target (y)
x = df.drop('Species', axis=1)
y = df['Species']


# 03:Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


## Train a Machine Learning Model
rf_classifier =RandomForestClassifier(random_state=42)
clf_classifier =DecisionTreeClassifier
rf_classifier.fit(x_train, y_train)


## Evaluating the model

# Making predictions on the test data
y_pred =rf_classifier.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print("Accuracy", accuracy)
print("Confusion Matrix: \n", conf_matrix)
print("Classification report: \n", class_report)







