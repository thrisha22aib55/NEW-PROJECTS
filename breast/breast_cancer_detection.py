# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and calculate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Breast Cancer Detection")
st.write("This app uses a Random Forest Classifier to detect breast cancer.")

st.sidebar.header("User Input Features")
def user_input_features():
    mean_radius = st.sidebar.slider('mean radius', float(X['mean radius'].min()), float(X['mean radius'].max()), float(X['mean radius'].mean()))
    mean_texture = st.sidebar.slider('mean texture', float(X['mean texture'].min()), float(X['mean texture'].max()), float(X['mean texture'].mean()))
    mean_perimeter = st.sidebar.slider('mean perimeter', float(X['mean perimeter'].min()), float(X['mean perimeter'].max()), float(X['mean perimeter'].mean()))
    mean_area = st.sidebar.slider('mean area', float(X['mean area'].min()), float(X['mean area'].max()), float(X['mean area'].mean()))
    mean_smoothness = st.sidebar.slider('mean smoothness', float(X['mean smoothness'].min()), float(X['mean smoothness'].max()), float(X['mean smoothness'].mean()))

    data = {
        'mean radius': mean_radius,
        'mean texture': mean_texture,
        'mean perimeter': mean_perimeter,
        'mean area': mean_area,
        'mean smoothness': mean_smoothness
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user input features with the entire dataset
# This will be useful for the encoding phase
df = pd.concat([input_df, X], axis=0)

# Selects only the first row (the user input data)
df = df[:1]

# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Apply model to make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
breast_cancer_labels = np.array(['malignant', 'benign'])
st.write(breast_cancer_labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Model Accuracy')
st.write(f"Accuracy: {accuracy:.2f}")

