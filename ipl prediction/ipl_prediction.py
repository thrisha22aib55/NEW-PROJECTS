import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the dataset
data = pd.read_csv('ipl_data.csv')

# Clean data to remove leading/trailing spaces
data['team1'] = data['team1'].str.strip()
data['team2'] = data['team2'].str.strip()
data['winning_team'] = data['winning_team'].str.strip()

# Combine unique team names from both columns to fit the LabelEncoder
teams = pd.concat([data['team1'], data['team2']]).unique()

# Encode categorical variables
label_encoder = LabelEncoder()
label_encoder.fit(teams)
data['team1'] = label_encoder.transform(data['team1'])
data['team2'] = label_encoder.transform(data['team2'])
data['winning_team'] = label_encoder.transform(data['winning_team'])

# Feature engineering: Create a feature for match outcome (1 if team1 wins, 0 otherwise)
data['team1_win'] = data.apply(lambda row: 1 if row['winning_team'] == row['team1'] else 0, axis=1)

# Split the dataset
X = data[['team1', 'team2']]
y = data['team1_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
cross_val_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validation accuracy: {cross_val_scores.mean()}')

# Function to predict the winning team
def predict_winner(team1, team2):
    team1_encoded = label_encoder.transform([team1])[0]
    team2_encoded = label_encoder.transform([team2])[0]
    prediction = model.predict([[team1_encoded, team2_encoded]])[0]
    winning_team = team1 if prediction == 1 else team2
    return winning_team

# Streamlit app
st.title('IPL Winning Team Prediction')

# Get user input
team1 = st.selectbox('Select Team 1', label_encoder.classes_)
team2 = st.selectbox('Select Team 2', label_encoder.classes_)

if st.button('Predict Winner'):
    if team1 != team2:
        winner = predict_winner(team1, team2)
        st.write(f'The predicted winning team is: {winner}')
    else:
        st.write('Please select two different teams.')
