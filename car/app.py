import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("car_prices.csv")
    return data

# Prepare the data
data = load_data()
X = data.drop(columns='Price')
y = data['Price']

numeric_features = ['Year', 'EngineSize', 'Horsepower', 'MPG_City', 'MPG_Highway', 'Weight', 'Length']
categorical_features = ['Make', 'Model']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Streamlit app
st.title('Car Price Predictor')

def get_user_input():
    Make = st.selectbox('Make', options=data['Make'].unique())
    Model = st.selectbox('Model', options=data[data['Make'] == Make]['Model'].unique())
    Year = st.slider('Year', min_value=2000, max_value=2022, value=2020)
    EngineSize = st.number_input('Engine Size (L)', min_value=0.0, max_value=10.0, value=2.0)
    Horsepower = st.number_input('Horsepower', min_value=50, max_value=1000, value=150)
    MPG_City = st.number_input('MPG (City)', min_value=0, max_value=100, value=30)
    MPG_Highway = st.number_input('MPG (Highway)', min_value=0, max_value=100, value=38)
    Weight = st.number_input('Weight (lbs)', min_value=1000, max_value=10000, value=3000)
    Length = st.number_input('Length (inches)', min_value=100, max_value=300, value=180)
    
    user_data = {
        'Make': Make,
        'Model': Model,
        'Year': Year,
        'EngineSize': EngineSize,
        'Horsepower': Horsepower,
        'MPG_City': MPG_City,
        'MPG_Highway': MPG_Highway,
        'Weight': Weight,
        'Length': Length
    }
    return pd.DataFrame(user_data, index=[0])

input_df = get_user_input()

if st.button('Predict Price'):
    prediction = model.predict(input_df)
    st.subheader(f'Predicted Price: ${prediction[0]:.2f}')
