import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    file_path = 'Crop_recommendation.csv'  # Update the path if needed
    return pd.read_csv(file_path)

# Train the model
@st.cache
def train_model(data):
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Custom CSS for Styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main Streamlit App
def main():
    # Add custom CSS for styling
    local_css("style.css")

    st.markdown('<h1 style="text-align: center; color: #4CAF50;">🌾 Crop Recommendation System 🌾</h1>', unsafe_allow_html=True)
    st.write("Enter the soil and environmental parameters to get a crop recommendation.")

    # Load data and train the model
    data = load_data()
    model, accuracy = train_model(data)

    # Display model accuracy
    st.markdown(f'<div class="accuracy">Model Accuracy: <b>{accuracy * 100:.2f}%</b></div>', unsafe_allow_html=True)

    # Input Form
    st.sidebar.markdown("<h3>Input Parameters</h3>", unsafe_allow_html=True)
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
    P = st.sidebar.slider("Phosphorus (P)", 0, 145, 40)
    K = st.sidebar.slider("Potassium (K)", 0, 205, 45)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 200.0)

    # Predict Button
    if st.sidebar.button("Recommend Crop"):
        new_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        prediction = model.predict(new_data)
        st.markdown(f'<div class="output">🌱 Recommended Crop: <b>{prediction[0]}</b></div>', unsafe_allow_html=True)

# Run the App
if __name__ == "__main__":
    main()
