pimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Set Streamlit Page Config (Must be the first command)
st.set_page_config(page_title="Movie Recommendations", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

df = load_data()

# Ensure correct column names
if 'genres' not in df.columns or 'title' not in df.columns:
    st.error("Dataset is missing required columns: 'genres' or 'title'")
    st.stop()

# Map genres to emotions (Example Mapping)
def map_genre_to_emotion(genre):
    if "Comedy" in genre:
        return "Happy"
    elif "Drama" in genre:
        return "Sad"
    elif "Thriller" in genre or "Action" in genre:
        return "Excited"
    elif "Romance" in genre:
        return "Love"
    else:
        return "Neutral"

df['emotion'] = df['genres'].apply(map_genre_to_emotion)

# Streamlit UI
st.title("🎬 Movie Recommendations Based on Emotions")
st.markdown("#### Find the perfect movie based on how you feel!")

# Sidebar Input
st.sidebar.header("🎭 Select Your Emotion")
user_input = st.sidebar.selectbox("Choose an Emotion", ["Happy", "Sad", "Excited", "Love", "Neutral"])

# Show Recommendations
if st.sidebar.button("🎥 Recommend Movies"):
    recommended_movies = df[df['emotion'].str.lower() == user_input.lower()]['title'].tolist()
    
    if recommended_movies:
        st.subheader(f"🎞 Recommended Movies for {user_input} Emotion:")
        for movie in recommended_movies[:5]:
            st.write(f"✅ {movie}")
    else:
        st.warning("No recommendations available for this emotion. Try another!")

# Graphs Section
st.markdown("---")
st.subheader("📊 Emotion Distribution in Movies")
fig, ax = plt.subplots()
sns.countplot(data=df, x='emotion', palette='coolwarm', ax=ax)
ax.set_xlabel("Emotion Category")
ax.set_ylabel("Number of Movies")
st.pyplot(fig)

# Confusion Matrix and Accuracy (Example Simulation)
st.subheader("📉 Confusion Matrix & Accuracy (Simulated Data)")
# Simulating some test labels for confusion matrix example
y_true = np.random.choice(["Happy", "Sad", "Excited", "Love", "Neutral"], size=100)
y_pred = np.random.choice(["Happy", "Sad", "Excited", "Love", "Neutral"], size=100)

cm = confusion_matrix(y_true, y_pred, labels=["Happy", "Sad", "Excited", "Love", "Neutral"])
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Happy", "Sad", "Excited", "Love", "Neutral"], yticklabels=["Happy", "Sad", "Excited", "Love", "Neutral"])
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
st.pyplot(fig_cm)

# Calculate and Display Accuracy
accuracy = accuracy_score(y_true, y_pred)
st.write(f"### 🔢 Model Accuracy (Simulated): {accuracy:.2f}")

# Explanation of How It Works
st.markdown("---")
st.subheader("📖 How It Works")
st.markdown(
    """
    1. The dataset contains movie titles and their genres.
    2. Genres are mapped to emotions based on predefined categories (e.g., Comedy -> Happy, Drama -> Sad).
    3. The user selects an emotion from the sidebar.
    4. The system filters movies that match the selected emotion and recommends them.
    5. A graph shows the distribution of movies per emotion.
    6. The confusion matrix (simulated for demo purposes) shows how well a classifier might predict emotions based on genres.
    7. Accuracy is calculated based on simulated predictions.
    """
)