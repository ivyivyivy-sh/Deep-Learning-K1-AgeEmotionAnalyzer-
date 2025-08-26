import streamlit as st
import keras
import pandas as pd
from src.analyzer import Analyzer

# Disable keras interactive logging
keras.config.disable_interactive_logging()

# Initialize analyzer
a = Analyzer()

# Session state initialization
if "done" not in st.session_state:
    st.session_state.done = False

if "results" not in st.session_state:
    st.session_state.results = None

def to_csv(df):
    """Convert DataFrame to CSV string"""
    return df.to_csv(index=False)

def progress(p):
    """Display progress bar"""
    st.progress(p)

@st.cache_resource
def load_model():
    """Load emotion detection model with caching"""
    return keras.models.load_model("modelv1.keras")

@st.cache_resource
def load_age_model():
    """Load age prediction model with caching"""
    # This function can be used if you have a separate age model
    # For now, age model is loaded within the Analyzer class
    return None

def analyze():
    """Main analysis function"""
    with st.spinner("Analyzing video for emotions and age..."):
        model = load_model()
        # Analyze video for both emotions and age
        st.session_state.done, st.session_state.results = a.analyze(
            file=file, 
            model=model, 
            skip=skip, 
            confidence=confidence
        )

# Main UI
st.title("Audience Emotion and Age Analyzer")

st.header("What do I do?")
st.write("Upload a video file for analysis. The system will detect faces, analyze emotions, and predict age ranges. Change settings as needed and press the Analyze button.")

# File upload section
file = st.file_uploader('Select video file to upload...', type=['mp4'])

# Configuration columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Analysis settings
skip = col1.number_input('Frames to skip', 1, None, 100, 
                        help="Higher values speed up analysis but may miss brief expressions")
col2.info("Skip frames to speed up analysis.  \n1 means all frames will be analyzed.")

confidence = col3.number_input('Emotion analyzer confidence', 0., 1., 0.5,
                              help="Minimum confidence level for emotion predictions")
col4.info("Return predictions above this probability. Higher values mean more conservative predictions.")

# Age prediction info
st.subheader("Age and Emotion Prediction")

# Analyze button
st.button("Analyze Video", on_click=analyze, type="primary")

# Results section
if st.session_state.done:
    st.success("Analysis completed successfully!")
    
    # Display summary statistics
    st.subheader("Analysis Summary")
    
    if not st.session_state.results.empty:
        # Total faces detected
        total_faces = len(st.session_state.results)
        st.write(f"**Total faces analyzed:** {total_faces}")
        
        # Age distribution analysis
        age_columns = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        available_age_cols = [col for col in age_columns if col in st.session_state.results.columns]
        
        if available_age_cols:
            # Get dominant age group for each face
            age_data = st.session_state.results[available_age_cols]
            dominant_ages = age_data.idxmax(axis=1)
            
            # Age distribution chart
            st.write("**Age Group Distribution:**")
            age_counts = dominant_ages.value_counts()
            st.bar_chart(age_counts)
            
            # Most common age group
            if not age_counts.empty:
                most_common_age = age_counts.index[0]
                st.write(f"**Most common age group:** {most_common_age}")
        
        # Emotion statistics
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        available_emotion_cols = [col for col in emotion_cols if col in st.session_state.results.columns]
        
        if available_emotion_cols:
            # Count occurrences of each emotion
            emotion_counts = {}
            for emotion in available_emotion_cols:
                emotion_counts[emotion] = st.session_state.results[emotion].sum()
            
            st.write("**Emotion Detection Summary:**")
            for emotion, count in emotion_counts.items():
                if count > 0:
                    st.write(f"- {emotion.capitalize()}: {int(count)} faces")
    
    # Download results
    st.subheader("Download Results")
    filename = file.name if file else "recorded_video"
    st.download_button(
        "Download Full Report (CSV)", 
        to_csv(st.session_state.results), 
        f'report/{filename}_emotion_age_results.csv',
        help="Contains detailed emotion and age predictions for each detected face"
    )
    
    # Show sample data
    with st.expander("View Sample Data"):
        st.dataframe(st.session_state.results.head(), use_container_width=True)

# Status information in sidebar
with st.sidebar:
    st.header("Analysis Status")
    if st.session_state.done:
        st.success("✅ Analysis Complete")
        if not st.session_state.results.empty:
            st.write(f"Faces analyzed: {len(st.session_state.results)}")
    else:
        st.info("⏳ Ready for analysis")
    
 