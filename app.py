import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Page Configuration
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️", layout="wide")

# 2. Advanced CSS for Background and "Glass" Cards
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
    }
    
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    h1 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        text-shadow: 2px 2px 10px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0055ff);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(0, 212, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic (Same as before) ---
data = {
    'text': ['Win cash now!', 'Meeting at 5?', 'Free gift card', 'Hello friend', 'Claim prize', 'How are you?'],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
model = MultinomialNB().fit(X, df['label'])

# --- UI Layout ---
st.title("🛡️ SENTINEL-X: AI SPAM ANALYSIS")
st.markdown("#### *Advanced Machine Learning Transaction & Communication Monitoring*")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_input = st.text_area("Analysis Engine Input", placeholder="Paste a suspicious message here...", height=150)
    
    if st.button("EXECUTE ANALYSIS"):
        if user_input:
            vect_input = tfidf.transform([user_input])
            prediction = model.predict(vect_input)[0]
            
            if prediction == 'spam':
                st.error("### ⚠️ THREAT DETECTED: SPAM")
            else:
                st.success("### ✅ MESSAGE VERIFIED: LEGITIMATE")
        else:
            st.warning("Input required for processing.")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with tech details
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
st.sidebar.header("System Metrics")
st.sidebar.info("Model: Multinomial Naive Bayes\n\nVectorization: TF-IDF\n\nStatus: Online")


