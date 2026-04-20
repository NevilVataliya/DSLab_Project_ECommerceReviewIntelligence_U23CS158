import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# ==========================================
# 0. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="E-Commerce AI", page_icon="📦", layout="wide")

# Ensure VADER lexicon is downloaded for the Helpfulness features
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ==========================================
# 1. LOAD MODELS (Cached for Speed)
# ==========================================
@st.cache_resource
def load_models():
    # Helpfulness Models
    ensemble = joblib.load('ecommerce_ensemble_model.joblib')
    fallbacks = joblib.load('ecommerce_fallbacks_ensembled.joblib')
    
    # NLP Sentiment Models
    sentiment_model = joblib.load('sentiment_lgb_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    return ensemble, fallbacks, sentiment_model, vectorizer

try:
    ensemble_models, fallback_models, nlp_model, tfidf_vec = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure all 4 .joblib files are in the directory.")
    models_loaded = False

sia = SentimentIntensityAnalyzer()
sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("⚙️ AI Engines")
st.sidebar.markdown("Select a tool below:")

app_mode = st.sidebar.radio(
    "",
    ["📝 Sentiment Analyzer (NLP)", "👍 Helpfulness Predictor"]
)

st.sidebar.divider()
st.sidebar.info(
    "**About this App:**\n\n"
    "Built with a Dual-Pipeline Architecture using LightGBM, Random Forest, and TF-IDF Text Vectorization."
)

# ==========================================
# 3. MODE 1: SENTIMENT ANALYZER
# ==========================================
if app_mode == "📝 Sentiment Analyzer (NLP)":
    st.title("📝 Semantic Tone Analyzer")
    st.markdown("Identify the underlying emotional tone of a product review using our custom-trained LightGBM NLP model.")
    
    review_text = st.text_area("Paste the review text here:", height=200, placeholder="E.g., The product arrived broken and the customer service was terrible...")
    
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter some text to analyze.")
        elif models_loaded:
            with st.spinner('Analyzing semantics...'):
                time.sleep(0.5) # Slight UX delay to feel like "thinking"
                
                # Vectorize and Predict
                text_vec = tfidf_vec.transform([review_text])
                pred_num = nlp_model.predict(text_vec)[0]
                probabilities = nlp_model.predict_proba(text_vec)[0]
                confidence = probabilities[pred_num] * 100
                
                final_tone = sentiment_mapping[pred_num]
                
                # Dynamic UX styling based on sentiment
                st.divider()
                st.subheader("Result:")
                if final_tone == "Positive":
                    st.success(f"### 🟢 {final_tone}\n**Confidence:** {confidence:.2f}%")
                elif final_tone == "Negative":
                    st.error(f"### 🔴 {final_tone}\n**Confidence:** {confidence:.2f}%")
                else:
                    st.info(f"### ⚪ {final_tone}\n**Confidence:** {confidence:.2f}%")

# ==========================================
# 4. MODE 2: HELPFULNESS PREDICTOR
# ==========================================
elif app_mode == "👍 Helpfulness Predictor":
    st.title("👍 Review Helpfulness Predictor")
    st.markdown("Predict how many upvotes a review will get per month. The system automatically routes data through a **4-State Ensemble Architecture** to handle missing historical data.")
    
    # Section A: Content
    st.header("1. Review Content")
    review_text = st.text_area("Review Text", height=150, placeholder="Type the review here...")
    
    col_a, col_b = st.columns(2)
    with col_a:
        overall_rating = st.slider("Star Rating Given by User", 1, 5, 5, help="The 1-5 star rating the user left for the product.")
    with col_b:
        image_count = st.number_input("Number of Images Attached", min_value=0, value=0, help="Did the user attach photos? Visual proof increases helpfulness.")

    # Section B: Metadata
    st.header("2. Historical Metadata")
    st.caption("Leave fields blank to simulate a **Cold Start** (New User or New Product). The AI will dynamically reroute the prediction.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 User History")
        user_avg_votes = st.text_input("User's Historical Avg Helpful Votes", placeholder="e.g., 2.5", help="If this user has reviewed before, how many votes do they usually get?")
    with col2:
        st.markdown("#### 📦 Product History")
        item_avg_votes = st.text_input("Product's Avg Helpful Votes", placeholder="e.g., 5.1", help="How many votes do reviews for this specific product usually get?")
        item_avg_rating = st.text_input("Product's Avg Star Rating", placeholder="e.g., 4.2", help="The overall 1-5 star rating of the product itself.")

    # Execute Prediction
    if st.button("Predict Helpfulness", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter some review text.")
        elif models_loaded:
            with st.spinner('Routing data through Ensemble Architecture...'):
                time.sleep(0.8) # UX delay
                
                # 1. Feature Engineering
                word_count = len(review_text.split())
                paragraph_count = review_text.count('\n') + 1
                vader_score = sia.polarity_scores(review_text)['compound']
                
                # 2. Parse historical inputs
                is_new_user = user_avg_votes.strip() == ""
                is_new_prod = item_avg_votes.strip() == "" or item_avg_rating.strip() == ""
                
                user_rep = float(user_avg_votes) if not is_new_user else 0.0
                item_rep = float(item_avg_votes) if not is_new_prod else 0.0
                item_rating = float(item_avg_rating) if not is_new_prod else 0.0
                rating_deviation = abs(overall_rating - item_rating) if not is_new_prod else 0.0

                # 3. The 4-State Router
                prediction = 0.0
                routed_model = ""

                if is_new_user and is_new_prod:
                    routed_model = "Pure Content Ensemble (Cold Start: Missing User & Product)"
                    features = pd.DataFrame([{'overall': overall_rating, 'word_count': word_count, 'image_count': image_count, 'paragraph_count': paragraph_count, 'sentiment_score': vader_score}])
                    model_dict = fallback_models['New_Both']
                    
                elif is_new_user:
                    routed_model = "Product-Biased Ensemble (Cold Start: Missing User)"
                    ordered_cols = ['overall', 'word_count', 'image_count', 'paragraph_count', 'item_avg_helpful_votes', 'rating_deviation', 'sentiment_score']
                    features = pd.DataFrame([{'overall': overall_rating, 'word_count': word_count, 'image_count': image_count, 'paragraph_count': paragraph_count, 'item_avg_helpful_votes': item_rep, 'rating_deviation': rating_deviation, 'sentiment_score': vader_score}])[ordered_cols]
                    model_dict = fallback_models['New_User']
                    
                elif is_new_prod:
                    routed_model = "User-Biased Ensemble (Cold Start: Missing Product)"
                    ordered_cols = ['overall', 'word_count', 'image_count', 'user_avg_helpful_votes', 'paragraph_count', 'sentiment_score']
                    features = pd.DataFrame([{'overall': overall_rating, 'word_count': word_count, 'image_count': image_count, 'user_avg_helpful_votes': user_rep, 'paragraph_count': paragraph_count, 'sentiment_score': vader_score}])[ordered_cols]
                    model_dict = fallback_models['New_Product']
                    
                else:
                    routed_model = "Main 8-Feature Ensemble (Ideal State: Complete Data)"
                    ordered_cols = ['overall', 'word_count', 'image_count', 'user_avg_helpful_votes', 'paragraph_count', 'item_avg_helpful_votes', 'rating_deviation', 'sentiment_score']
                    features = pd.DataFrame([{'overall': overall_rating, 'word_count': word_count, 'image_count': image_count, 'user_avg_helpful_votes': user_rep, 'paragraph_count': paragraph_count, 'item_avg_helpful_votes': item_rep, 'rating_deviation': rating_deviation, 'sentiment_score': vader_score}])[ordered_cols]
                    model_dict = ensemble_models
                    model_dict['lgbm_weight'] = 0.38
                    model_dict['rf_weight'] = 0.62

                # 4. Predict and Blend
                lgbm_pred = np.maximum(0, model_dict['lightgbm'].predict(features)[0])
                rf_pred = np.maximum(0, model_dict['random_forest'].predict(features)[0])
                prediction = (lgbm_pred * model_dict.get('lgbm_weight', 0.50)) + (rf_pred * model_dict.get('rf_weight', 0.50))

                # 5. Output
                st.divider()
                st.metric(label="Predicted Helpfulness", value=f"{prediction:.2f} Upvotes / Month")
                st.info(f"**System Log:** Data dynamically routed to **{routed_model}**")