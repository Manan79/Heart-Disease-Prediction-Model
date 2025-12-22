import streamlit as st
import pandas as pd
from datetime import datetime
import joblib  
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq     
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .safe-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        max-width: 70%;
        word-wrap: break-word;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .bot-bubble {
        background: #f8f9fa;
        color: #333;
        border: 1px solid #e9ecef;
    }
            
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        min-height: 500px;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
    }
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .chat-input-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset to get column names
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('heart_disease_enriched.csv')
        st.success("Dataset loaded successfully")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

# Load trained model - UPDATED
@st.cache_resource
def load_model():
    try:
        # Try loading with joblib first (since you used joblib to save)
        try:
            model = joblib.load("Machinelearning_code/heart_disease_pipeline.pkl")
            

            st.success("Model loaded successfully with joblib")
            return model 
        except Exception as e:
            # Fallback to pickle
            print("Joblib loading failed, trying pickle..." , e)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def predict_heart_disease(input_data):
    model  = load_model()
    if model is None:
        st.error("Could not load the model")
        return None, None

    try:

        features = ['gender', 'age', 'hypertension', 'smoking_history', 'bmi',
       'HbA1c_level', 'blood_glucose_level', 'diabetes',
       'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
       'family_history', 'physical_activity_level', 'alcohol_intake']
        
      
        
        # Check if all required features are present
        missing_features = [f for f in features if f not in input_data.columns]
        if missing_features:
            st.error(f"Missing features: {missing_features}")
            return None, None
        
        # Create DataFrame with only required features in correct order
        prediction_data = input_data[features].copy()
        
        prediction_data = prediction_data.copy()

        # Convert encoded UI values back to training categories
        prediction_data["physical_activity_level"] = prediction_data["physical_activity_level"].map(
            {1: "sedentary", 2: "active"}
        )

        prediction_data["alcohol_intake"] = prediction_data["alcohol_intake"].map(
            {0: "Never", 1: "Moderate", 2: "Regular"}
        )

        prediction = model.predict(prediction_data)
        probabilities = model.predict_proba(prediction_data)
        
        with st.expander("Debug Info: Prediction Details", expanded=False):  
                    st.write("Processed prediction data:", prediction_data )
                    # Debug probability information
                    st.write("Raw probabilities shape:", probabilities.shape)
                    st.write("Raw probabilities:", probabilities)
                    st.write("Input data columns:", input_data.columns.tolist())
                    # Feature importance visualization
                    st.image("image.png", caption="Feature Importances")
            
        
        # Ensure we have valid probabilities
        if probabilities.shape[1] != 2:
            st.error("Invalid probability output from model")
            return None, None
            
        # Extract probabilities for both classes
        no_disease_prob = float(probabilities[0][0])
        disease_prob = float(probabilities[0][1])
        
        # Validate probabilities
        if not (0 <= no_disease_prob <= 1 and 0 <= disease_prob <= 1):
            st.error("Invalid probability values")
            return None, None
            
        st.success("Prediction completed successfully")
        return int(prediction[0]), [no_disease_prob, disease_prob]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# AI risk factor summarizer
def generate_ai_summary(prediction, input_data, probability):

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides explanations for heart disease risk predictions based on model results "),
        ("user", "Given the following health data and prediction, provide a concise summary explaining the risk factors involved.\n\nHealth Data: {input_data}\nPrediction: {prediction}\nProbabilities: {probability}"),
        ("assistant", "If model predicts high risk, explain key risk factors such as age, hypertension, smoking history, BMI, HbA1c level, blood glucose level, and diabetes status. If low risk, highlight protective factors and healthy indicators."),
        ("assistant", "If model is uncertain (around 50% probability), mention the uncertainty and suggest consulting a healthcare professional for further evaluation."),
        ("system", "give in the para not in steps")
    ])
    chat_groq = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
    prompt = template.format_messages(input_data=input_data, prediction=prediction, probability=probability)
    response = chat_groq.invoke(prompt)
    return response.content
    


# Mock chatbot response
def get_chat_response(user_message, chat_history):
    SYSTEM_INSTRUCTION = """
    You are a specialized AI assistant. Your one and only purpose is to provide 
    information and tips about heart health. This includes cardiovascular health, 
    exercise, nutrition, sleep, stress management, and other related topics ,
    give answer only related to heart health and related topic .

    If the user asks a question that is NOT related to heart health (e.g., about 
    programming, history, politics, weather, etc.), you MUST politely decline. 
    Do not answer the unrelated question. Instead, gently remind them that you 
    only specialize in heart health and guide them back to a relevant topic.
    """
    chat_groq = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

    messagers = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for msg in chat_history:
        messagers.append({"role": msg['role'], "content": msg['content']})
    messagers.append({"role": "user", "content": user_message})
    response = chat_groq.invoke(messagers)
    return response.content


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Landing'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Sidebar navigation
with st.sidebar:
    st.markdown("## ❤️ Navigation")
    st.markdown("---")
    
    if st.button("🏠 Landing Page", use_container_width=True):
        st.session_state.page = 'Landing'
    if st.button("🔮 Prediction", use_container_width=True):
        st.session_state.page = 'Prediction'
    if st.button("💬 Health Chatbot", use_container_width=True):
        st.session_state.page = 'Chatbot'
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app helps assess heart disease risk using machine learning and provides heart health guidance.")

# Landing Page
if st.session_state.page == 'Landing':
    st.markdown('<h1 class="main-header">Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your health insights powered by AI</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("### Welcome to Your Heart Health Companion")
        st.markdown("""
        This innovative tool uses advanced machine learning to assess your risk of heart disease 
        based on key health parameters. Our goal is to provide you with valuable insights to 
        help you make informed decisions about your cardiovascular health.
        
        **Features:**
        - 🎯 Accurate risk prediction using trained ML models
        - 📊 Detailed AI-powered explanation of results
        - 💬 Interactive heart health chatbot
        - 📱 Mobile-friendly interface
        
        **How it works:**
        1. Navigate to the Prediction page
        2. Enter your health information
        3. Get instant risk assessment with AI explanation
        4. Chat with our health assistant for personalized tips
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("### Quick Facts")
        st.markdown("""
        ❤️ Heart disease is the leading cause of death worldwide
        
        ⏱️ Early detection can significantly improve outcomes
        
        🏃‍♂️ Lifestyle changes can reduce risk by up to 80%
        
        📊 Regular monitoring is key to prevention
        """)
        
        if st.button("Start Your Assessment Now", use_container_width=True):
            st.session_state.page = 'Prediction'
        st.markdown('</div>', unsafe_allow_html=True)

# Prediction Page
elif st.session_state.page == 'Prediction':
    st.markdown('<h1 class="main-header">Heart Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if not df.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Enter Your Health Information")
        
        # Define mappings as per dataset
        gender_map = {'Female': 0, 'Male': 1}
        smoking_map = {
            'never': 4,
            'former': 3,
            'current': 1,
            'ever': 2,
            'not current': 5,
            'No Info': 0
        }

        # Create two-column layout for inputs
        cols = st.columns(2)
        input_data = {}

        # Define the order of features as used in model training
        features_order = ['gender', 'age', 'hypertension', 'smoking_history', 'bmi',
       'HbA1c_level', 'blood_glucose_level', 'diabetes', 'hdl_cholesterol', 'ldl_cholesterol',
       'total_cholesterol','family_history', 'physical_activity_level', 'alcohol_intake']

        for i, column in enumerate(features_order):
            with cols[i % 2]:
                if column == 'gender':
                    gender_selected = st.selectbox('Gender', list(gender_map.keys()))
                    input_data[column] = [gender_map[gender_selected]]
                elif column == 'age':
                    input_data[column] = [st.slider('Age', 0, 100, 45)]
                elif column == 'hypertension':
                    input_data[column] = [st.selectbox('Hypertension', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")]
                elif column == 'smoking_history':
                    smoke_selected = st.selectbox('Smoking History', list(smoking_map.keys()))
                    input_data[column] = [smoking_map[smoke_selected]]
                elif column == 'bmi':
                    input_data[column] = [st.slider('BMI', 10.0, 70.0, 25.0, step=0.1)]
                elif column == 'HbA1c_level':
                    input_data[column] = [st.slider('HbA1c Level', 3.0, 10.0, 5.7, step=0.1)]
                elif column == 'hdl_cholesterol':
                    input_data[column] = [st.slider('hdl_cholesterol', 3.0, 10.0, 5.7, step=0.1)]
                elif column == 'ldl_cholesterol':
                    input_data[column] = [st.slider('ldl_cholesterol', 3.0, 10.0, 5.7, step=0.1)]
                elif column == 'blood_glucose_level':
                    input_data[column] = [st.slider('Blood Glucose Level', 50, 350, 120)]
                elif column == 'total_cholesterol':
                    input_data[column] = [st.slider('total_cholestrol', 50, 450, 120)]
                elif column == 'diabetes':
                    input_data[column] = [st.selectbox('Diabetes', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")]
                elif column == 'family_history':
                    input_data[column] = [st.selectbox('family_history', [0, 1] , format_func=lambda x: "Yes" if x == 1 else "No")]
                elif column == 'physical_activity_level':
                    input_data[column] = [st.selectbox('physical_activity_level', [1, 2] , format_func=lambda x: "sedentary" if x == 1 else "active")]
                elif column == 'alcohol_intake':
                    input_data[column] = [st.selectbox('alcohol_intake', [0,1,2] , format_func=lambda x: "Never" if x == 0 else "Moderate" if x ==1 else "Regularly")]




        st.markdown('</div>', unsafe_allow_html=True)
        if st.button('🔍 Assess Heart Disease Risk', use_container_width=True):
            with st.spinner('Analyzing your health data...'):
                # Convert input data to DataFrame for prediction
                input_df = pd.DataFrame(input_data)
                
                # Reorder columns to match model training
                input_df = input_df[features_order]
                
                # st.write("Input data for prediction:", input_df)
                
                # Make prediction
                prediction, probability = predict_heart_disease(input_df)
                
                if prediction is not None and probability is not None:
                    # Store results in session state
                    st.session_state.prediction = int(prediction)
                    st.session_state.probability = [float(p) for p in probability]
                    st.session_state.input_data = input_df
                    st.session_state.prediction_made = True
                    
                    # Generate AI summary
                    st.session_state.ai_summary = generate_ai_summary(prediction, input_df, probability)
                else:
                    st.error("Prediction failed - could not get valid results from the model")
        try:
            written_text = " "
            if st.session_state.probability[1] > 0.65 :
                written_text = "### ⚠️ Higher Risk of Heart Disease"
            elif st.session_state.probability[1] <= 0.65 and st.session_state.probability[1] >= 0.35:
                written_text = "### ⚠️ Moderate Risk of Heart Disease"
            else:
                written_text = "### ✅ Lower Risk of Heart Disease"
        except:
            written_text = "Model's Prediction"         
        
        # Display prediction results if available
        if st.session_state.get('prediction_made', False):
            st.markdown("## Prediction Results")
            
            if st.session_state.prediction == 1:
                # st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                # st.markdown("### ⚠️ Higher Risk of Heart Disease")
                st.markdown(f"{written_text}")
                st.markdown(f"**Probability:** {st.session_state.probability[1]*100:.1f}%")
                st.markdown("We recommend consulting with a healthcare professional for further evaluation.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"{written_text}")
                st.markdown(f"**Probability:** {st.session_state.probability[0]*100:.1f}%")
                st.markdown("Continue maintaining a healthy lifestyle with regular check-ups.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # AI Summary
            with st.expander("🤖 AI Explanation of Results", expanded=True):
                # st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### AI Analysis")
                st.info(st.session_state.ai_summary)
                st.markdown("""
                **Disclaimer:** This analysis is generated by an AI model and should not replace 
                professional medical advice. Always consult with healthcare professionals for 
                medical decisions.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("Unable to load dataset. Please check if the CSV file is available.")

# Chatbot Page
elif st.session_state.page == 'Chatbot':
    st.markdown('<h1 class="main-header">Heart Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me about heart health, exercise, nutrition, and prevention</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []

    # Display chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align: center; color: #6c757d; padding: 2rem;'>
            <h4>👋 Hello! I'm your Heart Health Assistant</h4>
            <p>I can help you with information about:</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>Heart-healthy foods and nutrition</li>
                <li>Exercise recommendations</li>
                <li>Blood pressure management</li>
                <li>Cholesterol control</li>
                <li>Stress reduction techniques</li>
                <li>Sleep and heart health</li>
                <li>Warning signs to watch for</li>
            </ul>
            <p><br>What would you like to know about heart health?</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input area
    user_input = st.chat_input("Type your heart health question here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input, 'timestamp': datetime.now()})
        # Get bot response
        bot_response = get_chat_response(user_input, st.session_state.chat_history)
        # Add bot response to chat history
        st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response, 'timestamp': datetime.now()})
        # Rerun so the newly appended messages are rendered immediately
        try:
            st.experimental_rerun()
        except Exception:
            # Fallback for older/newer   Streamlit versions
            try:
                st.rerun()
            except Exception:
                pass
        

# Footer
st.markdown("---")
st.markdown('<div class="footer">Made with ❤️ using Streamlit and AI</div>', unsafe_allow_html=True)    