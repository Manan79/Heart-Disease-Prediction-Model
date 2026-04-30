# Heart Disease Prediction Model ❤️

An advanced, interactive web application built with Streamlit that predicts the risk of heart disease using Machine Learning and provides personalized health insights powered by AI (Groq API and LangChain). 

**Created by: Manan**

---

## 🌟 Features

- **🎯 Accurate Risk Prediction**: Uses a trained Machine Learning model to evaluate heart disease risk based on 14 critical health parameters.
- **🤖 AI-Powered Health Insights**: Integrates with Llama-4 via Groq to provide detailed, human-readable explanations of your risk assessment.
- **💬 Interactive Health Chatbot**: A dedicated AI assistant specializing in heart health, nutrition, exercise, and prevention strategies.
- **📱 Modern & Responsive UI**: Beautifully designed interface with custom CSS, animated interactions, and dynamic UI elements for an enhanced user experience.
- **📊 Explanatory Data Visualizations**: Provides a breakdown of risk factors to help you better understand the model's decision-making process.

## 🛠️ Technologies Used

- **Frontend/Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
- **AI / LLM Integration**: [LangChain](https://www.langchain.com/), [Groq](https://groq.com/) API (Llama-4-maverick model)
- **Dependency Management**: `uv` / `pip` (Python >=3.12)
- **Environment Management**: `python-dotenv`

## 📁 Project Structure

```text
├── app.py                      # Main Streamlit application
├── heart_disease_enriched.csv  # Enriched dataset used for feature columns
├── Machinelearning_code/       # Directory containing ML notebooks and scalers
│   ├── Heart_disease_model_train.ipynb # Model training Jupyter Notebook
│   └── scaler.pkl              # Pre-fitted scaler for data normalization
├── pyproject.toml              # Project dependencies and metadata
├── requirements.txt            # Requirements file
├── uv.lock                     # UV lock file for exact dependency resolution
└── README.md                   # Project documentation
```

## ⚙️ Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manan79/Heart-Disease-Prediction-Model.git
   cd Heart-Disease-Prediction-Model
   ```

2. **Set up the virtual environment:**
   This project uses `uv` for lightning-fast dependency management. You can install dependencies via `uv` or standard `pip`.
   ```bash
   # Using uv (Recommended)
   uv sync
   
   # OR using pip
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key (required for the AI chatbot and result summaries):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🚀 How to Use the Application

1. **Landing Page**: Start on the landing page to get a quick overview of the app's capabilities and read some quick facts about heart health.
2. **Prediction Page**: Navigate to the "Prediction" tab using the sidebar. Fill in your health metrics (age, BMI, cholesterol, etc.) using the intuitive sliders and dropdowns. Click "Assess Heart Disease Risk" to get your prediction and a personalized AI explanation of your results.
3. **Health Chatbot**: Switch to the "Health Chatbot" tab to ask questions about heart health, diet, exercise, and lifestyle changes. The AI is specifically tuned to focus on cardiovascular wellness.

## 🏥 Health Parameters Analyzed

The ML model processes the following 14 features to determine the probability of heart disease:
- **Demographics:** Age, Gender
- **Clinical Metrics:** BMI, HbA1c Level, Blood Glucose Level
- **Vitals & History:** Hypertension, Diabetes, Family History
- **Cholesterol Profile:** Total Cholesterol, HDL, LDL
- **Lifestyle Factors:** Smoking History, Physical Activity Level, Alcohol Intake

## 🔮 Future Improvements

- Implementing additional machine learning models for comparative analysis.
- Adding historical tracking for users to monitor their risk over time.
- Expanding the AI chatbot's knowledge base with the latest cardiovascular research papers.
- Enhancing data visualizations for better feature importance understanding.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Manan79/Heart-Disease-Prediction-Model/issues).

## 👨‍💻 Author

**Manan**
- GitHub: [@Manan79](https://github.com/Manan79)

## ⚠️ Disclaimer
**This application is for educational and informational purposes only.** The predictions and AI-generated insights should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional regarding any medical conditions or decisions.
