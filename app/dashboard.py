from pathlib import Path

import joblib
import pandas as pd
import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"
ML_DATA_PATH = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_random_forest.pkl"
CHROMA_PATH = BASE_DIR / "database" / "chroma_db"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

st.set_page_config(
    page_title="Telecom AI | Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 2. CUSTOM NEO-BRUTALIST DESIGN
# =====================================================


st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #f7f3ed;
        color: #111111;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fffdf8;
        border-right: 2px solid #111111;
    }

    [data-testid="stSidebar"] h2 {
        font-weight: 900;
        color: #111111;
    }

    /* Headings */
    h1, h2, h3 {
        color: #111111 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    p, label, span {
        color: #222222;
    }

    /* Hide Deploy button and top toolbar */
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #ffdc45;
        color: #111111;
        border: 2px solid #111111;
        border-radius: 0;
        font-weight: 800;
        box-shadow: 4px 4px 0 #111111;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #4c91ff;
        color: #111111;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0 #111111;
    }

    /* Input boxes */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stSlider {
        border-radius: 0 !important;
    }

    /* Metric cards */
    .metric-card {
        border: 2px solid #111111;
        padding: 20px;
        min-height: 125px;
        box-shadow: 5px 5px 0 #111111;
        animation: appear 0.5s ease;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-yellow { background: #ffdc45; }
    .metric-blue { background: #82b5ff; }
    .metric-green { background: #85d99a; }
    .metric-coral { background: #ff8585; }

    .metric-label {
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 900;
        line-height: 1.2;
    }

    .metric-note {
        font-size: 13px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Section cards */
    .neo-card {
        background: #fffdf8;
        border: 2px solid #111111;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 5px 5px 0 #111111;
        animation: appear 0.5s ease;
    }

    .neo-heading {
        display: inline-block;
        background: #ffdc45;
        border: 2px solid #111111;
        padding: 8px 14px;
        font-size: 17px;
        font-weight: 900;
        margin-bottom: 15px;
        box-shadow: 3px 3px 0 #111111;
    }

    /* Insight cards */
    .insight-card {
        background: #e8efff;
        border: 2px solid #111111;
        padding: 13px;
        margin: 10px 0;
        font-weight: 650;
        box-shadow: 3px 3px 0 #111111;
    }

    /* Sidebar brand */
    .brand-box {
        background: #ffdc45;
        border: 2px solid #111111;
        padding: 15px;
        margin-bottom: 25px;
        box-shadow: 4px 4px 0 #111111;
        font-size: 20px;
        font-weight: 900;
    }

    /* New professional main heading */
    .platform-header {
        padding: 8px 0 24px 0;
        border-bottom: 1px solid #d8dce5;
        margin-bottom: 30px;
    }

    .platform-kicker {
        color: #4776b8;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .platform-title {
        color: #172033;
        font-size: clamp(28px, 3.2vw, 42px);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0;
    }

    .platform-subtitle {
        color: #647084;
        font-size: 15px;
        margin-top: 10px;
        font-weight: 500;
    }

    /* Assistant heading */
    .section-title {
        color: #172033;
        font-size: 29px;
        line-height: 1.25;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin: 0 0 8px 0;
    }

    .section-description {
        color: #647084;
        font-size: 15px;
        margin: 0 0 20px 0;
    }

    .assistant-info {
        background: #edf3fb;
        border: 1px solid #d9e4f2;
        border-left: 4px solid #5b8fd1;
        border-radius: 8px;
        padding: 13px 16px;
        color: #34445c;
        font-size: 14px;
        margin: 16px 0 22px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        border-top: 1px solid #d8dce5;
        margin-top: 35px;
        padding: 20px;
        font-size: 13px;
        font-weight: 700;
    }

    /* Animation */
    @keyframes appear {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Table */
    [data-testid="stDataFrame"] {
        border: 2px solid #111111;
    }

    /* Divider */
    hr {
        border-color: #d8dce5 !important;
        border-width: 1px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =====================================================
# 3. LOAD DATA AND MODEL
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_ml_columns():
    ml_df = pd.read_csv(ML_DATA_PATH)
    return ml_df.drop(columns=["Churn"]).columns.tolist()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    feature_columns = load_ml_columns()
    model = load_model()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def metric_card(label, value, note, color):
    st.markdown(
        f"""
        <div class="metric-card {color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_prediction_input(customer):
    input_df = pd.DataFrame([customer])
    input_df = pd.get_dummies(
        input_df,
        columns=input_df.select_dtypes(include=["object"]).columns,
        drop_first=True,
        dtype=int,
    )
    return input_df.reindex(columns=feature_columns, fill_value=0)


def get_risk(probability):
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def style_table(data):
    return (
        data.style
        .apply(
            lambda row: [
                "background-color: #e8efff"
                if row.name % 2 == 0
                else "background-color: #fffdf8"
                for _ in row
            ],
            axis=1,
        )
        .set_properties(
            **{
                "color": "#111111",
                "border": "1px solid #c8c8c8",
                "padding": "8px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#ffdc45"),
                        ("color", "#111111"),
                        ("font-weight", "900"),
                        ("border", "2px solid #111111"),
                    ],
                }
            ]
        )
    )


# =====================================================
# 5. RAG CHATBOT FUNCTIONS
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name="telecom_customer_knowledge")



def get_rag_answer(question):

    # =========================================
    # 1. BASIC CONVERSATION / READY ANSWERS
    # =========================================

    q = question.lower().strip()
    q = q.replace("?", "").replace("!", "").replace(".", "")

    basic_answers = {

        "hello": "Hello! Welcome to the Telecom AI Customer Intelligence Platform. How can I help you today?",

        "hi": "Hi! Nice to meet you. What would you like to know about our customer intelligence project?",

        "hey": "Hey! I'm your AI Customer Intelligence Assistant. Feel free to ask me anything about the project.",

        "how are you": "I'm doing great! Thanks for asking. I'm ready to help you with customer analytics, churn prediction and machine learning. How are you doing?",

        "how are u": "I'm doing great! Thanks for asking. How can I help you today?",

        "who are you": "I'm the AI Customer Intelligence Assistant, built for the Telecom Customer Churn Prediction project. I use Llama 3.2, Ollama, ChromaDB and Retrieval-Augmented Generation (RAG).",

        "what is your name": "I'm your Telecom AI Customer Intelligence Assistant!",

        "what can you do": "I can answer questions about customer churn, customer analytics, business insights, machine learning models, prediction results and retention recommendations using our project knowledge base.",

        "thank you": "You're very welcome! Feel free to ask me more questions about the Telecom AI project.",

        "thanks": "You're welcome! Happy to help.",

        "good morning": "Good morning! Welcome to the Telecom AI Customer Intelligence Platform. How can I help you?",

        "good afternoon": "Good afternoon! What would you like to know about our project?",

        "good evening": "Good evening! I'm ready to answer your questions about customer intelligence and churn prediction.",

        "bye": "Goodbye! Thank you for using the Telecom AI Customer Intelligence Platform. Have a great day!",

        "goodbye": "Goodbye! It was nice helping you. Have a great day!",

        "what is customer churn": "Customer churn means customers stop using a company's products or services. In our telecom project, churn means a customer has discontinued their telecom subscription.",

        "what is rag": "RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from a knowledge base and provides that information to an AI model to generate a context-based answer.",

        "what is ollama": "Ollama is a tool that allows us to run large language models locally on our computer. In this project, we use it to run Llama 3.2.",

        "what is chromadb": "ChromaDB is a vector database used to store document embeddings and retrieve relevant information based on semantic similarity.",

        "what is llama 3.2": "Llama 3.2 is the language model used in our project through Ollama to generate answers from retrieved project information.",

        "what dataset did we use": "We used the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 original columns.",

        "which model did we use": "We trained Logistic Regression and Random Forest models for customer churn prediction. Random Forest achieved a slightly better ROC-AUC and F1-score.",

        "what is machine learning": "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions without explicitly programming every rule.",

        "what is your purpose": "My purpose is to help users understand telecom customer behaviour, identify churn patterns and explore machine learning results from this project."
    }

    # Exact basic question matching
    if q in basic_answers:
        return basic_answers[q], ["Built-in FAQ"]

    # =========================================
    # 2. HANDLE COMMON GREETING VARIATIONS
    # =========================================

    if q.startswith("how are you"):
        return basic_answers["how are you"], ["Built-in FAQ"]

    if q.startswith(("hello", "hi ", "hey ")):
        return basic_answers["hello"], ["Built-in FAQ"]

    if q.startswith("thank"):
        return basic_answers["thank you"], ["Built-in FAQ"]

    # =========================================
    # 3. PROJECT KNOWLEDGE BASE SEARCH
    # =========================================

    embedding_model = load_embedding_model()
    collection = load_chroma_collection()

    document_count = collection.count()

    if document_count == 0:
        return (
            "The knowledge base is empty. Please run src/ingest_documents.py first.",
            []
        )

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(3, document_count),
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "I couldn't find relevant project information for this question.", []

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown")
        context_parts.append(
            f"Source: {source}\n{document}"
        )

    context = "\n\n".join(context_parts)

    # =========================================
    # 4. GENERATE ANSWER USING LLAMA 3.2
    # =========================================

    prompt = f"""
You are a friendly and professional Telecom Customer Intelligence Assistant.

Answer the user's question using the retrieved project information.

PROJECT INFORMATION:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Give a direct answer.
2. Use simple and understandable English.
3. Use only facts supported by the retrieved context.
4. Never invent statistics.
5. If information is unavailable, clearly say so.
6. Explain technical terms in simple language.
7. Distinguish facts from possible explanations.
8. Be friendly and conversational.

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        },
        timeout=600
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    sources = list(
        dict.fromkeys(
            metadata.get("source", "Unknown")
            for metadata in metadatas
        )
    )

    return answer, sources



# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            📡 TELECOM AI<br>
            <span style="font-size:13px;">
                CUSTOMER INTELLIGENCE PLATFORM
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### NAVIGATION")
    page = st.radio(
        "Select Page",
        [
            "Business Analytics",
            "Customer Churn Prediction",
            "Customer Risk Insights",
            "AI Customer Intelligence Assistant",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### PROJECT")
    st.markdown(
        """
        **AI-Powered Customer Intelligence**

        - Data Analytics
        - Statistical Analysis
        - Machine Learning
        - Customer Risk Scoring
        - Retention Recommendations
        - GenAI Business Insights
        - RAG Chatbot
        """
    )
    st.divider()
    st.caption("Built with Python, Pandas, Scikit-learn, Ollama and Streamlit")


# =====================================================
# 7. COMMON HEADER
# =====================================================


st.markdown(
    """
<div class="platform-header">
<div class="platform-kicker">TELECOM AI · CUSTOMER ANALYTICS</div>
<div class="platform-title">Customer Intelligence Platform</div>
<div class="platform-subtitle">Churn Prediction · Risk Intelligence · Business Insights</div>
</div>
    """,
    unsafe_allow_html=True
)



# =====================================================
# PAGE 1: BUSINESS ANALYTICS
# =====================================================

if page == "Business Analytics":
    st.markdown("## Business Analytics Overview")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("TOTAL CUSTOMERS", f"{total_customers:,}",
                    "Complete customer base", "metric-yellow")
    with c2:
        metric_card("CHURNED CUSTOMERS", f"{churned:,}",
                    "Customers who left", "metric-coral")
    with c3:
        metric_card("RETAINED CUSTOMERS", f"{retained:,}",
                    "Customers who stayed", "metric-green")
    with c4:
        metric_card("CHURN RATE", f"{churn_rate:.2f}%",
                    "Overall customer churn", "metric-blue")

    st.markdown("### Customer Churn Distribution")
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="neo-heading">Churn Count</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(df["Churn"].value_counts())

    with right:
        st.markdown(
            '<div class="neo-heading">Churn by Contract</div>',
            unsafe_allow_html=True,
        )
        contract_data = pd.crosstab(
            df["Contract"], df["Churn"], normalize="index"
        ) * 100
        st.bar_chart(contract_data)

    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="neo-heading">Churn by Payment Method</div>',
            unsafe_allow_html=True,
        )
        payment_data = pd.crosstab(
            df["PaymentMethod"], df["Churn"], normalize="index"
        ) * 100
        st.bar_chart(payment_data)

    with right:
        st.markdown(
            '<div class="neo-heading">Churn by Internet Service</div>',
            unsafe_allow_html=True,
        )
        internet_data = pd.crosstab(
            df["InternetService"], df["Churn"], normalize="index"
        ) * 100
        st.bar_chart(internet_data)

    st.markdown("### Key Business Insights")
    st.markdown(
        """
        <div class="insight-card">
            01. Overall churn rate is approximately 26.54%.
        </div>
        <div class="insight-card">
            02. Month-to-month contract customers show substantially higher churn.
        </div>
        <div class="insight-card">
            03. Electronic check customers have higher churn than other payment groups.
        </div>
        <div class="insight-card">
            04. New customers are the most vulnerable tenure segment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# PAGE 2: CUSTOMER CHURN PREDICTION
# =====================================================

elif page == "Customer Churn Prediction":
    st.markdown("## Individual Customer Churn Prediction")
    st.write("Enter customer details to estimate the probability of churn.")

    with st.form("customer_prediction_form"):
        st.markdown("### Customer Information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        col3, col4 = st.columns(2)
        with col3:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0,
                max_value=200.0, value=70.0
            )
        with col4:
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0,
                max_value=10000.0, value=840.0
            )

        submitted = st.form_submit_button(
            "PREDICT CUSTOMER CHURN", use_container_width=True
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = create_prediction_input(customer)
        probability = model.predict_proba(input_df)[0][1]
        risk = get_risk(probability)

        st.markdown("---")
        st.markdown("## Prediction Results")
        r1, r2 = st.columns(2)

        with r1:
            metric_card(
                "CHURN PROBABILITY", f"{probability:.2%}",
                "Model estimated probability", "metric-blue"
            )

        with r2:
            color = (
                "metric-coral" if risk == "HIGH"
                else "metric-yellow" if risk == "MEDIUM"
                else "metric-green"
            )
            metric_card(
                "CUSTOMER RISK", risk,
                "Model-based risk category", color
            )

        if risk == "HIGH":
            st.error(
                "Recommended action: Prioritize this customer for a retention "
                "call and investigate service concerns."
            )
        elif risk == "MEDIUM":
            st.warning(
                "Recommended action: Review customer satisfaction and consider "
                "personalized plan recommendations."
            )
        else:
            st.success(
                "Recommended action: Continue regular engagement and monitor "
                "customer activity."
            )

        st.caption(
            "Prediction is an estimate, not a guarantee of future churn. "
            "Risk thresholds are provisional business rules."
        )


# =====================================================
# PAGE 3: CUSTOMER RISK INSIGHTS
# =====================================================

elif page == "Customer Risk Insights":
    st.markdown("## Customer Risk Intelligence")
    st.write(
        "Identify at-risk customers, key churn drivers and retention opportunities."
    )

    risk_input = df.drop(columns=["customerID", "Churn"]).copy()
    risk_input = pd.get_dummies(
        risk_input,
        columns=risk_input.select_dtypes(include=["object"]).columns,
        drop_first=True,
        dtype=int,
    )
    risk_input = risk_input.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(risk_input)[:, 1]
    risk_report = pd.DataFrame(
        {
            "Customer ID": df["customerID"],
            "Tenure": df["tenure"],
            "Contract": df["Contract"],
            "Monthly Charges": df["MonthlyCharges"],
            "Churn Probability": probabilities,
        }
    )
    risk_report["Risk Category"] = [
        get_risk(p) for p in probabilities
    ]

    total = len(risk_report)
    high_count = (risk_report["Risk Category"] == "HIGH").sum()
    medium_count = (risk_report["Risk Category"] == "MEDIUM").sum()
    low_count = (risk_report["Risk Category"] == "LOW").sum()

    st.markdown(f"**Dataset:** {total:,} Customers")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("TOTAL CUSTOMERS", f"{total:,}",
                    "Customer records analysed", "metric-yellow")
    with c2:
        metric_card("HIGH RISK", f"{high_count:,}",
                    f"{high_count / total * 100:.2f}% of customers",
                    "metric-coral")
    with c3:
        metric_card("MEDIUM RISK", f"{medium_count:,}",
                    f"{medium_count / total * 100:.2f}% of customers",
                    "metric-blue")
    with c4:
        metric_card("LOW RISK", f"{low_count:,}",
                    f"{low_count / total * 100:.2f}% of customers",
                    "metric-green")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="neo-heading">01. Customer Risk Distribution</div>',
            unsafe_allow_html=True,
        )
        risk_counts = pd.Series(
            {
                "High Risk": high_count,
                "Medium Risk": medium_count,
                "Low Risk": low_count,
            }
        )
        risk_table = pd.DataFrame(
            {
                "Risk Category": risk_counts.index,
                "Customers": risk_counts.values,
                "Percentage": [
                    f"{x / total * 100:.2f}%"
                    for x in risk_counts.values
                ],
            }
        )
        st.dataframe(
            style_table(risk_table),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(risk_counts)

    with right:
        st.markdown(
            '<div class="neo-heading">02. Top Churn Prediction Factors</div>',
            unsafe_allow_html=True,
        )
        importance_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values(
            by="Importance", ascending=False
        ).head(10)

        importance_display = importance_df.copy()
        importance_display["Importance"] = (
            importance_display["Importance"].map(lambda x: f"{x:.4f}")
        )
        st.dataframe(
            style_table(importance_display),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Feature importance describes model contribution, not causal impact."
        )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="neo-heading">03. Churn Probability Distribution</div>',
            unsafe_allow_html=True,
        )
        probability_bins = pd.cut(
            risk_report["Churn Probability"],
            bins=[
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ],
            include_lowest=True,
        )
        probability_counts = probability_bins.value_counts(sort=False)
        probability_table = pd.DataFrame(
            {
                "Probability Range": [
                    f"{interval.left:.1f} - {interval.right:.1f}"
                    for interval in probability_counts.index
                ],
                "Customers": probability_counts.values,
                "Percentage": [
                    f"{x / total * 100:.2f}%"
                    for x in probability_counts.values
                ],
            }
        )
        st.dataframe(
            style_table(probability_table),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown(
            '<div class="neo-heading">04. High-Risk Customers</div>',
            unsafe_allow_html=True,
        )
        high_risk = risk_report[
            risk_report["Risk Category"] == "HIGH"
        ].sort_values(
            by="Churn Probability", ascending=False
        )
        high_risk_display = high_risk.head(10).copy()
        high_risk_display["Churn Probability"] = (
            high_risk_display["Churn Probability"].map(
                lambda x: f"{x:.2%}"
            )
        )
        st.dataframe(
            style_table(high_risk_display),
            hide_index=True,
            use_container_width=True,
        )
        csv_data = high_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            "DOWNLOAD HIGH-RISK CUSTOMER CSV",
            data=csv_data,
            file_name="high_risk_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="neo-heading">05. Key Business Insights</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-card">
                <b>01.</b> Identify high-risk customers before they leave.
            </div>
            <div class="insight-card">
                <b>02.</b> Tenure and Total Charges are among the strongest model predictors.
            </div>
            <div class="insight-card">
                <b>03.</b> Contract type and payment method are important churn-related characteristics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="neo-heading">06. Recommended Actions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-card">
                <b>01.</b> Prioritize high-risk customers for retention outreach.
            </div>
            <div class="insight-card">
                <b>02.</b> Offer suitable plans to month-to-month customers.
            </div>
            <div class="insight-card">
                <b>03.</b> Investigate service and pricing concerns.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Risk labels use provisional probability thresholds: High ≥ 70%, "
        "Medium ≥ 40% and Low < 40%. These are model predictions, not "
        "confirmed future outcomes."
    )


# =====================================================
# PAGE 4: AI CUSTOMER INTELLIGENCE ASSISTANT (RAG)
# =====================================================

elif page == "AI Customer Intelligence Assistant":
    st.markdown(
         """
<div class="section-title">AI Customer Intelligence Assistant</div>
<div class="section-description">Ask questions about customer churn, business insights and model performance.</div>
<div class="assistant-info">Search the project's knowledge base and get answers powered by the local Llama 3.2 model through Ollama.</div>
    """,
    unsafe_allow_html=True
)
   

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    question = st.chat_input(
        "Ask about customer churn, risk or ML performance..."
    )

    if question:
        st.session_state.rag_messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching project knowledge base and generating answer..."
            ):
                try:
                    answer, sources = get_rag_answer(question)
                    st.markdown(answer)
                    if sources:
                        st.caption("Sources: " + ", ".join(sources))

                    st.session_state.rag_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except requests.exceptions.ConnectionError:
                    error_message = (
                        "Cannot connect to Ollama. Please open Ollama and "
                        "make sure the local service is running."
                    )
                    st.error(error_message)
                    st.session_state.rag_messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                except requests.exceptions.Timeout:
                    error_message = (
                        "Ollama took too long to respond. Please try again."
                    )
                    st.error(error_message)
                    st.session_state.rag_messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                except Exception as error:
                    error_message = f"Something went wrong: {error}"
                    st.error(error_message)
                    st.session_state.rag_messages.append(
                        {"role": "assistant", "content": error_message}
                    )

    if st.session_state.rag_messages:
        if st.button("Clear Chat History"):
            st.session_state.rag_messages = []
            st.rerun()


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        TELECOM AI CUSTOMER INTELLIGENCE PLATFORM<br>
        Built with Python | Pandas | Scikit-learn | Streamlit | Ollama | ChromaDB
    </div>
    """,
    unsafe_allow_html=True,
)
