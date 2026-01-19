import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

# Page Configuration
st.set_page_config(page_title="AI-Based NIDS", layout="wide", page_icon="🛡️")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Data Simulation Module ---
def generate_traffic_data(n_samples=1000, seed=None):
    """Generates synthetic high-fidelity CIC-IDS2017-like traffic data."""
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time() % 1000))
    
    # Features: duration, protocol_type, service, flag, src_bytes, dst_bytes, count, srv_count
    data = []
    for _ in range(n_samples):
        label = random.choice(["Normal", "DDoS", "Brute Force", "Malware"])
        
        if label == "Normal":
            duration = np.random.uniform(0.1, 2.0)
            src_bytes = np.random.randint(100, 5000)
            dst_bytes = np.random.randint(100, 5000)
            count = np.random.randint(1, 10)
        elif label == "DDoS":
            duration = np.random.uniform(0.01, 0.1)
            src_bytes = np.random.randint(5000, 10000) # High volume
            dst_bytes = np.random.randint(10, 100)
            count = np.random.randint(50, 200) # Flooding
        elif label == "Brute Force":
            duration = np.random.uniform(2.0, 10.0) # Slower attempts
            src_bytes = np.random.randint(50, 200)
            dst_bytes = np.random.randint(50, 200)
            count = np.random.randint(20, 50)
        else: # Malware
            duration = np.random.uniform(5.0, 60.0) # Persistent connection
            src_bytes = np.random.randint(1000, 20000)
            dst_bytes = np.random.randint(1000, 20000)
            count = np.random.randint(1, 5)

        data.append([duration, src_bytes, dst_bytes, count, label])

    df = pd.DataFrame(data, columns=['Duration', 'Src_Bytes', 'Dst_Bytes', 'Conn_Count', 'Label'])
    return df

# --- 2. Sidebar Implementation ---
st.sidebar.title("🛡️ AstraGuard AI")
st.sidebar.subheader("NIDS Control Panel")

menu = st.sidebar.radio("Navigation", ["Dashboard", "Train Model", "Live Simulation", "Project Documentation"])

# Initialize session state for the model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.0
if 'data' not in st.session_state:
    st.session_state.data = None

# --- 3. Main Dashboard ---
if menu == "Dashboard":
    st.title("Network Security Dashboard")
    st.info("AI-Based Network Intrusion Detection System using Random Forest Classifier.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>System Status</h3><h2 style="color:#00ff00;">Active</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Model Accuracy</h3><h2>{st.session_state.accuracy:.2f}%</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Threat Level</h3><h2 style="color:#ffcc00;">Low</h2></div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Traffic Overview (Historical Samples)")
    df = generate_traffic_data(500)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='Label', data=df, palette='viridis', ax=ax)
    plt.title("Distribution of Traffic Types")
    st.pyplot(fig)

elif menu == "Train Model":
    st.title("Model Training Module")
    st.write("Train the Random Forest model using Simulated Data or Custom Dataset.")
    
    data_source = st.radio("Choose Data Source:", ["Simulated Dataset", "Upload CSV (CIC-IDS2017 Format)"])
    
    if data_source == "Upload CSV (CIC-IDS2017 Format)":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Custom Dataset Uploaded Successfully!")
    else:
        if st.button("Generate Simulated Dataset"):
            st.session_state.data = generate_traffic_data(5000, seed=42)
            st.success("Simulated Dataset Generated (5000 samples)")

    if st.session_state.data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head())
        
        if st.button("Start Training"):
            with st.spinner("Optimizing Random Forest Hyperparameters..."):
                df = st.session_state.data
                # Basic cleaning if it's a real dataset
                df = df.dropna()
                
                X = df.drop('Label', axis=1)
                y = df['Label']
                
                # Handle numeric encoding for real datasets if needed
                X = pd.get_dummies(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                
                st.session_state.model = model
                st.session_state.accuracy = model.score(X_test, y_test) * 100
                
                st.success(f"Model Trained Successfully! Accuracy: {st.session_state.accuracy:.2f}%")
                
                st.subheader("Performance Metrics")
                y_pred = model.predict(X_test)
                st.code(classification_report(y_test, y_pred))

elif menu == "Live Simulation":
    st.title("Live Network Traffic Simulation")
    
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        st.write("Generating live packets and predicting in real-time...")
        
        status_container = st.empty()
        log_container = st.empty()
        
        stop_btn = st.button("Stop Simulation")
        
        logs = []
        for i in range(20):
            if stop_btn: break
            
            # Generate a random test sample
            test_df = generate_traffic_data(1)
            features = test_df.drop('Label', axis=1)
            true_label = test_df['Label'].values[0]
            
            prediction = st.session_state.model.predict(features)[0]
            
            color = "red" if prediction != "Normal" else "green"
            status_text = f"🚨 ALERT: {prediction} detected!" if prediction != "Normal" else "✅ Traffic: Normal"
            
            status_container.markdown(f"<h2 style='color:{color}; text-align:center;'>{status_text}</h2>", unsafe_allow_html=True)
            
            log_entry = {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Duration": f"{features['Duration'].values[0]:.2f}",
                "Bytes": features['Src_Bytes'].values[0],
                "Prediction": prediction
            }
            logs.insert(0, log_entry)
            log_container.table(pd.DataFrame(logs).head(10))
            
            time.sleep(1.5)

elif menu == "Project Documentation":
    st.title("Project Overview")
    st.markdown("""
    ### Problem Statement
    Modern networks face sophisticated attacks (DDoS, Brute Force) that bypass traditional signature-based systems. AI-Based NIDS leverages machine learning to identify anomalous behavior in real-time.

    ### Objectives
    - Implement a scalable NIDS using Random Forest.
    - Visualize network traffic distributions.
    - Provide a real-time monitor for security administrators.

    ### End Users
    - **Network Admins:** Monitor infrastructure health.
    - **Security Analysts:** Deep dive into threat patterns.
    - **Small Businesses:** Affordable cybersecurity solution.
    """)
