# Viva Preparation: AI-Based NIDS

### Q1: Why did you choose Random Forest for this project?
**Answer:** Random Forest is an ensemble method that reduces overfitting by averaging multiple decision trees. It handles high-dimensional data well and provides feature importance metrics, making it ideal for network traffic analysis.

### Q2: What is the CIC-IDS2017 dataset?
**Answer:** It is a widely used cybersecurity dataset containing captured network traffic with various attack types (DDoS, FTP Patator, etc.). It reflects modern real-world attack scenarios.

### Q3: How do you handle real-time traffic?
**Answer:** In this demo, we use a simulation engine that generates packets mirroring real attack patterns. In a production environment, tools like `Tcpdump` or `Scapy` would capture real packets for inference.

### Q4: What is the difference between Signature-based and Anomaly-based IDS?
**Answer:** Signature-based looks for known patterns (e.g., specific strings in a file). Anomaly-based (ours) looks for deviations from "Normal" behavior, allowing it to detect new, unknown threats.

### Q5: What are the key features used for classification?
**Answer:** Packet duration, source/destination bytes, and connection counts are critical indicators of traffic health.

### Q6: How do you measure model performance?
**Answer:** We use Accuracy, Precision, Recall, and F1-Score. For unbalanced datasets (where attacks are rare), F1-score is more important than accuracy.

### Q7: What are the limitations of your system?
**Answer:** It currently relies on manual feature extraction and the performance depends on the quality of training data. It might struggle with encrypted traffic without deep packet inspection.

### Q8: What is a "False Positive" in NIDS?
**Answer:** When the system incorrectly flags legitimate traffic as an attack. High false-positive rates can overwhelm administrators.

### Q9: Can this system prevent attacks?
**Answer:** This is an IDS (Detection System). To prevent attacks, it would need to be integrated with a firewall (IPS - Intrusion Prevention System) to block malicious IPs.

### Q10: What is the role of Streamlit in this project?
**Answer:** Streamlit provides the frontend interface to visualize metrics, train the model, and view live simulation logs without needing complex web development.
