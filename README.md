# AI-Based Network Intrusion Detection System (AstraGuard)

## 1. Problem Statement
Cybersecurity threats are evolving at an alarming rate. Traditional Intrusion Detection Systems (IDS) rely on signature-based detection, which fails against **Zero-Day attacks** and polymorphic malware. Small organizations often lack the budget for high-end security tools, making them vulnerable to:
- **DDoS Attacks:** Flooding networks to cause downtime.
- **Brute Force:** Unauthorized access through automated credential testing.
- **Malicious Payload:** Exfiltration of sensitive data.

### Practical Challenges for Small Organizations
Small organizations often face a "security gap" because:
- **Cost:** Enterprise-grade IDS solutions (like FireEye or Darktrace) are prohibitively expensive.
- **Skill Gap:** They lack dedicated SOC teams to manually tune hundreds of rules.
- **Legacy Systems:** Older hardware may not support modern security appliance overhead.
- **Evolving Threats:** Signature-based tools are blind to zero-day attacks.

There is a critical need for an intelligent, machine-learning-based system that can detect anomalies in real-time without constant manual rule updates.

## 2. Project Description
**Objectives:**
- To build a robust ML-based NIDS using the **Random Forest** algorithm.
- To simulate high-fidelity network traffic (Normal vs. Malicious) based on the **CIC-IDS2017** feature set.
- To provide a real-time visualization dashboard for network administrators.

**Working Principle:**
The system captures or simulates network packets, extracts key features (Duration, Src\_Bytes, Dst\_Bytes, Conn\_Count), and passes them through a trained classifier. 

**Normal vs. Malicious Traffic:**
- **Normal Traffic:** Characterized by stable durations, consistent packet sizes, and balanced source-to-destination byte ratios.
- **Malicious Traffic:** Often exhibits patterns like "Burstiness" (DDoS), repetitive small-packet attempts (Brute Force), or unusually long persistence (Malware data exfiltration).

## 3. End Users
- **Network Administrators:** For monitoring organizational infrastructure.
- **Security Analysts:** For incident response and threat hunting.
- **Educational Institutes:** As a research tool for cybersecurity students.
- **Small Businesses:** As a cost-effective alternative to enterprise IDS.

## 4. Technology Stack
- **Python:** The core scripting language.
- **Pandas/NumPy:** For data manipulation and feature engineering.
- **Scikit-Learn:** For training the Random Forest model.
- **Streamlit:** For creating the interactive web dashboard.
- **Matplotlib/Seaborn:** For traffic pattern visualization.

## 5. Technical Implementation & Installation
### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run nids_main.py
   ```

## 6. Results
The system successfully classifies traffic with high accuracy (approx. 95-98% on simulated data). 
- **Dashboard:** Displays real-time prediction logs.
- **Visuals:** Shows the distribution of attack types.
- **Performance:** Minimal latency in packet classification.

## 7. Demo & Links
- **GitHub Repository:** [https://github.com/sr-857/AI-Based-NIDS](https://github.com/sr-857/AI-Based-NIDS) (Sample Link)
- **Local Dashboard:** `http://localhost:8501`
- **Screenshots:**
  - ![Dashboard](/home/roney/.gemini/antigravity/brain/c36db1bd-0b84-4d99-b3cf-3c0a80fa4158/nids_dashboard_main_1768844989957.png)

## 8. Certificates
Achievement entries included in `docs/certificates.md`.
