# 🩺 SafeSchool – AI-Powered Wearable for Student Safety & Stress Detection

**SafeSchool** is an IoT + AI wearable system designed to monitor **physiological and emotional well-being** of students in real time using multi-sensor data.  
The system leverages **PPG (MAX30102)**, **GSR**, **Accelerometer (MPU6050)**, **Temperature**, and **GPS** sensors — processed through **Edge + Cloud AI layers** — to detect stress, panic, or unsafe events and alert parents/teachers instantly.

> _“Because safety is more than CCTV — it’s emotional care.”_

---

## 🧩 System Overview

**Data Flow:**

Sensors → ESP32 Firmware → Wi-Fi → Cloud API → AI Analytics → Dashboard

markdown


| Layer | Technology | Function |
|:------|:------------|:----------|
| **Edge** | ESP32 (C/C++) | Collects sensor data, applies rule-based logic, sends data to cloud |
| **Cloud** | Flask / Express.js + PostgreSQL | Receives sensor data, manages alerts & storage |
| **AI Layer** | Python + scikit-learn | Adaptive stress detection and personalization |
| **Dashboard** | React.js + Chart.js + Leaflet | Real-time visualization and route tracking |
| **Deployment** | Render / Netlify / AWS | Cloud hosting and dashboard access |

---

## 🧠 Multi-Layer AI Architecture

The SafeSchool AI system integrates **three intelligence layers**:

1. **🧩 Rule-Based Edge AI (Threshold Model)**  
   - Runs on the ESP32 microcontroller.  
   - Uses static physiological thresholds:  
     ```text
     HR > 100, HRV < 25, GSR > 0.35 → Stress Alert
     ```
   - Enables instant detection without internet dependency.

2. **🧠 Adaptive ML Model (Cloud)**  
   - Learns baseline patterns for each child using daily sensor data.  
   - Implements supervised learning (Random Forest) on features:  
     - Heart Rate, HRV, GSR, Motion Variance  
   - Adjusts thresholds dynamically:  
     ```
     Adaptive HR = baseline_HR + 15
     Adaptive HRV = baseline_HRV * 0.7
     ```
   - Produces personalized stress index & daily trends.

3. **💬 LLM Context Layer (Planned)**  
   - Future enhancement: integrates a language model that interprets stress trends and behavioral context to recommend interventions.  
   - Example: “Student’s stress spikes before exams or after bus delays.”

Final decision = Weighted combination of all 3 models → **composite stress score**.

---

## ⚙️ Features

- 🧭 **Real-time monitoring** – physiological + motion signals
- 📶 **Cloud connectivity** – Wi-Fi data sync and dashboards
- 🧠 **Adaptive AI** – learns per-user baselines
- 🚨 **SOS & Fall detection** – immediate alerts via hardware buttons
- 📈 **Visualization dashboard** – live tracking, stress trends, and alert logs
- 🔐 **Privacy-first design** – no cameras or audio; only physiological data

---

## 🧪 AI Demonstration (Prototype Notebook)

File: [`analytics/stress_model.ipynb`](./analytics/stress_model.ipynb)

This notebook simulates physiological data and demonstrates both **rule-based** and **machine-learning** stress detection.

### Example Output
Sample rule-based predictions:
heart_rate hrv gsr rule_pred
0 92.4 47.5 0.31 Normal
...
Model Accuracy: 1.0
Adaptive thresholds per child: {'hr': 95, 'hrv': 31.5}




### Feature Importance
![Feature Importance](https://github.com/Abhii9180/SafeSchool_Wearable_AI/assets/feature_importance.png)

### To Run
```
cd analytics
jupyter notebook stress_model.ipynb
Required Python libraries:



pip install numpy pandas scikit-learn matplotlib
🧩 Hardware Stack
Component	Function
ESP32	Central MCU (Wi-Fi + BLE)
MAX30102	PPG Heart Rate & HRV sensor
GSR Sensor	Skin conductance → emotional arousal
MPU6050	6-axis accelerometer + gyroscope for motion/fall detection
LM35	Skin temperature sensor
GPS (NEO-6M)	Real-time location tracking
SOS Button + Vibration Motor	Manual alert + haptic feedback
Battery + TP4056	Power and safe charging

💰 Prototype Cost: ₹2,800–₹3,500 (scalable to ₹2,500 in bulk)



🧮 Cloud Backend (Overview)
API Framework: Node.js (Express) / Python (Flask)

Database: PostgreSQL (TimescaleDB)

Authentication: JWT tokens

Notifications: Twilio (SMS), SendGrid (Email), Firebase Push

Endpoints Example:

/api/v1/ingest → Receives sensor data

/api/v1/alerts → Fetches alert history

/api/v1/dashboard → Serves chart data



📊 Dashboard (Frontend)
Built using React.js, with:

Real-time stress indicator 🟢🟠🔴

Charts (Chart.js / Recharts)

Live map (Leaflet.js / Mapbox)

Alert logs with timestamps & location



🔐 Security & Privacy
Layer	Protection
Device	Token-authenticated HTTPS data transfer
Cloud	Rate-limiting, CORS, encrypted storage
Database	Anonymized IDs, no personal identifiers
Frontend	JWT stored in HttpOnly cookies



📈 Future Enhancements
🧠 On-device TinyML for offline classification

🌐 NB-IoT / LTE-M module for 24×7 connectivity

🩸 Integration with health tracking (heart disease, hypertension)

🤖 LLM-powered emotional insight engine

📱 React Native mobile app for parents



🧩 Folder Structure
SafeSchool_Wearable_AI/
├── README.md
├── analytics/
│   └── stress_model.ipynb
├── docs/
│   ├── hardware_overview.pdf
│   ├── software_architecture.pdf
│   ├── presentation_pitch.pdf
│   └── architecture_diagram.png





🧑‍💻 Author
Abhishek Kumar
B.Tech CSE, IIIT Guwahati (2026)
📧 ak2458ak@gmail.com | 💼 LinkedIn | 🧠 AI & Embedded Systems Enthusiast


### 🏆 Recognition & Achievements
- Selected among Top 30 teams at ThinkQbation Innovation Mela 2025 for innovative wearable AI project promoting student safety.

---

### 🪪 License
Released under the MIT License — free to use, modify, and share for academic or research purposes.


🏁 Summary
SafeSchool demonstrates how AI and embedded systems can be used responsibly to safeguard children’s emotional and physical well-being using non-invasive wearable technology.
This project bridges IoT, machine learning, and cloud computing — aligning directly with the future of digital health and AI-driven chronic disease management.

