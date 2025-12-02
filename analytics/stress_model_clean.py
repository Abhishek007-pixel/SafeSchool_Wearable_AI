"""
SafeSchool - Full Physiological Signal Stress Detection Model
-------------------------------------------------------------
This script generates:
    - Synthetic PPG waveform
    - Motion artifacts
    - Bandpass filtering
    - Peak detection
    - HR + HRV extraction
    - Accelerometer motion variance
    - Signal Quality Index (SQI)
    - Combined dataset
    - Random Forest stress classifier
Then performs:
    - Training
    - Demo prediction

This is the FULL physiologically-correct version (NO TensorFlow).
"""

# ============================
# 1. IMPORTS
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import joblib
import random
import os


# ============================
# 2. SYNTHETIC PPG GENERATOR
# ============================
def generate_ppg(fs=100, duration=10, hr_bpm=75, motion=False):
    """
    Creates a realistic PPG signal with:
    - Heart pulse
    - Harmonics
    - Respiration
    - Noise
    - Optional motion artifact
    """
    t = np.arange(0, duration, 1/fs)

    # Pulse waveform
    f_hr = hr_bpm / 60
    ppg = 0.6*np.sin(2*np.pi*f_hr*t) + 0.15*np.sin(2*np.pi*2*f_hr*t)

    # Respiration baseline (~0.25Hz)
    ppg += 0.03*np.sin(2*np.pi*0.25*t)

    # Add noise
    ppg += 0.02*np.random.randn(len(t))

    # Motion artifact
    if motion:
        start, end = int(fs*3), int(fs*6)
        ppg[start:end] += 0.6*np.sin(2*np.pi*0.8*t[start:end])

    return t, ppg


# ============================
# 3. FILTERING
# ============================
def bandpass_filter(ppg, fs=100):
    """
    Standard PPG band: 0.5–5 Hz
    Removes motion drift + noise.
    """
    b,a = butter(3, [0.5/(fs/2), 5/(fs/2)], btype="band")
    return filtfilt(b,a,ppg)


# ============================
# 4. PEAK DETECTION + HRV
# ============================
def extract_hr_hrv(ppg_f, fs):
    peaks, _ = find_peaks(ppg_f, distance=fs*0.4)

    if len(peaks) < 3:
        return None, None, None, None

    rr = np.diff(peaks) / fs  # seconds
    hr = 60 / np.mean(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    sdnn = np.std(rr)

    return hr, rmssd, sdnn, peaks


# ============================
# 5. MOTION SIMULATION
# ============================
def generate_motion_variance(fs=100, duration=10, motion=False):
    acc = 0.02*np.random.randn(duration*fs)

    if motion:
        start, end = int(fs*3), int(fs*6)
        acc[start:end] += 3.0*np.sin(2*np.pi*0.8*np.linspace(0,1,end-start))

    return np.var(acc)


# ============================
# 6. SQI (Signal Quality)
# ============================
def compute_sqi(ppg_f, peaks, motion_var):
    sqi = 1.0

    if peaks is None or len(peaks) < 4:
        sqi -= 0.4

    k = kurtosis(ppg_f)
    if k < 1 or k > 12:
        sqi -= 0.3

    if motion_var > 0.5:
        sqi -= 0.3

    return max(0, sqi)


# ============================
# 7. SYNTHETIC FULL DATASET
# ============================
def create_dataset(n=400, fs=100, duration=10):
    rows = []
    labels = []

    for i in range(n):

        # 60% normal
        if random.random() < 0.6:
            hr_bpm = random.randint(70, 90)
            gsr = random.uniform(0.20, 0.35)
            motion_flag = False
            label = "Normal"
        else:
            hr_bpm = random.randint(95, 140)
            gsr = random.uniform(0.45, 0.65)
            motion_flag = random.random() < 0.5
            label = "Stress"

        # Generate signals
        t, ppg = generate_ppg(fs, duration, hr_bpm, motion=motion_flag)
        ppg_f = bandpass_filter(ppg, fs)
        motion_var = generate_motion_variance(fs, duration, motion=motion_flag)

        hr, rmssd, sdnn, peaks = extract_hr_hrv(ppg_f, fs)
        sqi = compute_sqi(ppg_f, peaks, motion_var)

        if hr is None:
            hr, rmssd, sdnn = 0, 0, 0

        rows.append([hr, rmssd, gsr, motion_var, sqi])
        labels.append(label)

    df = pd.DataFrame(rows, columns=["hr","rmssd","gsr","motion_var","sqi"])
    df["label"] = labels
    return df


# ============================
# 8. TRAIN MODEL
# ============================
def train_model(df):
    df = df[df["sqi"] > 0.25]  # keep good quality samples

    X = df[["hr","rmssd","gsr","motion_var"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== MODEL TRAINED ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, "stress_model.joblib")
    print("Saved model → stress_model.joblib")

    return model


# ============================
# 9. DEMO PREDICTION
# ============================
def demo(model):
    print("\n=== DEMO PREDICTIONS ===")
    demo_cases = [
        {"hr": 112, "rmssd": 8, "gsr": 0.58, "motion_var": 0.12},
        {"hr": 82,  "rmssd": 45,"gsr": 0.26, "motion_var": 0.08},
        {"hr": 140, "rmssd": 5, "gsr": 0.62, "motion_var": 0.5},
    ]

    for case in demo_cases:
        X = np.array([[case["hr"], case["rmssd"], case["gsr"], case["motion_var"]]])
        pred = model.predict(X)[0]
        print(case, "=>", pred)


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    print("Generating Full Physiological Dataset...")
    df = create_dataset()
    print(df.head())

    print("\nTraining Final Model...")
    model = train_model(df)

    demo(model)

    print("\nPipeline completed successfully!")
