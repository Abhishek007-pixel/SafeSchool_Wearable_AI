import numpy as np
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg') # Essential for Flask to render plots without a GUI
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from flask import Flask, render_template, request

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================
# We wrap this in a try/except so the app doesn't crash if the file is missing during setup
try:
    model = joblib.load("stress_model.joblib")
    print("✅ Model loaded successfully!")
except:
    print("⚠️ WARNING: stress_model.joblib not found. Please run your training script first.")
    # Create a dummy model for testing UI if file is missing
    class DummyModel:
        def predict(self, X): return ["Normal"]
    model = DummyModel()

app = Flask(__name__)

# ==========================================
# 2. SIGNAL PROCESSING ENGINE (The "Under the Hood" Logic)
# ==========================================

def generate_ppg_segment(duration=4, fs=100, hr_bpm=75, motion_level=0.0):
    """
    Simulates a PPG wave. 
    High Motion Level = More noise (simulating running/panic).
    """
    t = np.arange(0, duration, 1/fs)
    
    # Basic Heart Pulse
    f_hr = hr_bpm / 60
    # A mix of two sine waves to approximate the dicrotic notch shape
    ppg = 0.6 * np.sin(2 * np.pi * f_hr * t) + 0.15 * np.sin(2 * np.pi * 2 * f_hr * t)
    
    # Add Baseline Wander (Respiration)
    ppg += 0.05 * np.sin(2 * np.pi * 0.25 * t)
    
    # Add Random Sensor Noise
    ppg += 0.02 * np.random.randn(len(t))
    
    # Add Motion Artifacts (The key feature for SafeSchool)
    if motion_level > 0.1:
        # Random bursts of high amplitude noise
        noise = motion_level * np.random.randn(len(t)) * 0.8
        ppg += noise
        
    return t, ppg

def bandpass_filter(ppg, fs=100):
    """
    The Clinical Filter: Removes baseline wander (breathing) and high freq noise.
    """
    # 0.5Hz to 5Hz is the standard range for PPG
    b, a = butter(3, [0.5/(fs/2), 5/(fs/2)], btype="band")
    return filtfilt(b, a, ppg)

def create_clinical_plot(hr_input, motion_input):
    """
    Generates the 3-Step Pipeline Graph.
    Returns: A base64 image string to embed in HTML.
    """
    fs = 100
    # 1. Generate Raw Data
    t, raw_ppg = generate_ppg_segment(hr_bpm=hr_input, motion_level=motion_input)
    
    # 2. Apply Filter
    filtered_ppg = bandpass_filter(raw_ppg, fs)
    
    # 3. Detect Peaks (Feature Extraction)
    peaks, _ = find_peaks(filtered_ppg, distance=fs*0.4)

    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Graph 1: Raw Sensor Input
    ax1.plot(t, raw_ppg, color='#e74c3c', alpha=0.8, linewidth=1.5)
    ax1.set_title("Step 1: Raw Sensor Input (Noisy)", fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    # Highlight noise if high motion
    if motion_input > 0.2:
        ax1.text(0.5, 0.9, "⚠️ Motion Artifacts Detected", transform=ax1.transAxes, color='red', fontweight='bold')

    # Graph 2: Filtered Output
    ax2.plot(t, filtered_ppg, color='#2980b9', linewidth=1.5)
    ax2.set_title("Step 2: Bandpass Filtered (0.5Hz - 5.0Hz)", fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    # Graph 3: Feature Extraction
    ax3.plot(t, filtered_ppg, color='#34495e', alpha=0.4)
    ax3.plot(t[peaks], filtered_ppg[peaks], "x", color='#27ae60', markersize=10, markeredgewidth=3, label='R-Peaks')
    ax3.set_title(f"Step 3: Feature Extraction ({len(peaks)} Beats Detected)", fontsize=12, fontweight='bold', loc='left')
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (seconds)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Convert plot to Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url

# ==========================================
# 3. ROUTES
# ==========================================

@app.route("/")
def home():
    # Show default values on first load
    return render_template("index.html", hr=85, rmssd=45, gsr=0.3, motion=0.1)

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get Inputs
    hr = float(request.form["hr"])
    rmssd = float(request.form["rmssd"])
    gsr = float(request.form["gsr"])
    motion = float(request.form["motion"])

    # 2. ML Model Prediction
    X = np.array([[hr, rmssd, gsr, motion]])
    try:
        pred = model.predict(X)[0]
    except:
        pred = "Error"

    # 3. Generate The Visualization (The visual proof)
    plot_url = create_clinical_plot(hr, motion)

    return render_template("index.html",
                           prediction=pred,
                           hr=hr, rmssd=rmssd, gsr=gsr, motion=motion,
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True, port=5000)