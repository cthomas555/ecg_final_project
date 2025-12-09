import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os

st.set_page_config(page_title="ECG Signal Analyzer", layout="wide")

st.title("ECG Signal Analyzer App")
st.markdown(
    """
This app lets you visualize ECG signals, apply a bandpass filter, detect R-peaks,  
and estimate heart rate using a simple peak detection algorithm.
"""
)

# ---------------------------
# Helper functions
# ---------------------------

def load_sample_ecg():
    """
    Try to load sample ECG from data/sample_ecg.csv.
    Expected formats:
    - Columns: 'time', 'ecg'
    - OR single column of ECG samples (time will be generated).
    If file is missing, generate a synthetic ECG-like signal.
    """
    sample_path = os.path.join("data", "sample_ecg.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        if "time" in df.columns and "ecg" in df.columns:
            time = df["time"].values
            ecg = df["ecg"].values
        else:
            # Assume first column is ECG, create synthetic time
            ecg = df.iloc[:, 0].values
            time = np.arange(len(ecg))
        return time, ecg, True  # True = loaded from file
    else:
        # Generate synthetic ECG-like waveform (~60 bpm, 10 seconds)
        fs = 250  # Hz
        duration = 10  # seconds
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # Simple synthetic ECG-ish signal (not physiologically perfect)
        heart_rate = 60 / 60.0  # beats per second
        ecg = 0.6 * np.sin(2 * np.pi * heart_rate * t)  # base waveform
        # Add a bit of noise
        ecg += 0.05 * np.random.randn(len(t))
        return t, ecg, False  # False = synthetic


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered


def detect_r_peaks(ecg_signal, fs, min_distance_sec=0.4, height_factor=0.5):
    """
    Simple R-peak detection using scipy.signal.find_peaks.
    - min_distance_sec: minimum allowed distance between peaks (sec)
    - height_factor: fraction of max amplitude used as a minimum height
    """
    distance_samples = int(min_distance_sec * fs)
    if distance_samples < 1:
        distance_samples = 1

    # Set a minimum height based on fraction of max
    min_height = height_factor * np.max(ecg_signal)

    peaks, properties = find_peaks(
        ecg_signal,
        distance=distance_samples,
        height=min_height
    )
    return peaks, properties


def compute_heart_rate(peaks, fs):
    """
    Compute instantaneous and average heart rate from R-peak indices.
    """
    if len(peaks) < 2:
        return None, None, None

    # Compute RR intervals (in seconds)
    rr_intervals = np.diff(peaks) / fs  # seconds
    # Instantaneous HR in bpm
    hr_inst = 60.0 / rr_intervals
    # Time points for HR (midpoint between successive peaks)
    hr_time = (peaks[1:] + peaks[:-1]) / 2 / fs

    hr_mean = np.mean(hr_inst)
    return hr_inst, hr_time, hr_mean


# ---------------------------
# Sidebar controls
# ---------------------------

st.sidebar.header("ECG Settings")

fs = st.sidebar.number_input(
    "Sampling rate (Hz)",
    min_value=100,
    max_value=2000,
    value=250,
    step=10,
    help="Sampling frequency of the ECG signal."
)

use_sample = st.sidebar.radio(
    "Data source",
    ("Use sample ECG", "Upload CSV"),
)

uploaded_file = None
if use_sample == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload ECG CSV file",
        type=["csv"],
        help="CSV with either a single ECG column or 'time' and 'ecg' columns."
    )

apply_filter = st.sidebar.checkbox(
    "Apply bandpass filter",
    value=True
)

if apply_filter:
    lowcut = st.sidebar.number_input(
        "Low cutoff frequency (Hz)",
        min_value=0.1,
        max_value=50.0,
        value=0.5,
        step=0.1
    )
    highcut = st.sidebar.number_input(
        "High cutoff frequency (Hz)",
        min_value=5.0,
        max_value=100.0,
        value=40.0,
        step=1.0
    )
else:
    lowcut, highcut = None, None

do_peak_detection = st.sidebar.checkbox(
    "Detect R-peaks and estimate heart rate",
    value=True
)

# ---------------------------
# Load data
# ---------------------------

if use_sample == "Use sample ECG":
    time, ecg_raw, loaded_from_file = load_sample_ecg()
    if loaded_from_file:
        st.success("Loaded sample ECG from data/sample_ecg.csv")
    else:
        st.info("No sample file found. Using a synthetic ECG signal.")
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "time" in df.columns and "ecg" in df.columns:
            time = df["time"].values
            ecg_raw = df["ecg"].values
        else:
            ecg_raw = df.iloc[:, 0].values
            time = np.arange(len(ecg_raw)) / fs
        st.success("Uploaded ECG CSV successfully.")
    else:
        st.warning("Please upload a CSV file or switch to sample ECG.")
        st.stop()

# Ensure time axis is consistent with sampling rate if time not imported:
if len(time) != len(ecg_raw):
    time = np.arange(len(ecg_raw)) / fs

# ---------------------------
# Apply optional filtering
# ---------------------------

if apply_filter:
    ecg_proc = apply_bandpass_filter(ecg_raw, fs, lowcut, highcut)
else:
    ecg_proc = ecg_raw.copy()

# ---------------------------
# Peak detection & HR
# ---------------------------

if do_peak_detection:
    peaks, properties = detect_r_peaks(ecg_proc, fs)
    hr_inst, hr_time, hr_mean = compute_heart_rate(peaks, fs)
else:
    peaks, properties, hr_inst, hr_time, hr_mean = None, None, None, None, None

# ---------------------------
# Layout: plots and results
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw ECG Signal")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(time, ecg_raw, linewidth=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title("Raw ECG")
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    st.subheader("Processed ECG Signal")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time, ecg_proc, label="Processed ECG", linewidth=1)
    if do_peak_detection and peaks is not None and len(peaks) > 0:
        ax2.plot(time[peaks], ecg_proc[peaks], "ro", label="R-peaks")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.set_title("Filtered ECG" if apply_filter else "Unfiltered ECG")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")

st.subheader("Heart Rate Analysis")

if do_peak_detection and hr_mean is not None:
    st.write(f"**Estimated Average Heart Rate:** {hr_mean:.1f} bpm")
    if hr_inst is not None and hr_time is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(hr_time, hr_inst, marker="o", linestyle="-")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Heart Rate (bpm)")
        ax3.set_title("Instantaneous Heart Rate")
        ax3.grid(True)
        st.pyplot(fig3)
else:
    if do_peak_detection:
        st.warning("Not enough peaks detected to estimate heart rate.")
    else:
        st.info("Enable R-peak detection in the sidebar to view heart rate.")
        

st.markdown(
    """
___
**Notes:**  
- This is a simplified ECG processing pipeline for educational use.  
- Peak detection performance depends on signal quality and parameter choices.
"""
)
