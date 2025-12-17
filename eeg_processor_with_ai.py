#!/usr/bin/env python3
"""
EEG Data Processor with AI Seizure Detection
Processes raw EEG data, performs filtering, FFT analysis, aggregation, and seizure detection
"""

import sqlite3
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import joblib
import time
from datetime import datetime

# Database path
DB_PATH = './eeg1.db'

# Sampling rate
SAMPLING_RATE = 250  # Hz

# Filter parameters
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 70   # Hz
NOTCH_FREQ = 60      # Hz (US power line frequency)
NOTCH_Q = 30

# FFT frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}

# Load ML model for seizure detection
try:
    MODEL = joblib.load('seizure_model_rf.joblib')
    SCALER = joblib.load('seizure_scaler.joblib')
    METADATA = joblib.load('model_metadata.joblib')
    THRESHOLD = METADATA.get('threshold', 0.35)
    print(f"✅ Loaded seizure detection model (threshold: {THRESHOLD})")
except FileNotFoundError:
    print("⚠️  Seizure model files not found. Seizure detection will be disabled.")
    MODEL = None
    SCALER = None
    THRESHOLD = 0.35

def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_filter_coefficients():
    """Create filter coefficients (bandpass and notch)"""
    # Bandpass filter (Butterworth, 4th order)
    sos_bandpass = signal.butter(
        4, 
        [BANDPASS_LOW, BANDPASS_HIGH], 
        btype='bandpass', 
        fs=SAMPLING_RATE, 
        output='sos'
    )
    
    # Notch filter (60 Hz power line interference)
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, SAMPLING_RATE)
    
    return sos_bandpass, b_notch, a_notch

def apply_filters(data, sos_bandpass, b_notch, a_notch):
    """Apply bandpass and notch filters to data"""
    # Apply bandpass filter
    filtered = signal.sosfilt(sos_bandpass, data)
    # Apply notch filter
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return filtered

def compute_fft_bands(data, fs=SAMPLING_RATE):
    """Compute power in each frequency band using FFT"""
    # Compute FFT
    n = len(data)
    yf = rfft(data)
    xf = rfftfreq(n, 1/fs)
    
    # Compute power spectral density
    psd = np.abs(yf) ** 2
    
    # Calculate power in each band
    band_powers = {}
    for band_name, (low, high) in BANDS.items():
        mask = (xf >= low) & (xf < high)
        band_powers[band_name] = np.sum(psd[mask])
    
    return band_powers

def extract_features_for_ml(channels):
    """
    Extract 27 features from 8 channels for seizure detection model
    Features: 8 channels + 7 differences + 4 ratios + 8 squared = 27 features
    """
    ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8 = channels
    
    # Original 8 channels
    original = list(channels)
    
    # Channel differences (7 features)
    diffs = [
        ch2 - ch1, ch3 - ch2, ch4 - ch3, ch5 - ch4,
        ch6 - ch5, ch7 - ch6, ch8 - ch7
    ]
    
    # Channel ratios (4 features) - avoid division by zero
    ratios = [
        ch1 / ch2 if ch2 != 0 else 0,
        ch3 / ch4 if ch4 != 0 else 0,
        ch5 / ch6 if ch6 != 0 else 0,
        ch7 / ch8 if ch8 != 0 else 0
    ]
    
    # Squared channels (8 features)
    squared = [ch**2 for ch in channels]
    
    # Concatenate all features
    return np.array(original + diffs + ratios + squared)

def predict_seizure_batch(samples_array):
    """
    Predict seizure for a batch of samples
    samples_array: numpy array of shape (n_samples, 8)
    Returns: list of (prediction, probability) tuples
    """
    if MODEL is None or SCALER is None:
        return [(0, 0.0) for _ in range(len(samples_array))]
    
    # Extract features for all samples
    features = np.array([extract_features_for_ml(sample) for sample in samples_array])
    
    # Scale features
    features_scaled = SCALER.transform(features)
    
    # Predict probabilities
    probabilities = MODEL.predict_proba(features_scaled)[:, 1]  # Probability of seizure class
    
    # Make predictions based on threshold
    predictions = (probabilities >= THRESHOLD).astype(int)
    
    return list(zip(predictions, probabilities))

def process_upload(upload_id):
    """Process a single upload: filter, FFT, aggregate, and detect seizures"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"\n{'='*60}")
    print(f"Processing upload_id: {upload_id}")
    print(f"{'='*60}")
    
    # Get upload info
    cursor.execute('SELECT row_count, channel_count FROM uploads WHERE upload_id = ?', (upload_id,))
    upload_info = cursor.fetchone()
    
    if not upload_info:
        print(f"❌ Upload {upload_id} not found")
        conn.close()
        return
    
    row_count = upload_info['row_count']
    channel_count = upload_info['channel_count']
    
    print(f"📊 Samples: {row_count}, Channels: {channel_count}")
    
    # Fetch all samples
    cursor.execute('''
        SELECT t_ms, seq, ch1_uV, ch2_uV, ch3_uV, ch4_uV, ch5_uV, ch6_uV, ch7_uV, ch8_uV
        FROM eeg_samples
        WHERE upload_id = ?
        ORDER BY t_ms
    ''', (upload_id,))
    
    samples = cursor.fetchall()
    
    if len(samples) == 0:
        print(f"⚠️  No samples found for upload {upload_id}")
        conn.close()
        return
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Create filter coefficients
    sos_bandpass, b_notch, a_notch = create_filter_coefficients()
    
    # Extract channel data
    timestamps = [s['t_ms'] for s in samples]
    sequences = [s['seq'] for s in samples]
    channels = [
        np.array([s['ch1_uV'] for s in samples]),
        np.array([s['ch2_uV'] for s in samples]),
        np.array([s['ch3_uV'] for s in samples]),
        np.array([s['ch4_uV'] for s in samples]),
        np.array([s['ch5_uV'] for s in samples]),
        np.array([s['ch6_uV'] for s in samples]),
        np.array([s['ch7_uV'] for s in samples]),
        np.array([s['ch8_uV'] for s in samples])
    ]
    
    # Apply filters to each channel
    print("🔄 Applying filters...")
    filtered_channels = []
    for i, channel_data in enumerate(channels, 1):
        filtered = apply_filters(channel_data, sos_bandpass, b_notch, a_notch)
        filtered_channels.append(filtered)
    
    print("✅ Filtering complete")
    
    # SEIZURE DETECTION (on filtered data)
    if MODEL is not None:
        print("🧠 Running seizure detection...")
        
        # Prepare samples for batch prediction
        samples_array = np.column_stack(filtered_channels)  # Shape: (n_samples, 8)
        
        # Batch predict
        predictions = predict_seizure_batch(samples_array)
        
        # Insert predictions into database
        cursor.execute('DELETE FROM seizure_predictions WHERE upload_id = ?', (upload_id,))
        
        insert_query = '''
            INSERT INTO seizure_predictions 
            (upload_id, t_ms, seq, prediction, probability, threshold, is_seizure)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        
        prediction_data = [
            (upload_id, timestamps[i], sequences[i], pred, prob, THRESHOLD, pred)
            for i, (pred, prob) in enumerate(predictions)
        ]
        
        cursor.executemany(insert_query, prediction_data)
        
        seizure_count = sum(p[0] for p in predictions)
        seizure_rate = (seizure_count / len(predictions)) * 100
        
        print(f"✅ Seizure detection complete: {seizure_count}/{len(predictions)} samples ({seizure_rate:.1f}%)")
    
    # FFT ANALYSIS
    print("🔄 Computing FFT...")
    
    # Clear existing FFT data
    cursor.execute('DELETE FROM fft_data WHERE upload_id = ?', (upload_id,))
    
    # Window size: 1 second = 250 samples at 250 Hz
    window_size = SAMPLING_RATE
    n_windows = len(timestamps) // window_size
    
    fft_records = []
    
    for window_idx in range(n_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        
        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx - 1] if end_idx < len(timestamps) else timestamps[-1]
        
        # Compute FFT for each channel in this window
        for ch_idx, channel_data in enumerate(filtered_channels, 1):
            window_data = channel_data[start_idx:end_idx]
            
            if len(window_data) < window_size:
                continue
            
            band_powers = compute_fft_bands(window_data)
            
            fft_records.append((
                upload_id,
                t_start,
                t_end,
                ch_idx,
                band_powers['delta'],
                band_powers['theta'],
                band_powers['alpha'],
                band_powers['beta'],
                band_powers['gamma']
            ))
    
    # Insert FFT data
    cursor.executemany('''
        INSERT INTO fft_data 
        (upload_id, t_start_ms, t_end_ms, channel, delta, theta, alpha, beta, gamma)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', fft_records)
    
    print(f"✅ FFT analysis complete: {len(fft_records)} windows")
    
    # DATA AGGREGATION (1 Hz = every 250 samples)
    print("🔄 Aggregating data to 1 Hz...")
    
    # Clear existing aggregated data
    cursor.execute('DELETE FROM aggregated_data_1hz WHERE upload_id = ?', (upload_id,))
    
    aggregated_records = []
    
    for window_idx in range(n_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        
        t_window = timestamps[start_idx]
        
        # Compute mean and std for each channel
        window_stats = []
        for channel_data in filtered_channels:
            window_data = channel_data[start_idx:end_idx]
            avg = np.mean(window_data)
            std = np.std(window_data)
            window_stats.extend([avg, std])
        
        aggregated_records.append((upload_id, t_window, *window_stats))
    
    # Insert aggregated data
    cursor.executemany('''
        INSERT INTO aggregated_data_1hz 
        (upload_id, t_ms, 
         avg_ch1, avg_ch2, avg_ch3, avg_ch4, avg_ch5, avg_ch6, avg_ch7, avg_ch8,
         std_ch1, std_ch2, std_ch3, std_ch4, std_ch5, std_ch6, std_ch7, std_ch8)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', aggregated_records)
    
    print(f"✅ Data aggregation complete: {len(aggregated_records)} windows")
    
    # Commit all changes
    conn.commit()
    conn.close()
    
    print(f"{'='*60}")
    print(f"✅ Upload {upload_id} processed successfully")
    print(f"{'='*60}\n")

def process_all_pending():
    """Process all uploads that haven't been processed yet"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Find uploads without aggregated data (indicates not processed)
    cursor.execute('''
        SELECT u.upload_id 
        FROM uploads u
        LEFT JOIN aggregated_data_1hz a ON u.upload_id = a.upload_id
        WHERE a.upload_id IS NULL
        ORDER BY u.upload_id
    ''')
    
    pending = cursor.fetchall()
    conn.close()
    
    if len(pending) == 0:
        print("✅ No pending uploads to process")
        return
    
    print(f"📋 Found {len(pending)} pending uploads")
    
    for row in pending:
        upload_id = row['upload_id']
        try:
            process_upload(upload_id)
        except Exception as e:
            print(f"❌ Error processing upload {upload_id}: {e}")
            import traceback
            traceback.print_exc()

def monitor_and_process():
    """Continuously monitor for new uploads and process them"""
    print("🔄 Starting monitoring mode...")
    print("Press Ctrl+C to stop\n")
    
    processed_uploads = set()
    
    try:
        while True:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all upload IDs
            cursor.execute('SELECT upload_id FROM uploads ORDER BY upload_id')
            all_uploads = {row['upload_id'] for row in cursor.fetchall()}
            
            conn.close()
            
            # Find new uploads
            new_uploads = all_uploads - processed_uploads
            
            if new_uploads:
                print(f"🆕 Found {len(new_uploads)} new upload(s)")
                for upload_id in sorted(new_uploads):
                    try:
                        process_upload(upload_id)
                        processed_uploads.add(upload_id)
                    except Exception as e:
                        print(f"❌ Error processing upload {upload_id}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Sleep for 5 seconds
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")

if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("EEG Data Processor with AI Seizure Detection")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"Bandpass Filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"Notch Filter: {NOTCH_FREQ} Hz")
    print(f"Seizure Threshold: {THRESHOLD}")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'monitor':
            monitor_and_process()
        elif sys.argv[1].isdigit():
            # Process specific upload
            upload_id = int(sys.argv[1])
            process_upload(upload_id)
        else:
            print("Usage:")
            print("  python eeg_processor_with_ai.py           - Process all pending uploads")
            print("  python eeg_processor_with_ai.py monitor   - Monitor and auto-process new uploads")
            print("  python eeg_processor_with_ai.py <id>      - Process specific upload ID")
    else:
        # Process all pending
        process_all_pending()
