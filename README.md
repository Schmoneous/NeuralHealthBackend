# Neural Health - Backend System

## Overview

The Neural Health backend is a dual-layer system combining Node.js REST API server with Python-based machine learning processing. Running on Raspberry Pi 5, it handles EEG data ingestion, storage, real-time seizure detection, frequency analysis, and secure multi-user access.

---

## Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND SYSTEM                             │
│                   (Raspberry Pi 5)                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Node.js REST API Server                 │    │
│  │                  (index.js)                          │    │
│  │                                                       │    │
│  │  • Express.js web framework                          │    │
│  │  • Firebase Admin authentication                     │    │
│  │  • Role-based access control                         │    │
│  │  • JSON request/response handling                    │    │
│  │  • CORS enabled for mobile clients                   │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                        │
│                      ▼                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SQLite Database (eeg1.db)               │    │
│  │                                                       │    │
│  │  Tables:                                              │    │
│  │  • uploads (metadata)                                 │    │
│  │  • eeg_samples (raw 250Hz data)                       │    │
│  │  • aggregated_data_1hz (mobile-optimized)            │    │
│  │  • fft_data (frequency analysis)                      │    │
│  │  • seizure_predictions (ML results)                   │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                        │
│                      ▼                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │        Python ML Processing Pipeline                 │    │
│  │          (eeg_processor_with_ai.py)                  │    │
│  │                                                       │    │
│  │  • Signal filtering (scipy)                          │    │
│  │  • Seizure detection (Random Forest)                 │    │
│  │  • FFT analysis (numpy)                              │    │
│  │  • Data aggregation                                  │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                        │
│                      ▼                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Database Viewer Tool                        │    │
│  │             (view_eeg.py)                            │    │
│  │                                                       │    │
│  │  • Interactive inspection                            │    │
│  │  • Seizure visualization                             │    │
│  │  • Data validation                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## System Requirements

**Hardware:**
- Raspberry Pi 5 (4GB+ RAM recommended)
- 32GB+ microSD card
- Stable power supply (5V/5A USB-C)
- Network connectivity (Ethernet or Wi-Fi)

**Software:**
- Raspberry Pi OS (64-bit, Debian-based)
- Node.js 18+ with npm
- Python 3.9+ with pip
- SQLite 3 (included with OS)

**Python Packages:**
```bash
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
joblib>=1.2.0
```

**Node.js Packages:**
```bash
express>=4.18.0
sqlite3>=5.1.0
cors>=2.8.0
firebase-admin>=11.0.0
```

---

## File Structure
```
backend/
├── index.js                      # REST API server (Node.js)
├── eeg_processor_with_ai.py     # ML processing pipeline (Python)
├── view_eeg.py                  # Database inspection tool (Python)
├── eeg1.db                      # SQLite database (auto-created)
├── package.json                 # Node.js dependencies
├── requirements.txt             # Python dependencies
├── ML_Models/
│   ├── seizure_model_rf.joblib  # Trained Random Forest model
│   ├── seizure_scaler.joblib    # Feature scaler
│   └── model_metadata.joblib    # Model configuration
└── firebase-key.json            # Firebase service account (DO NOT COMMIT)
```

---

## Core Files

### 1. index.js - Node.js REST API Server

**Purpose:** Central hub for data management, authentication, and mobile app communication.

**Size:** ~600 lines  
**Port:** 3000 (configurable via PORT environment variable)  
**Language:** JavaScript (Node.js)

#### Key Responsibilities:

**1. Database Initialization**
- Creates all tables on first run
- Sets up indexes for query performance
- Handles schema migrations gracefully

**Tables Created:**
```javascript
// uploads - Recording metadata
CREATE TABLE uploads (
    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_uid TEXT NOT NULL,
    received_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    channel_count INTEGER DEFAULT 8,
    source TEXT DEFAULT 'mobile_app',
    is_ai_model_data INTEGER DEFAULT 0
)

// eeg_samples - Raw 250Hz data
CREATE TABLE eeg_samples (
    upload_id INTEGER NOT NULL,
    t_ms INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    ch1_uV REAL, ch2_uV REAL, ..., ch8_uV REAL,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
)

// aggregated_data_1hz - Mobile-optimized
CREATE TABLE aggregated_data_1hz (
    upload_id INTEGER NOT NULL,
    t_ms INTEGER NOT NULL,
    avg_ch1 REAL, ..., avg_ch8 REAL,
    std_ch1 REAL, ..., std_ch8 REAL,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
)

// fft_data - Frequency analysis
CREATE TABLE fft_data (
    upload_id INTEGER NOT NULL,
    t_start_ms INTEGER NOT NULL,
    t_end_ms INTEGER NOT NULL,
    channel INTEGER NOT NULL,
    delta REAL, theta REAL, alpha REAL, beta REAL, gamma REAL,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
)

// seizure_predictions - ML results
CREATE TABLE seizure_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    t_ms INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    is_seizure INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
)
```

**2. Authentication Middleware**
```javascript
async function requireAuth(req, res, next) {
    // Extract Bearer token from header
    const authHeader = req.headers.authorization;
    const token = authHeader.split('Bearer ')[1];
    
    // Verify Firebase ID token
    const decodedToken = await admin.auth().verifyIdToken(token);
    req.user = decodedToken;  // Attach user info to request
    
    next();  // Proceed to route handler
}
```
- Verifies every request has valid Firebase token
- Extracts user UID from token
- Attaches user info to request object
- Rejects unauthenticated requests with 401

**3. Role-Based Access Control**
```javascript
async function requireDoctor(req, res, next) {
    // Query Firestore for user's role
    const userDoc = await admin.firestore()
        .collection('users')
        .doc(req.user.uid)
        .get();
    
    const userData = userDoc.data();
    if (userData.role === 'doctor') {
        req.isDoctor = true;
        next();
    } else {
        res.status(403).json({ error: 'Doctor role required' });
    }
}
```
- Checks Firestore for user role
- Restricts doctor-only endpoints
- Validates patient-doctor relationships

#### API Endpoints:

**Health Check:**
```
GET /healthz

Response: { "ok": true, "timestamp": "2024-12-17T10:30:00Z" }
```
- Used by mobile app to verify server connectivity
- No authentication required
- Returns 200 if server is running

**Data Ingestion:**
```
POST /api/data
Authorization: Bearer <firebase_token>
Content-Type: application/json

Body:
{
  "csvData": "t_ms,seq,ch1_uV,ch2_uV,...\n0,0,123.4,234.5,...",
  "numChannels": 8,
  "source": "mobile_app"
}

Response:
{
  "message": "Data ingested successfully",
  "upload_id": 123,
  "samples_inserted": 5000
}
```
- Accepts CSV or UART formatted data
- Parses and validates channel counts
- Uses transactions for atomic inserts
- Returns upload_id for tracking

**Get User Uploads:**
```
GET /uploads?limit=20
Authorization: Bearer <firebase_token>

Response: [
  {
    "upload_id": 123,
    "user_uid": "firebase_uid_here",
    "received_at": "2024-12-17T10:00:00.000Z",
    "row_count": 5000,
    "channel_count": 8,
    "source": "mobile_app"
  }
]
```
- Returns uploads for authenticated user only
- Sorted by most recent first
- Configurable result limit

**Get Raw Samples:**
```
GET /uploads/:id/samples?limit=5000&t_start=0&t_end=10000
Authorization: Bearer <firebase_token>

Response: [
  {
    "t_ms": 0,
    "seq": 0,
    "ch1_uV": 123.4,
    "ch2_uV": 234.5,
    ...,
    "ch8_uV": 890.1
  }
]
```
- Returns raw 250Hz EEG samples
- Optional time range filtering
- Ownership verification (user or assigned doctor)

**Get Aggregated Data:**
```
GET /mobile/aggregated/:id?limit=3600
Authorization: Bearer <firebase_token>

Response: [
  {
    "upload_id": 123,
    "t_ms": 0,
    "avg_ch1": 125.3, "std_ch1": 15.2,
    ...,
    "avg_ch8": 200.1, "std_ch8": 20.5
  }
]
```
- Returns 1Hz aggregated data
- Much smaller than raw samples (250x compression)
- Used for mobile overview graphs

**Get FFT Data:**
```
GET /mobile/fft/:id?channel=1&limit=1000
Authorization: Bearer <firebase_token>

Response:
{
  "upload_id": 123,
  "windows": [
    {
      "t_start_ms": 0,
      "t_end_ms": 1000,
      "channel": 1,
      "bands": {
        "delta": 12.5,
        "theta": 8.3,
        "alpha": 15.7,
        "beta": 6.2,
        "gamma": 3.1
      }
    }
  ],
  "window_duration": "1 second",
  "bands": ["delta", "theta", "alpha", "beta", "gamma"]
}
```
- Returns frequency analysis per channel
- 1-second windows
- 5 clinically relevant bands

**Get Seizure Summary:**
```
GET /mobile/seizures/:id/summary
Authorization: Bearer <firebase_token>

Response:
{
  "upload_id": 123,
  "processed": true,
  "summary": {
    "total_samples": 5000,
    "seizure_detections": 45,
    "high_confidence_seizures": 12,
    "seizure_rate_percent": 0.9,
    "avg_probability": 0.42,
    "max_probability": 0.87,
    "threshold": 0.35,
    "duration_seconds": 20.0
  },
  "episodes": [
    {
      "start_ms": 5000,
      "end_ms": 6500,
      "duration_ms": 1500,
      "sample_count": 375,
      "avg_probability": 0.65,
      "max_probability": 0.85
    }
  ],
  "episode_count": 3
}
```
- Comprehensive seizure analysis
- Episode detection (grouped continuous events)
- Statistics for decision-making
- Powers mobile UI seizure banners

**Doctor: Get Patient Uploads:**
```
GET /doctor/patients/:patientId/uploads?limit=100
Authorization: Bearer <firebase_token>

Response: [/* array of uploads */]
```
- Doctor-only endpoint
- Verifies patient-doctor relationship in Firestore
- Returns all patient recordings
- Used in doctor dashboard

#### Performance Optimizations:

**Database Indexes:**
```javascript
CREATE INDEX idx_eeg_upload ON eeg_samples(upload_id);
CREATE INDEX idx_eeg_time ON eeg_samples(t_ms);
CREATE INDEX idx_seizure_upload ON seizure_predictions(upload_id);
CREATE INDEX idx_seizure_flag ON seizure_predictions(is_seizure);
```
- Speeds up queries by 10-100x
- Essential for large datasets
- Minimal storage overhead

**Transaction Batching:**
```javascript
db.run('BEGIN TRANSACTION');
// Insert thousands of samples
db.run('COMMIT');
```
- Groups inserts for atomicity
- Dramatically faster than individual inserts
- All-or-nothing for data integrity

**Connection Pooling:**
```javascript
const db = new sqlite3.Database('./eeg1.db');
// Single connection reused throughout application
```
- Avoids connection overhead
- SQLite works well with single connection
- Thread-safe for read operations

---

### 2. eeg_processor_with_ai.py - ML Processing Pipeline

**Purpose:** Transforms raw EEG data into actionable medical insights through signal processing and machine learning.

**Size:** ~500 lines  
**Language:** Python 3.9+  
**Dependencies:** numpy, scipy, scikit-learn, joblib

#### Processing Pipeline:

**Phase 1: Signal Filtering**

**Bandpass Filter (0.5-70 Hz):**
```python
sos_bandpass = signal.butter(
    4,  # 4th order filter
    [0.5, 70],  # Passband frequencies
    btype='bandpass',
    fs=250,  # Sampling rate
    output='sos'  # Second-order sections for stability
)
```
- Removes DC offset and high-frequency noise
- Preserves all brain wave bands (delta through gamma)
- Butterworth design for flat passband
- 4th order provides good roll-off

**Notch Filter (60 Hz):**
```python
b_notch, a_notch = signal.iirnotch(
    60,  # US power line frequency
    30,  # Quality factor
    250  # Sampling rate
)
```
- Eliminates power line interference
- Essential for clean signals
- Narrow notch preserves nearby frequencies

**Application:**
```python
filtered = signal.sosfilt(sos_bandpass, raw_data)
filtered = signal.filtfilt(b_notch, a_notch, filtered)
```
- Zero-phase filtering (filtfilt) prevents time shifts
- Applied to all 8 channels independently
- Results stored back in original arrays

**Phase 2: Seizure Detection**

**Model Loading:**
```python
MODEL = joblib.load('seizure_model_rf.joblib')
SCALER = joblib.load('seizure_scaler.joblib')
METADATA = joblib.load('model_metadata.joblib')
THRESHOLD = METADATA.get('threshold', 0.35)
```
- Trained Random Forest Classifier
- StandardScaler for feature normalization
- Configurable threshold (default 0.35)

**Feature Engineering:**
```python
def extract_features_for_ml(channels):
    ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8 = channels
    
    # Original 8 channels
    original = list(channels)
    
    # Channel differences (7 features)
    diffs = [ch2-ch1, ch3-ch2, ch4-ch3, ch5-ch4, ch6-ch5, ch7-ch6, ch8-ch7]
    
    # Channel ratios (4 features)
    ratios = [ch1/ch2, ch3/ch4, ch5/ch6, ch7/ch8]
    
    # Squared channels (8 features)
    squared = [ch**2 for ch in channels]
    
    # Total: 8 + 7 + 4 + 8 = 27 features
    return np.array(original + diffs + ratios + squared)
```
- Captures spatial relationships between channels
- Ratios detect amplitude imbalances
- Squared values amplify large deviations
- Optimized for seizure detection

**Batch Prediction:**
```python
def predict_seizure_batch(samples_array):
    # Extract features for all samples
    features = np.array([extract_features_for_ml(sample) 
                         for sample in samples_array])
    
    # Scale features
    features_scaled = SCALER.transform(features)
    
    # Predict probabilities
    probabilities = MODEL.predict_proba(features_scaled)[:, 1]
    
    # Apply threshold
    predictions = (probabilities >= THRESHOLD).astype(int)
    
    return list(zip(predictions, probabilities))
```
- Processes all samples at once (vectorized)
- Much faster than per-sample prediction (~1000 samples/second)
- Returns (prediction, probability) tuples

**Database Storage:**
```python
cursor.executemany('''
    INSERT INTO seizure_predictions 
    (upload_id, t_ms, seq, prediction, probability, threshold, is_seizure)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', prediction_data)
```
- Stores every sample's prediction
- Includes probability for post-processing
- Threshold recorded for reproducibility

**Phase 3: FFT Analysis**

**Window-Based FFT:**
```python
window_size = 250  # 1 second at 250 Hz
n_windows = len(samples) // window_size

for window_idx in range(n_windows):
    start_idx = window_idx * window_size
    end_idx = start_idx + window_size
    window_data = channel_data[start_idx:end_idx]
    
    band_powers = compute_fft_bands(window_data)
    # Store results...
```
- 1-second temporal resolution
- Balances frequency resolution with time localization
- Processes each channel independently

**Band Power Calculation:**
```python
def compute_fft_bands(data, fs=250):
    # Compute FFT
    n = len(data)
    yf = rfft(data)  # Real FFT (positive frequencies only)
    xf = rfftfreq(n, 1/fs)  # Frequency bins
    
    # Power spectral density
    psd = np.abs(yf) ** 2
    
    # Integrate power in each band
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        mask = (xf >= low) & (xf < high)
        band_powers[band_name] = np.sum(psd[mask])
    
    return band_powers
```
- Uses real FFT (rfft) for efficiency
- Integrates power across frequency ranges
- Returns absolute power values (μV²)

**Clinical Significance of Bands:**
- **Delta (0.5-4 Hz):** Deep sleep, brain injuries
- **Theta (4-8 Hz):** Drowsiness, emotional processing
- **Alpha (8-13 Hz):** Relaxed wakefulness, closed eyes
- **Beta (13-30 Hz):** Active thinking, focus, anxiety
- **Gamma (30-70 Hz):** Cognitive processing, binding

**Phase 4: Data Aggregation**

**1Hz Compression:**
```python
for window_idx in range(n_windows):
    start_idx = window_idx * window_size
    end_idx = start_idx + window_size
    
    # Compute statistics for each channel
    window_stats = []
    for channel_data in filtered_channels:
        window_data = channel_data[start_idx:end_idx]
        avg = np.mean(window_data)
        std = np.std(window_data)
        window_stats.extend([avg, std])
    
    aggregated_records.append((upload_id, t_window, *window_stats))
```
- Reduces from 250 Hz to 1 Hz (250x compression)
- Preserves mean and standard deviation
- Enables efficient mobile data transmission

**Storage:**
```python
cursor.executemany('''
    INSERT INTO aggregated_data_1hz 
    (upload_id, t_ms, 
     avg_ch1, ..., avg_ch8, std_ch1, ..., std_ch8)
    VALUES (?, ?, ?, ..., ?)
''', aggregated_records)
```
- One row per second
- 16 values per row (8 means + 8 std devs)
- Used by mobile app for overview graphs

#### Operating Modes:

**1. Process All Pending:**
```bash
python3 eeg_processor_with_ai.py
```
- Finds uploads without processed data
- Processes each upload once
- Exits when complete
- Use after initial setup

**2. Monitor Mode:**
```bash
python3 eeg_processor_with_ai.py monitor
```
- Continuously watches for new uploads
- Checks every 5 seconds
- Auto-processes new data
- Run as background service
- Ctrl+C to stop

**3. Process Specific Upload:**
```bash
python3 eeg_processor_with_ai.py 123
```
- Processes only upload_id 123
- Useful for reprocessing
- Deletes old processed data first

#### Performance Metrics:

- **Filtering**: ~0.1ms per channel per sample
- **Feature Extraction**: ~0.01ms per sample
- **ML Inference**: ~0.001ms per sample (batched)
- **FFT**: ~1ms per window per channel
- **Total**: ~5-10 seconds for 5000 samples (20 seconds of EEG)

#### Error Handling:
```python
try:
    process_upload(upload_id)
except Exception as e:
    print(f"❌ Error processing upload {upload_id}: {e}")
    import traceback
    traceback.print_exc()
    # Continue with next upload (don't crash)
```
- Catches all exceptions
- Prints detailed traceback
- Continues processing other uploads
- Logs errors for debugging

---

### 3. view_eeg.py - Database Inspection Tool

**Purpose:** Interactive command-line tool for exploring, validating, and visualizing database contents.

**Size:** ~400 lines  
**Language:** Python 3  
**Interface:** Menu-driven CLI

#### Features:

**1. List All Uploads**
```
=================================================================================================
ID     User UID                       Received At          Samples  Agg    FFT    Seizure       
=================================================================================================
123    firebase_uid_12345             2024-12-17 10:00:00  5000     20     160    45/5000      
122    firebase_uid_67890             2024-12-17 09:45:00  7500     30     240    12/7500      
=================================================================================================
Total uploads: 2
```
- Shows all recordings with processing status
- Indicates which stages are complete
- Seizure detection shown as detections/total

**2. Upload Summary**
```
============================================================
Upload ID: 123
============================================================
User UID: firebase_uid_12345
Received: 2024-12-17T10:00:00.000Z
Samples: 5000
Channels: 8
Source: mobile_app

EEG Samples in DB: 5000
Aggregated Windows: 20
FFT Windows: 160
Seizure Predictions: 5000
  Detected: 45 (0.9%)
  Avg Probability: 0.42
  Max Probability: 0.87
============================================================
```
- Detailed stats for specific upload
- Verifies data integrity
- Shows processing completeness

**3. View Raw Samples**
```
========================================================================================================================
EEG Samples (first 10) for Upload 123
========================================================================================================================
Time (ms)    Seq    CH1      CH2      CH3      CH4      CH5      CH6      CH7      CH8     
========================================================================================================================
0            0      123.45   234.56   345.67   456.78   567.89   678.90   789.01   890.12  
4            1      124.12   235.23   346.34   457.45   568.56   679.67   790.78   891.89  
========================================================================================================================
```
- Inspect raw 250Hz data
- All 8 channels visible
- Configurable row limit

**4. View Aggregated Data**
```
========================================================================================================================
Aggregated Data (1Hz, first 10) for Upload 123
========================================================================================================================
Time (ms)    AvgCH1   AvgCH2   AvgCH3   AvgCH4   StdCH1   StdCH2  
========================================================================================================================
0            125.30   236.40   347.50   458.60   15.20    18.30   
1000         126.10   237.20   348.30   459.40   14.80    17.90   
========================================================================================================================
```
- View 1Hz compressed data
- Shows means and standard deviations
- Verifies aggregation correctness

**5. View FFT Data**
```
====================================================================================================
FFT Data (first 10) for Upload 123, Channel 1
====================================================================================================
Start (ms)   End (ms)     Delta      Theta      Alpha      Beta       Gamma     
====================================================================================================
0            1000         12.50      8.30       15.70      6.20       3.10      
1000         2000         13.20      7.90       16.10      5.80       2.90      
====================================================================================================
```
- Frequency analysis per channel
- 1-second windows
- All 5 brain wave bands

**6. View Seizure Predictions**
```
====================================================================================================
Seizure Predictions (first 20) for Upload 123
====================================================================================================
Time (ms)    Seq    Prediction   Probability  Threshold    Is Seizure
====================================================================================================
0            0      0            0.2345       0.35         NO        
4            1      0            0.1987       0.35         NO        
8            2      1            0.6789       0.35         YES       
12           3      1            0.7234       0.35         YES       
====================================================================================================
```
- Individual sample predictions
- Probability scores for all samples
- Threshold used for binary decision

**7. Seizure Detection Summary**
```
============================================================
Seizure Detection Summary - Upload 123
============================================================
Total Samples: 5000
Duration: 20.0 seconds
Seizure Detections: 45 (0.90%)
High Confidence (>70%): 12

Probability Stats:
  Average: 0.4234
  Maximum: 0.8765
  Minimum: 0.0123
  Threshold: 0.35

Seizure Episodes: 3
============================================================
Episode    Start (ms)   End (ms)     Duration     Samples   
============================================================
1          5000         6500         1.50s        375       
2          12000        12800        0.80s        200       
3          18000        19200        1.20s        300       
============================================================
```
- Comprehensive statistical analysis
- Episode detection (grouped events)
- High-confidence seizure count
- Used for medical decision-making

**8. Compare Data Types**
```
============================================================
Data Comparison for Upload 123
============================================================
Expected (from upload): 5000
EEG Samples: 5000 ✅
Aggregated (1Hz): 20 (expected ~20)
FFT Windows: 160 (expected ~20)
Seizure Predictions: 5000 ✅
  Seizures Detected: 45 (0.9%)
============================================================
```
- Validates processing pipeline
- Identifies missing or incomplete data
- Quick health check

**9. Visualize Seizure Timeline**
```
================================================================================
Seizure Detection Timeline - Upload 123
================================================================================
Total Samples: 5000
Bins: 5 (each bin = 1000 samples)

Bin   1 [    0-  1000] ████████████████████                   150/ 1000 (0.456)
Bin   2 [ 1000-  2000] ██████                                  45/ 1000 (0.234)
Bin   3 [ 2000-  3000] ███                                     20/ 1000 (0.189)
Bin   4 [ 3000-  4000] █████████████████████████              200/ 1000 (0.567)
Bin   5 [ 4000-  5000] ████████                                60/ 1000 (0.312)
================================================================================
```
- ASCII bar chart showing temporal distribution
- Each bar shows seizure density
- Average probability per bin
- Quick visual pattern recognition

#### Use Cases:

**Quality Assurance:**
- Verify all uploads processed successfully
- Check for missing data in pipeline
- Validate seizure detection ran

**Debugging:**
- Inspect raw data for sensor issues
- Verify filter outputs look reasonable
- Check ML model predictions

**Medical Review:**
- Generate seizure reports for doctors
- Identify concerning patterns
- Extract specific episodes for analysis

**Development:**
- Test new features before deploying
- Validate database schema changes
- Benchmark query performance

---

## Data Flow Diagram
```
1. EEG Sensors (Raspberry Pi 3B+)
          ↓
2. Network Transfer (Wi-Fi/Ethernet)
          ↓
3. Node.js Server (index.js) - POST /api/data
          ↓
4. SQLite Database - INSERT INTO eeg_samples
          ↓
5. Python Processor (eeg_processor_with_ai.py) detects new upload
          ↓
6. Signal Filtering (scipy)
          ↓
7. Seizure Detection (Random Forest)
          ↓
8. FFT Analysis (numpy)
          ↓
9. Data Aggregation (1Hz)
          ↓
10. SQLite Database - UPDATE all tables
          ↓
11. Mobile App - GET /mobile/seizures/:id/summary
          ↓
12. Display in Patient/Doctor UI
```

---

## Deployment

### Initial Setup:
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# 3. Install Python packages
pip3 install numpy scipy scikit-learn joblib

# 4. Clone repository
git clone https://github.com/yourusername/neural-health.git
cd neural-health

# 5. Install Node dependencies
npm install

# 6. Set environment variables
export FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/firebase-key.json
export PORT=3000

# 7. Start Node.js server
node index.js

# 8. In new terminal, start Python processor
python3 eeg_processor_with_ai.py monitor
```

### Running as Services:

**Node.js Service (systemd):**
```bash
# Create service file
sudo nano /etc/systemd/system/neural-health.service

[Unit]
Description=Neural Health API Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/neural-health
Environment="FIREBASE_SERVICE_ACCOUNT_PATH=/home/pi/firebase-key.json"
Environment="PORT=3000"
ExecStart=/usr/bin/node index.js
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable neural-health
sudo systemctl start neural-health
sudo systemctl status neural-health
```

**Python Processor Service:**
```bash
# Create service file
sudo nano /etc/systemd/system/eeg-processor.service

[Unit]
Description=EEG ML Processor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/neural-health
ExecStart=/usr/bin/python3 eeg_processor_with_ai.py monitor
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable eeg-processor
sudo systemctl start eeg-processor
sudo systemctl status eeg-processor
```

### Monitoring:
```bash
# Check server logs
sudo journalctl -u neural-health -f

# Check processor logs
sudo journalctl -u eeg-processor -f

# Check system resources
htop
df -h  # Disk space
free -h  # Memory
```

---

## Troubleshooting

### Common Issues:

**"Database is locked"**
- SQLite doesn't handle concurrent writes well
- Solution: Ensure only one Python process runs
- Use `monitor` mode instead of multiple instances

**"Model files not found"**
- Check ML model files exist in working directory
- Verify file permissions (readable by pi user)
- Processor will warn but continue without seizure detection

**"Firebase authentication failed"**
- Verify firebase-key.json path is correct
- Check file permissions (readable by node process)
- Ensure service account has proper roles in Firebase

**"Memory exhausted"**
- Large uploads (>100k samples) can use significant RAM
- Solution: Process in smaller batches
- Monitor with `free -h` and `htop`

**"No uploads being processed"**
- Check Python processor is running: `ps aux | grep eeg_processor`
- Verify database has pending uploads: `python3 view_eeg.py`
- Check logs for errors: `sudo journalctl -u eeg-processor -f`

---

## Performance Tuning

### Database Optimizations:
```sql
-- WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Increase cache size (pages * 4KB)
PRAGMA cache_size=-64000;  -- 64MB cache

-- Disable synchronous writes (faster, slightly less safe)
PRAGMA synchronous=NORMAL;
```

### Python Optimizations:
```python
# Process larger batches (use more RAM for speed)
BATCH_SIZE = 10000  # Instead of processing one upload at a time

# Use numpy vectorization
filtered = np.apply_along_axis(apply_filters, axis=0, arr=raw_data)
```

### Node.js Optimizations:
```javascript
// Increase JSON limit for large uploads
app.use(express.json({ limit: '100mb' }));

// Enable compression
const compression = require('compression');
app.use(compression());
```

---

## Security Considerations

**Firebase Token Verification:**
- Every request verified server-side
- Tokens expire after 1 hour
- Mobile app automatically refreshes

**Role-Based Access:**
- Patients can only access own data
- Doctors can access assigned patients only
- Enforced at both API and Firestore levels

**SQL Injection Prevention:**
- All queries use parameterized statements
- No string concatenation for SQL
- SQLite binding prevents injection

**Sensitive Data:**
- firebase-key.json must be secured (600 permissions)
- Never commit to git (use .gitignore)
- Rotate keys periodically

---

## Backup Strategy

**Database Backup:**
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
cp eeg1.db "backups/eeg1_$DATE.db"

# Keep only last 30 days
find backups/ -name "eeg1_*.db" -mtime +30 -delete
```

**Model Backup:**
```bash
# Backup ML models (important!)
cp seizure_model_rf.joblib backups/
cp seizure_scaler.joblib backups/
cp model_metadata.joblib backups/
```

**Automated Backups:**
```bash
# Add to crontab (daily at 2 AM)
crontab -e

0 2 * * * /home/pi/neural-health/backup.sh
```

---

## Testing

**API Endpoint Tests:**
```bash
# Health check
curl http://localhost:3000/healthz

# Get uploads (requires token)
curl -H "Authorization: Bearer <firebase_token>" \
     http://localhost:3000/uploads?limit=5

# Get seizure summary
curl -H "Authorization: Bearer <firebase_token>" \
     http://localhost:3000/mobile/seizures/123/summary
```

**Database Integrity Tests:**
```bash
# Check for missing data
python3 view_eeg.py
# Select option 8 (Compare data types)

# Verify seizure predictions
python3 view_eeg.py
# Select option 7 (Seizure summary)
```

**Load Testing:**
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test ingestion endpoint (100 requests, 10 concurrent)
ab -n 100 -c 10 -p sample_data.json \
   -T application/json \
   -H "Authorization: Bearer <token>" \
   http://localhost:3000/api/data
```

---

## Maintenance

**Regular Tasks:**

**Daily:**
- Check service status: `systemctl status neural-health eeg-processor`
- Monitor disk space: `df -h`
- Review error logs: `journalctl -u neural-health --since today`

**Weekly:**
- Verify backups completed successfully
- Check database size and growth rate
- Review seizure detection accuracy (false positive rate)

**Monthly:**
- Update system packages: `sudo apt update && sudo apt upgrade`
- Review and rotate logs
- Validate Firebase token expiry handling

**Annually:**
- Retrain ML model with new data
- Update Node.js and Python versions
- Review security and update firebase-key if needed

---

## API Rate Limits

**Recommended Limits:**
- Max upload size: 50MB
- Max samples per upload: 100,000
- Max uploads per user per day: 1,000
- Max requests per minute: 60

**Implementation:**
```javascript
const rateLimit = require('express-rate-limit');

const apiLimiter = rateLimit({
    windowMs: 60 * 1000,  // 1 minute
    max: 60  // 60 requests per minute
});

app.use('/api/', apiLimiter);
```

---

## Logging

**Application Logs:**
```javascript
// Node.js logging
console.log('✅ Info message');
console.error('❌ Error message');
console.warn('⚠️  Warning message');
```

**Log Rotation:**
```bash
# Install logrotate
sudo apt install logrotate

# Configure rotation
sudo nano /etc/logrotate.d/neural-health

/var/log/neural-health/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 pi pi
}
```

---

## Resources

**SQLite Documentation:**
- [Command Line Interface](https://www.sqlite.org/cli.html)
- [Performance Tips](https://www.sqlite.org/optoverview.html)

**Scipy Signal Processing:**
- [Signal Filtering](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [FFT Functions](https://docs.scipy.org/doc/scipy/reference/fft.html)

**Scikit-learn ML:**
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Model Persistence](https://scikit-learn.org/stable/model_persistence.html)

**Firebase Admin SDK:**
- [Authentication](https://firebase.google.com/docs/auth/admin)
- [Node.js Setup](https://firebase.google.com/docs/admin/setup)

---

**Maintained by:** Ivebens Eliacin  
**Last Updated:** December 2025  
**Node.js Version:** 18.x  
**Python Version:** 3.9+  
**Database:** SQLite 3
