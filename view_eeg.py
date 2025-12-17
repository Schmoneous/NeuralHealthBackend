#!/usr/bin/env python3
"""
EEG Database Viewer
Interactive tool to inspect database contents
"""

import sqlite3
import sys
from datetime import datetime
import numpy as np

DB_PATH = './eeg1.db'

def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def list_uploads():
    """List all uploads"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            u.upload_id,
            u.user_uid,
            u.received_at,
            u.row_count,
            u.channel_count,
            u.source,
            COUNT(DISTINCT a.t_ms) as agg_count,
            COUNT(DISTINCT f.t_start_ms) as fft_count,
            COUNT(s.prediction_id) as seizure_count,
            SUM(CASE WHEN s.is_seizure = 1 THEN 1 ELSE 0 END) as seizure_detections
        FROM uploads u
        LEFT JOIN aggregated_data_1hz a ON u.upload_id = a.upload_id
        LEFT JOIN fft_data f ON u.upload_id = f.upload_id
        LEFT JOIN seizure_predictions s ON u.upload_id = s.upload_id
        GROUP BY u.upload_id
        ORDER BY u.upload_id DESC
    ''')
    
    uploads = cursor.fetchall()
    conn.close()
    
    if len(uploads) == 0:
        print("📭 No uploads found")
        return
    
    print(f"\n{'='*100}")
    print(f"{'ID':<6} {'User UID':<30} {'Received At':<20} {'Samples':<8} {'Agg':<6} {'FFT':<6} {'Seizure':<15}")
    print(f"{'='*100}")
    
    for u in uploads:
        seizure_status = f"{u['seizure_detections'] or 0}/{u['seizure_count'] or 0}" if u['seizure_count'] else "N/A"
        print(f"{u['upload_id']:<6} {u['user_uid'][:28]:<30} {u['received_at'][:19]:<20} "
              f"{u['row_count']:<8} {u['agg_count'] or 0:<6} {u['fft_count'] or 0:<6} {seizure_status:<15}")
    
    print(f"{'='*100}\n")
    print(f"Total uploads: {len(uploads)}")

def show_summary(upload_id):
    """Show summary for specific upload"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Upload info
    cursor.execute('SELECT * FROM uploads WHERE upload_id = ?', (upload_id,))
    upload = cursor.fetchone()
    
    if not upload:
        print(f"❌ Upload {upload_id} not found")
        conn.close()
        return
    
    print(f"\n{'='*60}")
    print(f"Upload ID: {upload['upload_id']}")
    print(f"{'='*60}")
    print(f"User UID: {upload['user_uid']}")
    print(f"Received: {upload['received_at']}")
    print(f"Samples: {upload['row_count']}")
    print(f"Channels: {upload['channel_count']}")
    print(f"Source: {upload['source']}")
    
    # Sample count
    cursor.execute('SELECT COUNT(*) as count FROM eeg_samples WHERE upload_id = ?', (upload_id,))
    sample_count = cursor.fetchone()['count']
    print(f"\nEEG Samples in DB: {sample_count}")
    
    # Aggregated count
    cursor.execute('SELECT COUNT(*) as count FROM aggregated_data_1hz WHERE upload_id = ?', (upload_id,))
    agg_count = cursor.fetchone()['count']
    print(f"Aggregated Windows: {agg_count}")
    
    # FFT count
    cursor.execute('SELECT COUNT(*) as count FROM fft_data WHERE upload_id = ?', (upload_id,))
    fft_count = cursor.fetchone()['count']
    print(f"FFT Windows: {fft_count}")
    
    # Seizure predictions
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_seizure = 1 THEN 1 ELSE 0 END) as seizures,
            AVG(probability) as avg_prob,
            MAX(probability) as max_prob
        FROM seizure_predictions 
        WHERE upload_id = ?
    ''', (upload_id,))
    
    seizure = cursor.fetchone()
    if seizure and seizure['total'] > 0:
        print(f"\nSeizure Predictions: {seizure['total']}")
        print(f"  Detected: {seizure['seizures']} ({seizure['seizures']/seizure['total']*100:.1f}%)")
        print(f"  Avg Probability: {seizure['avg_prob']:.3f}")
        print(f"  Max Probability: {seizure['max_prob']:.3f}")
    else:
        print(f"\nSeizure Predictions: Not available")
    
    print(f"{'='*60}\n")
    
    conn.close()

def view_samples(upload_id, limit=10):
    """View raw EEG samples"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f'''
        SELECT * FROM eeg_samples 
        WHERE upload_id = ?
        ORDER BY t_ms
        LIMIT {limit}
    ''', (upload_id,))
    
    samples = cursor.fetchall()
    conn.close()
    
    if len(samples) == 0:
        print(f"❌ No samples found for upload {upload_id}")
        return
    
    print(f"\n{'='*120}")
    print(f"EEG Samples (first {limit}) for Upload {upload_id}")
    print(f"{'='*120}")
    print(f"{'Time (ms)':<12} {'Seq':<6} {'CH1':<8} {'CH2':<8} {'CH3':<8} {'CH4':<8} {'CH5':<8} {'CH6':<8} {'CH7':<8} {'CH8':<8}")
    print(f"{'='*120}")
    
    for s in samples:
        print(f"{s['t_ms']:<12} {s['seq']:<6} {s['ch1_uV']:<8.2f} {s['ch2_uV']:<8.2f} {s['ch3_uV']:<8.2f} {s['ch4_uV']:<8.2f} "
              f"{s['ch5_uV']:<8.2f} {s['ch6_uV']:<8.2f} {s['ch7_uV']:<8.2f} {s['ch8_uV']:<8.2f}")
    
    print(f"{'='*120}\n")

def view_aggregated(upload_id, limit=10):
    """View aggregated data"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f'''
        SELECT * FROM aggregated_data_1hz 
        WHERE upload_id = ?
        ORDER BY t_ms
        LIMIT {limit}
    ''', (upload_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    if len(data) == 0:
        print(f"❌ No aggregated data found for upload {upload_id}")
        return
    
    print(f"\n{'='*120}")
    print(f"Aggregated Data (1Hz, first {limit}) for Upload {upload_id}")
    print(f"{'='*120}")
    print(f"{'Time (ms)':<12} {'AvgCH1':<8} {'AvgCH2':<8} {'AvgCH3':<8} {'AvgCH4':<8} {'StdCH1':<8} {'StdCH2':<8}")
    print(f"{'='*120}")
    
    for d in data:
        print(f"{d['t_ms']:<12} {d['avg_ch1']:<8.2f} {d['avg_ch2']:<8.2f} {d['avg_ch3']:<8.2f} {d['avg_ch4']:<8.2f} "
              f"{d['std_ch1']:<8.2f} {d['std_ch2']:<8.2f}")
    
    print(f"{'='*120}\n")

def view_fft(upload_id, channel=1, limit=10):
    """View FFT data for specific channel"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f'''
        SELECT * FROM fft_data 
        WHERE upload_id = ? AND channel = ?
        ORDER BY t_start_ms
        LIMIT {limit}
    ''', (upload_id, channel))
    
    fft = cursor.fetchall()
    conn.close()
    
    if len(fft) == 0:
        print(f"❌ No FFT data found for upload {upload_id}, channel {channel}")
        return
    
    print(f"\n{'='*100}")
    print(f"FFT Data (first {limit}) for Upload {upload_id}, Channel {channel}")
    print(f"{'='*100}")
    print(f"{'Start (ms)':<12} {'End (ms)':<12} {'Delta':<10} {'Theta':<10} {'Alpha':<10} {'Beta':<10} {'Gamma':<10}")
    print(f"{'='*100}")
    
    for f in fft:
        print(f"{f['t_start_ms']:<12} {f['t_end_ms']:<12} {f['delta']:<10.2f} {f['theta']:<10.2f} "
              f"{f['alpha']:<10.2f} {f['beta']:<10.2f} {f['gamma']:<10.2f}")
    
    print(f"{'='*100}\n")

def view_seizure_predictions(upload_id, limit=20):
    """View seizure predictions"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f'''
        SELECT * FROM seizure_predictions 
        WHERE upload_id = ?
        ORDER BY t_ms
        LIMIT {limit}
    ''', (upload_id,))
    
    predictions = cursor.fetchall()
    conn.close()
    
    if len(predictions) == 0:
        print(f"❌ No seizure predictions found for upload {upload_id}")
        return
    
    print(f"\n{'='*100}")
    print(f"Seizure Predictions (first {limit}) for Upload {upload_id}")
    print(f"{'='*100}")
    print(f"{'Time (ms)':<12} {'Seq':<6} {'Prediction':<12} {'Probability':<12} {'Threshold':<12} {'Is Seizure':<12}")
    print(f"{'='*100}")
    
    for p in predictions:
        print(f"{p['t_ms']:<12} {p['seq']:<6} {p['prediction']:<12} {p['probability']:<12.4f} "
              f"{p['threshold']:<12.2f} {'YES' if p['is_seizure'] else 'NO':<12}")
    
    print(f"{'='*100}\n")

def view_seizure_summary(upload_id):
    """View comprehensive seizure detection summary"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if predictions exist
    cursor.execute('''
        SELECT COUNT(*) as count FROM seizure_predictions WHERE upload_id = ?
    ''', (upload_id,))
    
    if cursor.fetchone()['count'] == 0:
        print(f"❌ No seizure predictions found for upload {upload_id}")
        conn.close()
        return
    
    # Get summary statistics
    cursor.execute('''
        SELECT 
            COUNT(*) as total_samples,
            SUM(CASE WHEN is_seizure = 1 THEN 1 ELSE 0 END) as seizure_detections,
            SUM(CASE WHEN probability > 0.7 THEN 1 ELSE 0 END) as high_confidence,
            AVG(probability) as avg_prob,
            MAX(probability) as max_prob,
            MIN(probability) as min_prob,
            AVG(threshold) as threshold,
            MIN(t_ms) as start_time,
            MAX(t_ms) as end_time
        FROM seizure_predictions
        WHERE upload_id = ?
    ''', (upload_id,))
    
    summary = cursor.fetchone()
    
    duration_seconds = (summary['end_time'] - summary['start_time']) / 1000.0
    seizure_rate = (summary['seizure_detections'] / summary['total_samples']) * 100
    
    print(f"\n{'='*60}")
    print(f"Seizure Detection Summary - Upload {upload_id}")
    print(f"{'='*60}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Duration: {duration_seconds:.1f} seconds")
    print(f"Seizure Detections: {summary['seizure_detections']} ({seizure_rate:.2f}%)")
    print(f"High Confidence (>70%): {summary['high_confidence']}")
    print(f"")
    print(f"Probability Stats:")
    print(f"  Average: {summary['avg_prob']:.4f}")
    print(f"  Maximum: {summary['max_prob']:.4f}")
    print(f"  Minimum: {summary['min_prob']:.4f}")
    print(f"  Threshold: {summary['threshold']:.2f}")
    
    # Detect episodes (continuous seizure events)
    cursor.execute('''
        SELECT 
            MIN(t_ms) as start_ms,
            MAX(t_ms) as end_ms,
            COUNT(*) as sample_count,
            AVG(probability) as avg_probability
        FROM (
            SELECT 
                t_ms,
                probability,
                t_ms - ROW_NUMBER() OVER (ORDER BY t_ms) * 4 as grp
            FROM seizure_predictions
            WHERE upload_id = ? AND is_seizure = 1
        )
        GROUP BY grp
        ORDER BY start_ms
    ''', (upload_id,))
    
    episodes = cursor.fetchall()
    
    if episodes:
        print(f"\nSeizure Episodes: {len(episodes)}")
        print(f"{'='*60}")
        print(f"{'Episode':<10} {'Start (ms)':<12} {'End (ms)':<12} {'Duration':<12} {'Samples':<10}")
        print(f"{'='*60}")
        
        for i, ep in enumerate(episodes, 1):
            duration = (ep['end_ms'] - ep['start_ms']) / 1000.0
            print(f"{i:<10} {ep['start_ms']:<12} {ep['end_ms']:<12} {duration:<12.2f}s {ep['sample_count']:<10}")
    
    print(f"{'='*60}\n")
    conn.close()

def compare_data_types(upload_id):
    """Compare data counts across tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get counts from each table
    cursor.execute('SELECT row_count FROM uploads WHERE upload_id = ?', (upload_id,))
    upload = cursor.fetchone()
    if not upload:
        print(f"❌ Upload {upload_id} not found")
        conn.close()
        return
    
    expected = upload['row_count']
    
    cursor.execute('SELECT COUNT(*) as count FROM eeg_samples WHERE upload_id = ?', (upload_id,))
    samples = cursor.fetchone()['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM aggregated_data_1hz WHERE upload_id = ?', (upload_id,))
    aggregated = cursor.fetchone()['count']
    
    cursor.execute('SELECT COUNT(DISTINCT t_start_ms) as count FROM fft_data WHERE upload_id = ?', (upload_id,))
    fft = cursor.fetchone()['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM seizure_predictions WHERE upload_id = ?', (upload_id,))
    seizures = cursor.fetchone()['count']
    
    cursor.execute('SELECT SUM(CASE WHEN is_seizure = 1 THEN 1 ELSE 0 END) as count FROM seizure_predictions WHERE upload_id = ?', (upload_id,))
    seizure_detected = cursor.fetchone()['count'] or 0
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Data Comparison for Upload {upload_id}")
    print(f"{'='*60}")
    print(f"Expected (from upload): {expected}")
    print(f"EEG Samples: {samples} {'✅' if samples == expected else '❌'}")
    print(f"Aggregated (1Hz): {aggregated} (expected ~{expected//250})")
    print(f"FFT Windows: {fft} (expected ~{expected//250})")
    print(f"Seizure Predictions: {seizures} {'✅' if seizures == expected else '❌'}")
    if seizures > 0:
        rate = (seizure_detected / seizures) * 100
        print(f"  Seizures Detected: {seizure_detected} ({rate:.1f}%)")
    print(f"{'='*60}\n")

def visualize_seizure_timeline(upload_id):
    """Create a simple ASCII timeline of seizure detections"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT t_ms, probability, is_seizure, threshold
        FROM seizure_predictions
        WHERE upload_id = ?
        ORDER BY t_ms
    ''', (upload_id,))
    
    predictions = cursor.fetchall()
    conn.close()
    
    if len(predictions) == 0:
        print(f"❌ No predictions found for upload {upload_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"Seizure Detection Timeline - Upload {upload_id}")
    print(f"{'='*80}")
    
    # Group into bins (every 1000 samples for visualization)
    bin_size = 1000
    n_bins = (len(predictions) + bin_size - 1) // bin_size
    
    print(f"Total Samples: {len(predictions)}")
    print(f"Bins: {n_bins} (each bin = {bin_size} samples)\n")
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(predictions))
        bin_data = predictions[start_idx:end_idx]
        
        seizures = sum(1 for p in bin_data if p['is_seizure'])
        avg_prob = sum(p['probability'] for p in bin_data) / len(bin_data)
        
        bar_length = int((seizures / len(bin_data)) * 50)
        bar = '█' * bar_length
        
        print(f"Bin {i+1:3d} [{start_idx:6d}-{end_idx:6d}] {bar:<50} {seizures:4d}/{len(bin_data):4d} ({avg_prob:.3f})")
    
    print(f"{'='*80}\n")

def interactive_menu():
    """Interactive menu"""
    while True:
        print("\n" + "="*60)
        print("EEG Database Viewer")
        print("="*60)
        print("1. List all uploads")
        print("2. Show summary of an upload")
        print("3. View EEG samples")
        print("4. View aggregated data")
        print("5. View FFT data")
        print("6. View seizure predictions")
        print("7. View seizure detection summary")
        print("8. Compare data types")
        print("9. Visualize seizure timeline")
        print("0. Exit")
        print("="*60)
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            list_uploads()
        elif choice == '2':
            upload_id = int(input("Enter upload_id: "))
            show_summary(upload_id)
        elif choice == '3':
            upload_id = int(input("Enter upload_id: "))
            limit = int(input("Number of samples to show (default 10): ") or 10)
            view_samples(upload_id, limit)
        elif choice == '4':
            upload_id = int(input("Enter upload_id: "))
            limit = int(input("Number of windows to show (default 10): ") or 10)
            view_aggregated(upload_id, limit)
        elif choice == '5':
            upload_id = int(input("Enter upload_id: "))
            channel = int(input("Enter channel (1-8): "))
            limit = int(input("Number of windows to show (default 10): ") or 10)
            view_fft(upload_id, channel, limit)
        elif choice == '6':
            upload_id = int(input("Enter upload_id: "))
            limit = int(input("Number of predictions to show (default 20): ") or 20)
            view_seizure_predictions(upload_id, limit)
        elif choice == '7':
            upload_id = int(input("Enter upload_id: "))
            view_seizure_summary(upload_id)
        elif choice == '8':
            upload_id = int(input("Enter upload_id: "))
            compare_data_types(upload_id)
        elif choice == '9':
            upload_id = int(input("Enter upload_id: "))
            visualize_seizure_timeline(upload_id)
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == '__main__':
    interactive_menu()
