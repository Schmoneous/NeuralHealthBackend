const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const admin = require('firebase-admin');

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Firebase Admin
const serviceAccount = require(process.env.FIREBASE_SERVICE_ACCOUNT_PATH || './firebase-key.json');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// SQLite Database Connection
const db = new sqlite3.Database('./eeg1.db', (err) => {
  if (err) {
    console.error('❌ Error opening database:', err.message);
  } else {
    console.log('✅ Connected to SQLite database');
    initDatabase();
  }
});

// Initialize database tables
function initDatabase() {
  db.serialize(() => {
    // Uploads table
    db.run(`
      CREATE TABLE IF NOT EXISTS uploads (
        upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_uid TEXT NOT NULL,
        received_at TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        channel_count INTEGER DEFAULT 8,
        source TEXT DEFAULT 'mobile_app',
        is_ai_model_data INTEGER DEFAULT 0
      )
    `, (err) => {
      if (err) console.error('Error creating uploads table:', err);
      else console.log('✅ Uploads table ready');
    });

    // EEG Samples table (8 channels)
    db.run(`
      CREATE TABLE IF NOT EXISTS eeg_samples (
        upload_id INTEGER NOT NULL,
        t_ms INTEGER NOT NULL,
        seq INTEGER NOT NULL,
        ch1_uV REAL,
        ch2_uV REAL,
        ch3_uV REAL,
        ch4_uV REAL,
        ch5_uV REAL,
        ch6_uV REAL,
        ch7_uV REAL,
        ch8_uV REAL,
        FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
      )
    `, (err) => {
      if (err) console.error('Error creating eeg_samples table:', err);
      else console.log('✅ EEG samples table ready');
    });

    // Aggregated data table (1 Hz mobile-optimized)
    db.run(`
      CREATE TABLE IF NOT EXISTS aggregated_data_1hz (
        upload_id INTEGER NOT NULL,
        t_ms INTEGER NOT NULL,
        avg_ch1 REAL, avg_ch2 REAL, avg_ch3 REAL, avg_ch4 REAL,
        avg_ch5 REAL, avg_ch6 REAL, avg_ch7 REAL, avg_ch8 REAL,
        std_ch1 REAL, std_ch2 REAL, std_ch3 REAL, std_ch4 REAL,
        std_ch5 REAL, std_ch6 REAL, std_ch7 REAL, std_ch8 REAL,
        FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
      )
    `, (err) => {
      if (err) console.error('Error creating aggregated_data_1hz table:', err);
      else console.log('✅ Aggregated data table ready');
    });

    // FFT data table
    db.run(`
      CREATE TABLE IF NOT EXISTS fft_data (
        upload_id INTEGER NOT NULL,
        t_start_ms INTEGER NOT NULL,
        t_end_ms INTEGER NOT NULL,
        channel INTEGER NOT NULL,
        delta REAL,
        theta REAL,
        alpha REAL,
        beta REAL,
        gamma REAL,
        FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
      )
    `, (err) => {
      if (err) console.error('Error creating fft_data table:', err);
      else console.log('✅ FFT data table ready');
    });

    // Seizure predictions table
    db.run(`
      CREATE TABLE IF NOT EXISTS seizure_predictions (
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
    `, (err) => {
      if (err) console.error('Error creating seizure_predictions table:', err);
      else console.log('✅ Seizure predictions table ready');
    });

    // Create indexes for performance
    db.run('CREATE INDEX IF NOT EXISTS idx_eeg_upload ON eeg_samples(upload_id)');
    db.run('CREATE INDEX IF NOT EXISTS idx_eeg_time ON eeg_samples(t_ms)');
    db.run('CREATE INDEX IF NOT EXISTS idx_agg_upload ON aggregated_data_1hz(upload_id)');
    db.run('CREATE INDEX IF NOT EXISTS idx_fft_upload ON fft_data(upload_id)');
    db.run('CREATE INDEX IF NOT EXISTS idx_fft_channel ON fft_data(channel)');
    db.run('CREATE INDEX IF NOT EXISTS idx_seizure_upload ON seizure_predictions(upload_id)');
    db.run('CREATE INDEX IF NOT EXISTS idx_seizure_time ON seizure_predictions(t_ms)');
    db.run('CREATE INDEX IF NOT EXISTS idx_seizure_flag ON seizure_predictions(is_seizure)');
    console.log('✅ Database indexes created');
  });
}

// Middleware: Verify Firebase Token
async function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized: No token provided' });
  }

  const token = authHeader.split('Bearer ')[1];

  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    req.user = decodedToken;
    next();
  } catch (error) {
    console.error('❌ Token verification failed:', error);
    return res.status(401).json({ error: 'Unauthorized: Invalid token' });
  }
}

// Middleware: Check if user is a doctor
async function requireDoctor(req, res, next) {
  try {
    const userDoc = await admin.firestore()
      .collection('users')
      .doc(req.user.uid)
      .get();
    
    const userData = userDoc.data();
    if (userData && userData.role === 'doctor') {
      req.isDoctor = true;
      next();
    } else {
      return res.status(403).json({ error: 'Forbidden: Doctor role required' });
    }
  } catch (error) {
    console.error('❌ Error checking doctor role:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// ROUTES

// Health check
app.get('/healthz', (req, res) => {
  res.json({ ok: true, timestamp: new Date().toISOString() });
});

// POST /api/data - Ingest EEG data
app.post('/api/data', requireAuth, async (req, res) => {
  const { csvData, uart, numChannels = 8, node, source = 'mobile_app' } = req.body;
  const userId = req.user.uid;

  console.log(`📥 Received data from user: ${userId}`);

  if (!csvData && !uart) {
    return res.status(400).json({ error: 'No data provided' });
  }

  try {
    let parsedData = [];

    if (csvData) {
      const lines = csvData.trim().split('\n');
      const hasHeader = lines[0].includes('t_ms') || lines[0].includes('seq');
      const dataLines = hasHeader ? lines.slice(1) : lines;

      parsedData = dataLines.map(line => {
        const values = line.split(',').map(v => parseFloat(v.trim()));
        return {
          t_ms: values[0],
          seq: values[1],
          channels: values.slice(2, 2 + numChannels)
        };
      });
    } else if (uart) {
      // Handle UART format
      parsedData = uart.map(sample => ({
        t_ms: sample.t_ms,
        seq: sample.seq,
        channels: sample.channels || []
      }));
    }

    if (parsedData.length === 0) {
      return res.status(400).json({ error: 'No valid data to insert' });
    }

    // Insert upload record
    const uploadStmt = db.prepare(`
      INSERT INTO uploads (user_uid, received_at, row_count, channel_count, source)
      VALUES (?, ?, ?, ?, ?)
    `);

    uploadStmt.run(
      userId,
      new Date().toISOString(),
      parsedData.length,
      numChannels,
      source,
      function(err) {
        if (err) {
          console.error('❌ Error inserting upload:', err);
          return res.status(500).json({ error: 'Database error' });
        }

        const uploadId = this.lastID;
        console.log(`✅ Created upload_id: ${uploadId}`);

        // Insert EEG samples
        const sampleStmt = db.prepare(`
          INSERT INTO eeg_samples (
            upload_id, t_ms, seq,
            ch1_uV, ch2_uV, ch3_uV, ch4_uV, ch5_uV, ch6_uV, ch7_uV, ch8_uV
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `);

        let insertedCount = 0;
        db.serialize(() => {
          db.run('BEGIN TRANSACTION');

          parsedData.forEach(sample => {
            const channels = sample.channels;
            sampleStmt.run(
              uploadId,
              sample.t_ms,
              sample.seq,
              channels[0] || null,
              channels[1] || null,
              channels[2] || null,
              channels[3] || null,
              channels[4] || null,
              channels[5] || null,
              channels[6] || null,
              channels[7] || null,
              (err) => {
                if (err) console.error('Error inserting sample:', err);
                else insertedCount++;
              }
            );
          });

          db.run('COMMIT', (err) => {
            if (err) {
              console.error('❌ Transaction commit failed:', err);
              return res.status(500).json({ error: 'Failed to save data' });
            }

            sampleStmt.finalize();
            console.log(`✅ Inserted ${insertedCount} samples for upload ${uploadId}`);

            res.json({
              message: 'Data ingested successfully',
              upload_id: uploadId,
              samples_inserted: insertedCount
            });
          });
        });
      }
    );

    uploadStmt.finalize();

  } catch (error) {
    console.error('❌ Error processing data:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /uploads - Get user's uploads
app.get('/uploads', requireAuth, (req, res) => {
  const userId = req.user.uid;
  const limit = parseInt(req.query.limit) || 20;

  const query = `
    SELECT upload_id, user_uid, received_at, row_count, channel_count, source, is_ai_model_data
    FROM uploads
    WHERE user_uid = ?
    ORDER BY received_at DESC
    LIMIT ?
  `;

  db.all(query, [userId, limit], (err, rows) => {
    if (err) {
      console.error('❌ Error fetching uploads:', err);
      return res.status(500).json({ error: 'Database error' });
    }

    res.json(rows);
  });
});

// GET /uploads/:id/samples - Get raw samples
app.get('/uploads/:id/samples', requireAuth, (req, res) => {
  const uploadId = parseInt(req.params.id);
  const limit = parseInt(req.query.limit) || 5000;
  const tStart = req.query.t_start ? parseInt(req.query.t_start) : null;
  const tEnd = req.query.t_end ? parseInt(req.query.t_end) : null;

  // Verify ownership
  db.get('SELECT user_uid FROM uploads WHERE upload_id = ?', [uploadId], async (err, upload) => {
    if (err || !upload) {
      return res.status(404).json({ error: 'Upload not found' });
    }

    if (upload.user_uid !== req.user.uid) {
      // Check if requester is a doctor with access
      try {
        const userDoc = await admin.firestore().collection('users').doc(req.user.uid).get();
        const isDoctor = userDoc.data()?.role === 'doctor';
        
        if (!isDoctor) {
          return res.status(403).json({ error: 'Access denied' });
        }

        // Check if patient is assigned to this doctor
        const patientDoc = await admin.firestore()
          .collection('users-healthInfromation')
          .doc(upload.user_uid)
          .get();
        
        const assignedDoctor = patientDoc.data()?.doctorId;
        if (assignedDoctor !== req.user.uid) {
          return res.status(403).json({ error: 'Patient not assigned to you' });
        }
      } catch (error) {
        console.error('Error checking doctor access:', error);
        return res.status(500).json({ error: 'Internal server error' });
      }
    }

    // Build query
    let query = `
      SELECT t_ms, seq, ch1_uV, ch2_uV, ch3_uV, ch4_uV, ch5_uV, ch6_uV, ch7_uV, ch8_uV
      FROM eeg_samples
      WHERE upload_id = ?
    `;
    const params = [uploadId];

    if (tStart !== null) {
      query += ' AND t_ms >= ?';
      params.push(tStart);
    }
    if (tEnd !== null) {
      query += ' AND t_ms <= ?';
      params.push(tEnd);
    }

    query += ' ORDER BY t_ms LIMIT ?';
    params.push(limit);

    db.all(query, params, (err, rows) => {
      if (err) {
        console.error('❌ Error fetching samples:', err);
        return res.status(500).json({ error: 'Database error' });
      }

      res.json(rows);
    });
  });
});

// GET /mobile/aggregated/:id - Get 1Hz aggregated data
app.get('/mobile/aggregated/:id', requireAuth, (req, res) => {
  const uploadId = parseInt(req.params.id);
  const limit = parseInt(req.query.limit) || 3600;

  db.get('SELECT user_uid FROM uploads WHERE upload_id = ?', [uploadId], async (err, upload) => {
    if (err || !upload) {
      return res.status(404).json({ error: 'Upload not found' });
    }

    if (upload.user_uid !== req.user.uid) {
      // Check doctor access
      try {
        const userDoc = await admin.firestore().collection('users').doc(req.user.uid).get();
        const isDoctor = userDoc.data()?.role === 'doctor';
        
        if (!isDoctor) {
          return res.status(403).json({ error: 'Access denied' });
        }

        const patientDoc = await admin.firestore()
          .collection('users-healthInfromation')
          .doc(upload.user_uid)
          .get();
        
        if (patientDoc.data()?.doctorId !== req.user.uid) {
          return res.status(403).json({ error: 'Access denied' });
        }
      } catch (error) {
        return res.status(500).json({ error: 'Internal server error' });
      }
    }

    const query = `
      SELECT * FROM aggregated_data_1hz
      WHERE upload_id = ?
      ORDER BY t_ms
      LIMIT ?
    `;

    db.all(query, [uploadId, limit], (err, rows) => {
      if (err) {
        console.error('❌ Error fetching aggregated data:', err);
        return res.status(500).json({ error: 'Database error' });
      }

      res.json(rows);
    });
  });
});

// GET /mobile/fft/:id - Get FFT data
app.get('/mobile/fft/:id', requireAuth, (req, res) => {
  const uploadId = parseInt(req.params.id);
  const channel = req.query.channel ? parseInt(req.query.channel) : null;
  const limit = parseInt(req.query.limit) || 1000;

  db.get('SELECT user_uid FROM uploads WHERE upload_id = ?', [uploadId], async (err, upload) => {
    if (err || !upload) {
      return res.status(404).json({ error: 'Upload not found' });
    }

    if (upload.user_uid !== req.user.uid) {
      // Check doctor access
      try {
        const userDoc = await admin.firestore().collection('users').doc(req.user.uid).get();
        const isDoctor = userDoc.data()?.role === 'doctor';
        
        if (!isDoctor) {
          return res.status(403).json({ error: 'Access denied' });
        }

        const patientDoc = await admin.firestore()
          .collection('users-healthInfromation')
          .doc(upload.user_uid)
          .get();
        
        if (patientDoc.data()?.doctorId !== req.user.uid) {
          return res.status(403).json({ error: 'Access denied' });
        }
      } catch (error) {
        return res.status(500).json({ error: 'Internal server error' });
      }
    }

    let query = `
      SELECT t_start_ms, t_end_ms, channel, delta, theta, alpha, beta, gamma
      FROM fft_data
      WHERE upload_id = ?
    `;
    const params = [uploadId];

    if (channel !== null) {
      query += ' AND channel = ?';
      params.push(channel);
    }

    query += ' ORDER BY t_start_ms LIMIT ?';
    params.push(limit);

    db.all(query, params, (err, rows) => {
      if (err) {
        console.error('❌ Error fetching FFT data:', err);
        return res.status(500).json({ error: 'Database error' });
      }

      const response = {
        upload_id: uploadId,
        windows: rows.map(row => ({
          t_start_ms: row.t_start_ms,
          t_end_ms: row.t_end_ms,
          channel: row.channel,
          bands: {
            delta: row.delta,
            theta: row.theta,
            alpha: row.alpha,
            beta: row.beta,
            gamma: row.gamma
          }
        })),
        window_duration: '1 second',
        bands: ['delta', 'theta', 'alpha', 'beta', 'gamma']
      };

      res.json(response);
    });
  });
});

// GET /mobile/seizures/:id/summary - Get seizure detection summary
app.get('/mobile/seizures/:id/summary', requireAuth, (req, res) => {
  const uploadId = parseInt(req.params.id);

  db.get('SELECT user_uid FROM uploads WHERE upload_id = ?', [uploadId], async (err, upload) => {
    if (err || !upload) {
      return res.status(404).json({ error: 'Upload not found' });
    }

    if (upload.user_uid !== req.user.uid) {
      // Check doctor access
      try {
        const userDoc = await admin.firestore().collection('users').doc(req.user.uid).get();
        const isDoctor = userDoc.data()?.role === 'doctor';
        
        if (!isDoctor) {
          return res.status(403).json({ error: 'Access denied' });
        }

        const patientDoc = await admin.firestore()
          .collection('users-healthInfromation')
          .doc(upload.user_uid)
          .get();
        
        if (patientDoc.data()?.doctorId !== req.user.uid) {
          return res.status(403).json({ error: 'Access denied' });
        }
      } catch (error) {
        return res.status(500).json({ error: 'Internal server error' });
      }
    }

    // Check if predictions exist
    db.get(
      'SELECT COUNT(*) as count FROM seizure_predictions WHERE upload_id = ?',
      [uploadId],
      (err, countRow) => {
        if (err || !countRow || countRow.count === 0) {
          return res.json({
            upload_id: uploadId,
            processed: false,
            message: 'No seizure predictions available for this upload'
          });
        }

        // Get summary statistics
        const summaryQuery = `
          SELECT 
            COUNT(*) as total_samples,
            SUM(CASE WHEN is_seizure = 1 THEN 1 ELSE 0 END) as seizure_detections,
            SUM(CASE WHEN probability > 0.7 THEN 1 ELSE 0 END) as high_confidence_seizures,
            AVG(probability) as avg_probability,
            MAX(probability) as max_probability,
            AVG(threshold) as threshold,
            (MAX(t_ms) - MIN(t_ms)) / 1000.0 as duration_seconds
          FROM seizure_predictions
          WHERE upload_id = ?
        `;

        db.get(summaryQuery, [uploadId], (err, summary) => {
          if (err) {
            console.error('❌ Error fetching summary:', err);
            return res.status(500).json({ error: 'Database error' });
          }

          const seizureRate = (summary.seizure_detections / summary.total_samples) * 100;

          // Detect episodes (continuous seizure events within 1 second)
          const episodeQuery = `
            SELECT 
              MIN(t_ms) as start_ms,
              MAX(t_ms) as end_ms,
              COUNT(*) as sample_count,
              AVG(probability) as avg_probability,
              MAX(probability) as max_probability
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
          `;

          db.all(episodeQuery, [uploadId], (err, episodes) => {
            if (err) {
              console.error('❌ Error fetching episodes:', err);
              episodes = [];
            }

            const response = {
              upload_id: uploadId,
              processed: true,
              summary: {
                total_samples: summary.total_samples,
                seizure_detections: summary.seizure_detections,
                high_confidence_seizures: summary.high_confidence_seizures,
                seizure_rate_percent: parseFloat(seizureRate.toFixed(2)),
                avg_probability: parseFloat(summary.avg_probability.toFixed(3)),
                max_probability: parseFloat(summary.max_probability.toFixed(3)),
                threshold: parseFloat(summary.threshold.toFixed(2)),
                duration_seconds: parseFloat(summary.duration_seconds.toFixed(1))
              },
              episodes: episodes.map(ep => ({
                start_ms: ep.start_ms,
                end_ms: ep.end_ms,
                duration_ms: ep.end_ms - ep.start_ms,
                sample_count: ep.sample_count,
                avg_probability: parseFloat(ep.avg_probability.toFixed(3)),
                max_probability: parseFloat(ep.max_probability.toFixed(3))
              })),
              episode_count: episodes.length
            };

            res.json(response);
          });
        });
      }
    );
  });
});

// GET /doctor/patients/:patientId/uploads - Doctor access to patient uploads
app.get('/doctor/patients/:patientId/uploads', requireAuth, requireDoctor, async (req, res) => {
  const patientId = req.params.patientId;
  const limit = parseInt(req.query.limit) || 100;

  try {
    // Verify patient is assigned to this doctor
    const patientDoc = await admin.firestore()
      .collection('users-healthInfromation')
      .doc(patientId)
      .get();
    
    if (!patientDoc.exists || patientDoc.data()?.doctorId !== req.user.uid) {
      return res.status(403).json({ error: 'Patient not assigned to you' });
    }

    const query = `
      SELECT upload_id, user_uid, received_at, row_count, channel_count, source, is_ai_model_data
      FROM uploads
      WHERE user_uid = ?
      ORDER BY received_at DESC
      LIMIT ?
    `;

    db.all(query, [patientId, limit], (err, rows) => {
      if (err) {
        console.error('❌ Error fetching patient uploads:', err);
        return res.status(500).json({ error: 'Database error' });
      }

      res.json(rows);
    });

  } catch (error) {
    console.error('❌ Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Database: ./eeg1.db`);
  console.log(`🔥 Firebase Admin initialized`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down gracefully...');
  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err);
    } else {
      console.log('✅ Database connection closed');
    }
    process.exit(0);
  });
});
