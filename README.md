# AI-Dashboard
This project is a simple AI-powered dashboard built with Flask.
It lets you upload sensor CSV files, visualize the data, compute statistics, and automatically detect anomalies using a machine learning model.


## 1. Clone the repository
```text
 git clone https://github.com/DarshanaCV/AI-Dashboard.git
 cd AI-Dashboard
```

## 2. Install dependencies
```text
 pip install -r requirements.txt
```

## 3. Train the anomaly detection model (only needed once)
```text
 python train_model.py
```
This will generate a file called anomaly_model.pkl.

## 4. Start the Flask app
```text 
 python app.py
```

## 5. Open in browser
 Go to http://127.0.0.1:5000
 Upload a CSV (format: time, sensor1, sensor2)
 You’ll see:
- A preview of the first 10 rows
- Line plots for both sensors
- Summary statistic
- Anomalies marked on the graph and listed in tables


# How the anomaly model works
I used Isolation Forest from scikit-learn.
For training, I generated synthetic sensor data:
  1. Normal readings with small noise.
  2. A few sudden spikes.
  3. A gradual drift (slow increase over time).
The model learns the “normal” pattern and flags unusual points as anomalies.
During upload, the app loads the trained model (anomaly_model.pkl) and predicts on the new data.

Predictions:
- 1 = normal point
- -1 = anomaly


# Database schema (SQLite via SQLAlchemy)
Each uploaded file gets saved into a table called upload_summary.

```text
upload_summary
-------------
id                (integer, primary key)
filename          (string, uploaded file name)
summary           (text, JSON string of summary stats)
sensor1_anomalies (text, JSON list of sensor1 anomalies)
sensor2_anomalies (text, JSON list of sensor2 anomalies)
```

You can fetch results later via API:
- /api/summaries -> list all uploads
- /api/summary/<id> -> details of a specific upload


# Known limitations
- The anomaly detection is basic, Isolation Forest on synthetic data.
- It may misclassify if your real sensor data is very different.
- No authentication, anyone can upload files.
- Only works with CSVs having exact columns: time, sensor1, sensor2.
- Plots are static images (PNG). For more interactivity, libraries like Plotly could be used.
- Database is SQLite (fine for demo, but not for large-scale production).

Upload your CSVs, visualize the data, and let the dashboard find anomalies for you.
