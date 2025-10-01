from flask import Flask, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io,base64
import matplotlib
matplotlib.use('Agg')
from flask_sqlalchemy import SQLAlchemy
import json

app=Flask(__name__)

model=joblib.load("anomaly_model.pkl")

#database config
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///summaries.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)

#schema definition
class UploadSummary(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    filename=db.Column(db.String(100))
    summary=db.Column(db.Text)
    sensor1_anomalies=db.Column(db.Text)
    sensor2_anomalies=db.Column(db.Text)

with app.app_context():
    db.create_all()

#index route to upload csv file
@app.route('/')
def index(): 
    html="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload CSV File</title>
            <style>
                body{
                    font-family: Arial, sans-serif;
                    background: #f0f4f8;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }
                .upload-container{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 500px;
                    width: 90%;
                }
                .header{
                    background: #a78bfa;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }
                .header h1{
                    margin: 0;
                    font-size: 28px;
                }
                .header p{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }
                .upload-form{
                    margin-top: 20px;
                }
                .file-input-wrapper{
                    position: relative;
                    margin: 20px 0;
                    padding: 40px;
                    border: 3px dashed #a78bfa;
                    border-radius: 10px;
                    background: #faf5ff;
                    transition: all 0.3s ease;
                }
                .file-input-wrapper:hover{
                    border-color: #8b5cf6;
                    background: #f3e8ff;
                }
                .file-input-wrapper input[type="file"]{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    opacity: 0;
                    cursor: pointer;
                }
                .file-input-label{
                    color: #a78bfa;
                    font-size: 18px;
                    font-weight: bold;
                }
                .submit-btn{
                    background: #a78bfa;
                    color: white;
                    border: none;
                    padding: 15px 40px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 18px;
                    font-weight: bold;
                    transition: all 0.3s ease;
                    width: 100%;
                    margin-top: 20px;
                }
                .submit-btn:hover{
                    background: #8b5cf6;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(167, 139, 250, 0.4);
                }
                .submit-btn:disabled{
                    background: #d1d5db;
                    cursor: not-allowed;
                    transform: none;
                }
                .file-name{
                    margin-top: 15px;
                    color: #6b7280;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="upload-container">
                <div class="header">
                    <h1>Anomaly Detection</h1>
                    <p>Upload your sensor data CSV file for analysis</p>
                </div>
                
                <form class="upload-form" method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
                    <div class="file-input-wrapper">
                        <div class="file-input-label">Click or drag file to upload</div>
                        <div class="file-name" id="fileName">No file selected</div>
                        <input type="file" name="csv_file" accept=".csv" required id="fileInput">
                    </div>
                    
                    <button type="submit" class="submit-btn" id="submitBtn">
                        Upload & Analyze
                    </button>
                </form>
            </div>

            <script>
                const fileInput = document.getElementById('fileInput');
                const fileName = document.getElementById('fileName');
                const submitBtn = document.getElementById('submitBtn');

                fileInput.addEventListener('change', function(e) {
                    if(e.target.files.length > 0){
                        fileName.textContent='✓ '+e.target.files[0].name;
                        fileName.style.color='#059669';
                        fileName.style.fontWeight='bold';
                    } else{
                        fileName.textContent='No file selected';
                        fileName.style.color='#6b7280';
                        fileName.style.fontWeight='normal';
                    }
                });
            </script>
        </body>
        </html>
        """
    return html

#upload route to detected anomalies
@app.route('/upload',methods=['POST'])
def upload():
    if 'csv_file' not in request.files:
        return "No file uploaded"
    
    file=request.files['csv_file']

    if file.filename=='':
        return "Please select a file"
    
    if not file.filename.endswith('.csv'):
        return "Only CSV files allowed"

    df=pd.read_csv(file)
    X=df[['sensor1','sensor2']]
    predictions=model.predict(X)

    anomaly=predictions==-1
    anomalies_df=df[anomaly].copy()

    sensor1_anomalies=[]
    sensor2_anomalies=[]

    sensor1_mean=df['sensor1'].mean()
    sensor2_mean=df['sensor2'].mean()

    for id, row in anomalies_df.iterrows():
        s1_dev=abs(row['sensor1']-sensor1_mean)
        s2_dev=abs(row['sensor2']-sensor2_mean)

        if s1_dev>s2_dev:
            sensor1_anomalies.append(row.to_dict())
        else:
            sensor2_anomalies.append(row.to_dict())

    anomalies_s1=pd.DataFrame(sensor1_anomalies) if sensor1_anomalies else pd.DataFrame()
    anomalies_s2=pd.DataFrame(sensor2_anomalies) if sensor2_anomalies else pd.DataFrame()

    #plotting
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12, 8))

    ax1.plot(df['time'],df['sensor1'],linewidth=2,color='blue',label='Sensor 1')
    if not anomalies_s1.empty:
        ax1.scatter(anomalies_s1['time'],anomalies_s1['sensor1'],marker='x',color='red',label='Anomalies')
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Sensor 1", fontsize=12)
    ax1.set_title("Sensor 1 vs Time", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['time'],df['sensor2'],linewidth=2,color='green',label='Sensor 2')
    if not anomalies_s2.empty:
        ax2.scatter(anomalies_s2['time'],anomalies_s2['sensor2'],marker='x',color='red',label='Anomalies')
    ax2.set_xlabel("Time",fontsize=12)
    ax2.set_ylabel("Sensor 2",fontsize=12)
    ax2.set_title("Sensor 2 vs Time",fontsize=13,fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    img=io.BytesIO()
    plt.savefig(img,format='png',bbox_inches='tight',dpi=100)
    plt.close()
    img.seek(0)
    plot_url=base64.b64encode(img.getvalue()).decode()

    summary=df[['sensor1','sensor2']].describe().to_json()
    s1_anomalies_json=anomalies_s1.to_json(orient='records') if not anomalies_s1.empty else "[]"
    s2_anomalies_json=anomalies_s2.to_json(orient='records') if not anomalies_s2.empty else "[]"

    new_summary=UploadSummary(
        filename=file.filename,
        summary=summary,
        sensor1_anomalies=s1_anomalies_json,
        sensor2_anomalies=s2_anomalies_json
    )
    db.session.add(new_summary)
    db.session.commit()

    saved_id=new_summary.id

    anomaly_display = ""
    
    if not anomalies_s1.empty:
        s1_html=anomalies_s1.to_html(classes='data-table', index=False)
        anomaly_display+=f"<h3 style='color: #dc3545;'>Sensor 1 Anomalies ({len(anomalies_s1)} detected)</h3>{s1_html}"
    else:
        anomaly_display+="<h3 style='color: #28a745;'>Sensor 1: No anomalies detected</h3>"
    
    if not anomalies_s2.empty:
        s2_html=anomalies_s2.to_html(classes='data-table', index=False)
        anomaly_display+=f"<h3 style='color: #ff8c00;'>Sensor 2 Anomalies ({len(anomalies_s2)} detected)</h3>{s2_html}"
    else:
        anomaly_display+="<h3 style='color: #28a745;'>Sensor 2: No anomalies detected</h3>"

    preview_html=df.head(10).to_html(classes='data-table', index=False)
    stats_html=df[['sensor1', 'sensor2']].describe().to_html(classes='data-table')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results</title>
        <style>
            body{{
                font-family: Arial, sans-serif;
                background: #f0f4f8;
                margin: 0;
                padding: 10px;
            }}
            .container{{
                max-width: 1400px;
                margin: auto;
            }}
            .header{{
                background: #a78bfa;
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
            }}
            .card{{
                background: white;
                padding: 25px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .card h2{{
                color: #333;
                border-bottom: 3px solid #a78bfa;
                padding-bottom: 10px;
            }}
            .data-table{{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .data-table th{{
                background: #a78bfa;
                color: white;
                padding: 10px;
                text-align: left;
            }}
            .data-table td{{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            .data-table tr:hover{{
                background: #f5f5f5;
            }}
            img{{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            button{{
                background: #a78bfa;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
            }}
            button:hover{{
                background: #8b5cf6;
            }}
            input[type="number"]{{
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                width: 200px;
                font-size: 16px;
            }}
            #summary-output{{
                background: #2d2d2d;
                color: #00ff00;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                margin-top: 10px;
            }}
            .success-badge{{
                background: #28a745;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
            }}
            .anomaly-section{{
                background: #fff9e6;
                padding: 20px;
                border-radius: 8px;
                border-left: 5px solid #ffc107;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Analysis Results</h1>
                <p>File: <strong>{file.filename}</strong></p>
                <span class="success-badge">Saved as ID: {saved_id}</span>
            </div>

            <div class="card">
                <h2>Data Preview (First 10 Rows)</h2>
                {preview_html}
            </div>

            <div class="card">
                <h2>Sensor Visualizations</h2>
                <img src="data:image/png;base64,{plot_url}" alt="Sensor Plots">
            </div>

            <div class="card">
                <h2>Summary Statistics</h2>
                {stats_html}
            </div>

            <div class="card">
                <h2>Anomaly Detection Results</h2>
                <div class="anomaly-section">
                    {anomaly_display}
                </div>
            </div>

            <div class="card">
                <h2>API Testing</h2>
                <button onclick="window.open('/api/summaries', '_blank')">
                    View All Summaries
                </button>
                <button onclick="window.location.href='/'">
                    Upload Another File
                </button>
                <br><br>
                <div>
                    <input type="number" id="summary-id" placeholder="Enter Summary ID">
                    <button onclick="fetchSummary()">Get Summary by ID</button>
                </div>
                <pre id="summary-output"></pre>
            </div>
        </div>

        <script>
        async function fetchSummary(){{
            const id=document.getElementById('summary-id').value;
            const output=document.getElementById('summary-output');
            
            if(!id){{
                alert("Please enter a valid ID!");
                return;
            }}

            try{{
                const response=await fetch(`/api/summary/${{id}}`);
                
                if(response.status===404){{
                    output.textContent="Summary not found!";
                    return;
                }}

                const data=await response.json();
                output.textContent=JSON.stringify(data, null, 2);
            }} catch(error){{
                output.textContent="Error: " + error.message;
            }}
        }}
        </script>
    </body>
    </html>
    """

    return html

#api/summaries route to get all summaries
@app.route('/api/summaries',methods=['GET'])
def get_summaries():
    summaries=UploadSummary.query.all()
    result=[]
    for s in summaries:
        result.append({
            "id":s.id,
            "filename":s.filename,
            "summary":json.loads(s.summary),
            "sensor1_anomalies":json.loads(s.sensor1_anomalies),
            "sensor2_anomalies":json.loads(s.sensor2_anomalies)
        })
    return{"summaries":result}

#api/summary/id to get a specific summary
@app.route('/api/summary/<int:summary_id>',methods=['GET'])
def get_summary_by_id(summary_id):
    summary=UploadSummary.query.get(summary_id)
    
    if summary is None:
        return {"error": "Summary not found"},404

    return {
        "id":summary.id,
        "filename":summary.filename,
        "summary":json.loads(summary.summary),
        "sensor1_anomalies":json.loads(summary.sensor1_anomalies),
        "sensor2_anomalies":json.loads(summary.sensor2_anomalies)
    }


if __name__=='__main__':
    app.run(debug=True)