import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

def generate_training_data(samples=2000):
    
    #normal data
    time=np.linspace(0,1,samples)
    sensor1=np.random.normal(0.11,0.015,samples)
    sensor2=np.random.normal(1.20,0.025,samples)

    #adding spikes
    unique_ids=np.random.choice(range(samples),100,replace=False)
    for id in unique_ids:
        if np.random.rand()>0.5: #50% chance for sensor 1 and 2
            sensor1[id]+=np.random.uniform(2,5) #random float 
        else:
            sensor2[id]+=np.random.uniform()

    #adding dirft
    start=np.random.randint(1000, 1500) #choosing random index
    length=400
    drift=np.linspace(0,2,length)
    
    if np.random.rand()>0.5:
        sensor1[start:start+length]+=drift
    else:
        sensor2[start:start+length]+=drift


    return pd.DataFrame({
        "time":time,
        "sensor1":sensor1,
        "sensor2":sensor2
    })


#training model
train_data=generate_training_data()
X_train=train_data[['sensor1','sensor2']]

model=IsolationForest(    
    contamination=0.08, #after setting auto, 8% was better threshold
    random_state=42,
    n_estimators=150,
    max_samples='auto',
    max_features=2)

model.fit(X_train)

predictions=model.predict(X_train)
anomalies=(predictions==-1).sum()
print(f"{anomalies} in training data")


#saving model
joblib.dump(model,'anomaly_model.pkl')
print("Model trained and saved as 'anomaly_model.pkl'")