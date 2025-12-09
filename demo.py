import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import dagshub
import mlflow
import warnings
import joblib
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


dagshub.init(repo_owner='kalluridinesh70', repo_name='MLPipeline_with_dagshub', mlflow=True)


remote_server_url="https://dagshub.com/kalluridinesh70/MLPipeline_with_dagshub.mlflow"
mlflow.set_tracking_uri(remote_server_url)

df = pd.read_csv("seattle-weather.csv")
df1 = df[df['precipitation'] != 0.0].reset_index(drop=True)
df2 = df1.drop(columns=["date"])

x = df2[["precipitation","temp_max","temp_min","wind"]]
y = df2["weather"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=500, class_weight='balanced')  
    model.fit(x_train, y_train)
    
    preds = model.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    classification = classification_report(y_test, preds)
    cmatrix=confusion_matrix(y_test,preds)
    print("accuracy", accuracy)
    print("classification", classification)
    print("cmatrix", cmatrix)
    # Save and log classification report
    with open("classification_report.txt", "w") as f:
        f.write(classification)
    mlflow.log_metric("accuracy", accuracy)
    np.savetxt("confusion_matrix.csv", cmatrix, delimiter=",", fmt="%d")
    mlflow.log_artifact("confusion_matrix.csv")
    mlflow.log_artifact("classification_report.txt")
    