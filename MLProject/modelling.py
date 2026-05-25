import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.metrics import accuracy_score, f1_score
=======
from sklearn.metrics import accuracy_score
>>>>>>> dc9c426e6aba7b92756ccd97e1f9dd9f32961c7b

df = pd.read_csv("heart_disease_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
<<<<<<< HEAD
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Workflow-CI")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
=======
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, random_state=42)
>>>>>>> dc9c426e6aba7b92756ccd97e1f9dd9f32961c7b
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
<<<<<<< HEAD
    f1 = f1_score(y_test, pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
    print("F1:", f1)
=======

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
>>>>>>> dc9c426e6aba7b92756ccd97e1f9dd9f32961c7b
