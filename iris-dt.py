import mlflow
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import dagshub
dagshub.init(repo_owner='092914rkumar', repo_name='mlflow-dagshub', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/092914rkumar/mlflow-dagshub.mlflow")

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 3
random_state = 42

mlflow.set_experiment("Iris Decision Tree Experiment")

with mlflow.start_run(run_name="Iris Decision Tree Classifier"):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model, "decision_tree_model")
    mlflow.set_tag("model_type", "DecisionTreeClassifier")
    mlflow.set_tag("dataset", "Iris")

    print(f"Accuracy: {accuracy}")