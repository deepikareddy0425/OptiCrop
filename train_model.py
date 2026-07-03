"""
train_model.py
---------------
OptiCrop - Smart Agricultural Production Optimization Engine
Implements Epics 2-4 from the project report:

  Epic 2: Data Collection & Analysis   (load + univariate/bivariate/multivariate EDA)
  Epic 3: Data Pre-Processing          (null check, outlier handling, train/test split)
  Epic 4: Model Building               (K-Means clustering, Logistic Regression,
                                         Decision Tree, Random Forest, KNN comparison,
                                         evaluation, and saving the best model)

Usage:
    python train_model.py

Outputs:
    model/model.pkl        -> best-performing trained classifier
    model/scaler.pkl        -> StandardScaler fitted on training data
    model/label_encoder.pkl -> LabelEncoder mapping crop name <-> class index
    static/images/*.png     -> EDA and evaluation plots
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts / servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

DATA_PATH = os.path.join("data", "Crop_recommendation.csv")
MODEL_DIR = "model"
IMG_DIR = os.path.join("static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Run `python generate_dataset.py` first, "
            "or place the Kaggle Crop_recommendation.csv into the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


def check_nulls(df):
    print("\n--- Null value check ---")
    print(df.isnull().sum())


def handle_outliers(df):
    # Example from report: Potassium (K) tends to contain outliers -> log-transform
    print("\n--- Handling outliers (K via IQR + log transform) ---")
    q1, q3 = df["K"].quantile(0.25), df["K"].quantile(0.75)
    iqr = q3 - q1
    upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    n_outliers = ((df["K"] > upper) | (df["K"] < lower)).sum()
    print(f"Detected {n_outliers} potential outliers in K (bounds: {lower:.2f}-{upper:.2f})")
    df["K_log"] = np.log1p(df["K"])
    return df


def run_eda(df):
    print("\n--- Running EDA (univariate / bivariate / multivariate) ---")

    # Univariate: distribution of each numeric feature
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#30a2da")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "univariate_distributions.png"))
    plt.close()

    # Bivariate: humidity vs crop label
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="humidity", y="rainfall", hue="label", legend=False, alpha=0.6)
    plt.title("Humidity vs Rainfall by Crop (bivariate)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "bivariate_humidity_rainfall.png"))
    plt.close()

    # Multivariate: correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Feature Correlation (multivariate)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"))
    plt.close()

    print(f"EDA plots saved to {IMG_DIR}/")


def run_kmeans(X_scaled):
    print("\n--- K-Means Clustering (Elbow Method) ---")
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "elbow_graph.png"))
    plt.close()
    print(f"Elbow graph saved to {IMG_DIR}/elbow_graph.png")

    # Fit final KMeans with a reasonable k (matches crop-group intuition)
    best_k = 6
    kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    clusters = kmeans_final.fit_predict(X_scaled)
    print(f"K-Means fitted with k={best_k}, cluster sizes: {np.bincount(clusters)}")
    return kmeans_final


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"[{name}] Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    check_nulls(df)
    df = handle_outliers(df)
    run_eda(df)

    X = df[FEATURES].values
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Epic 3, Story 4: train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

    # Epic 4, Story 1: K-Means clustering (exploratory, not used for final prediction)
    run_kmeans(X_scaled)

    # Epic 4, Story 2-3: train & compare classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(evaluate(name, y_test, preds))
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n--- Model comparison (sorted by F1) ---")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    # Confusion matrix for the best model
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"))
    plt.close()

    # Epic 4, Story 4: save best model + scaler + label encoder
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\nSaved best model ({best_name}) to {MODEL_DIR}/model.pkl")
    print(f"Saved scaler to {MODEL_DIR}/scaler.pkl")
    print(f"Saved label encoder to {MODEL_DIR}/label_encoder.pkl")
    print("\nTraining complete. You can now run: python app.py")


if __name__ == "__main__":
    main()
