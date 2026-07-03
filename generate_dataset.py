"""
generate_dataset.py
--------------------
Generates data/Crop_recommendation.csv — a synthetic-but-realistic
agricultural dataset with the same schema as the Kaggle
"Crop Recommendation Dataset" referenced in the project report:

    N, P, K, temperature, humidity, ph, rainfall, label

Each of the 22 crop classes is generated from a distribution centered
on typical real-world agronomic values for that crop, so the trained
models behave sensibly. Run this once before train_model.py if you
don't have your own Crop_recommendation.csv from Kaggle.

If you already downloaded the real dataset from Kaggle, just place it
at data/Crop_recommendation.csv and skip this script entirely.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Approximate agronomic parameter centers (N, P, K, temp C, humidity %, pH, rainfall mm)
# and standard deviations used to synthesize samples per crop.
CROP_PROFILES = {
    "rice":        dict(N=(80, 15), P=(45, 10), K=(40, 10), temperature=(24, 2),  humidity=(82, 5),  ph=(6.4, 0.4), rainfall=(230, 30)),
    "maize":       dict(N=(78, 15), P=(48, 10), K=(20, 8),  temperature=(23, 3),  humidity=(65, 8),  ph=(6.3, 0.4), rainfall=(85, 20)),
    "chickpea":    dict(N=(40, 10), P=(68, 10), K=(80, 10), temperature=(19, 3),  humidity=(16, 5),  ph=(7.3, 0.4), rainfall=(80, 15)),
    "kidneybeans": dict(N=(21, 8),  P=(68, 10), K=(20, 6),  temperature=(18, 3),  humidity=(21, 5),  ph=(5.7, 0.4), rainfall=(105, 20)),
    "pigeonpeas":  dict(N=(21, 8),  P=(68, 10), K=(20, 6),  temperature=(27, 4),  humidity=(48, 10), ph=(5.8, 0.5), rainfall=(150, 30)),
    "mothbeans":   dict(N=(21, 8),  P=(48, 10), K=(20, 6),  temperature=(28, 3),  humidity=(53, 10), ph=(6.8, 0.5), rainfall=(50, 15)),
    "mungbean":    dict(N=(21, 8),  P=(47, 10), K=(20, 6),  temperature=(28, 3),  humidity=(85, 5),  ph=(6.7, 0.4), rainfall=(48, 12)),
    "blackgram":   dict(N=(40, 8),  P=(67, 10), K=(19, 6),  temperature=(29, 3),  humidity=(65, 8),  ph=(7.1, 0.4), rainfall=(68, 15)),
    "lentil":      dict(N=(19, 6),  P=(68, 10), K=(19, 6),  temperature=(24, 3),  humidity=(65, 8),  ph=(6.9, 0.4), rainfall=(46, 12)),
    "pomegranate": dict(N=(19, 6),  P=(18, 5),  K=(40, 8),  temperature=(21, 3),  humidity=(90, 4),  ph=(6.4, 0.4), rainfall=(107, 20)),
    "banana":      dict(N=(100, 12),P=(82, 10), K=(50, 10), temperature=(27, 2),  humidity=(80, 5),  ph=(6.0, 0.4), rainfall=(105, 20)),
    "mango":       dict(N=(20, 6),  P=(27, 8),  K=(30, 8),  temperature=(31, 3),  humidity=(50, 8),  ph=(5.7, 0.4), rainfall=(95, 20)),
    "grapes":      dict(N=(23, 6),  P=(132,10), K=(200, 15),temperature=(24, 3),  humidity=(82, 5),  ph=(6.0, 0.4), rainfall=(70, 15)),
    "watermelon":  dict(N=(99, 12), P=(17, 5),  K=(50, 8),  temperature=(25, 3),  humidity=(85, 5),  ph=(6.5, 0.4), rainfall=(50, 12)),
    "muskmelon":   dict(N=(100,12), P=(18, 5),  K=(50, 8),  temperature=(28, 3),  humidity=(92, 3),  ph=(6.4, 0.4), rainfall=(24, 8)),
    "apple":       dict(N=(21, 6),  P=(134,10), K=(200, 15),temperature=(22, 3),  humidity=(92, 3),  ph=(5.9, 0.4), rainfall=(112, 20)),
    "orange":      dict(N=(19, 6),  P=(16, 5),  K=(10, 4),  temperature=(22, 3),  humidity=(92, 3),  ph=(7.0, 0.4), rainfall=(110, 20)),
    "papaya":      dict(N=(50, 10), P=(59, 10), K=(50, 8),  temperature=(33, 3),  humidity=(92, 3),  ph=(6.7, 0.4), rainfall=(142, 25)),
    "coconut":     dict(N=(22, 6),  P=(17, 5),  K=(31, 8),  temperature=(27, 2),  humidity=(94, 3),  ph=(5.9, 0.4), rainfall=(175, 25)),
    "cotton":      dict(N=(118,12), P=(46, 10), K=(20, 6),  temperature=(24, 3),  humidity=(80, 5),  ph=(6.9, 0.4), rainfall=(80, 20)),
    "jute":        dict(N=(78, 12), P=(47, 10), K=(40, 8),  temperature=(25, 2),  humidity=(80, 5),  ph=(6.7, 0.4), rainfall=(175, 25)),
    "coffee":      dict(N=(101,12), P=(28, 8),  K=(30, 8),  temperature=(25, 3),  humidity=(58, 8),  ph=(6.8, 0.4), rainfall=(159, 25)),
}

SAMPLES_PER_CROP = 100
rows = []

for crop, prof in CROP_PROFILES.items():
    for _ in range(SAMPLES_PER_CROP):
        row = {}
        for feat, (mean, std) in prof.items():
            val = np.random.normal(mean, std)
            if feat in ("N", "P", "K"):
                val = max(0, round(val))
            elif feat == "ph":
                val = round(np.clip(val, 3.5, 9.5), 2)
            elif feat == "humidity":
                val = round(np.clip(val, 10, 100), 2)
            elif feat == "rainfall":
                val = round(max(0, val), 2)
            elif feat == "temperature":
                val = round(val, 2)
            row[feat] = val
        row["label"] = crop
        rows.append(row)

df = pd.DataFrame(rows, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "Crop_recommendation.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} rows across {len(CROP_PROFILES)} crops -> {out_path}")
