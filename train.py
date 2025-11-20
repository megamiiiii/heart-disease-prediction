# ======================================================
# TRAINING MODEL HRFLM (RandomForest + LogisticRegression)
# Versi BEBAS ERROR (tanpa class RFProgress)
# ======================================================

import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ----------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------
print("\n[1/10] 📥 Mengambil dataset dari UCI...")
heart = fetch_ucirepo(id=45)

X = heart.data.features
y = heart.data.targets

df = pd.concat([X, y], axis=1)
print("   ➤ Dataset berhasil dimuat.")

print("\n============================================")
print("📌 INFORMASI DATASET")
print("============================================")
print("Jumlah data awal:", df.shape[0])
print("Jumlah fitur:", df.shape[1] - 1)
print("Jumlah missing value:", df.isna().sum().sum())

# ----------------------------------------------------
# 2. DROP MISSING VALUES
# ----------------------------------------------------
print("\n[2/10] 🧽 Menghapus missing values...")
df = df.dropna()
print("   ➤ Selesai. Jumlah data sekarang:", df.shape[0])

# ----------------------------------------------------
# 3. KONVERSI TARGET
# ----------------------------------------------------
print("\n[3/10] 🔄 Mengonversi label target (1–4 → 1)...")
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
print("   ➤ Konversi selesai.")

X = df.drop("num", axis=1)
y = df["num"]

print("\n============================================")
print("📌 INFORMASI LABEL")
print("============================================")
print("Kelas 0 (sehat)    :", (y == 0).sum())
print("Kelas 1 (penyakit) :", (y == 1).sum())

# ----------------------------------------------------
# 4. SPLIT DATA
# ----------------------------------------------------
print("\n[4/10] ✂️ Melakukan train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# ----------------------------------------------------
# 5. SCALING
# ----------------------------------------------------
print("\n[5/10] 📏 Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 6. TRAIN RANDOM FOREST (TANPA CUSTOM CLASS)
# ----------------------------------------------------
print("\n[6/10] 🌲 Melatih RandomForest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

start_time = time.time()
rf.fit(X_train, y_train)
print(f"   ✔ RF selesai dilatih ({time.time() - start_time:.2f} detik)")

# ----------------------------------------------------
# 7. TRAIN LOGISTIC REGRESSION
# ----------------------------------------------------
print("\n[7/10] 🧪 Melatih Logistic Regression...")

lr = LogisticRegression(max_iter=2000)

start_time = time.time()
lr.fit(X_train_scaled, y_train)
print(f"   ✔ LR selesai dilatih ({time.time() - start_time:.2f} detik)")

# ----------------------------------------------------
# 8. HYBRID MODEL
# ----------------------------------------------------
print("\n[8/10] 🔗 Menggabungkan RF + LR...")

pred_rf = rf.predict_proba(X_test)[:, 1]
pred_lr = lr.predict_proba(X_test_scaled)[:, 1]

final_prob = (pred_rf + pred_lr) / 2
final_pred = (final_prob >= 0.5).astype(int)

# ----------------------------------------------------
# 9. EVALUASI
# ----------------------------------------------------
print("\n[9/10] 📊 Menghitung metrik evaluasi...")

acc = accuracy_score(y_test, final_pred)
class_error = 1 - acc
prec = precision_score(y_test, final_pred)
rec = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

cm = confusion_matrix(y_test, final_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n============================================")
print("📌 HASIL EVALUASI HRFLM")
print("============================================")
print("Accuracy            :", round(acc * 100, 2), "%")
print("Classification Err. :", round(class_error * 100, 2), "%")
print("Precision           :", round(prec * 100, 2), "%")
print("F-measure           :", round(f1 * 100, 2), "%")
print("Sensitivity (TPR)   :", round(sensitivity * 100, 2), "%")
print("Specificity (TNR)   :", round(specificity * 100, 2), "%")
print("\nConfusion Matrix:\n", cm)

# ----------------------------------------------------
# 10. SAVE MODEL – AMAN UNTUK FLASK
# ----------------------------------------------------
print("\n[10/10] 💾 Menyimpan model (AMAN untuk Flask)...")

model = {
    "rf": rf,
    "lr": lr,
    "scaler": scaler
}

with open("model_hrflm.pkl", "wb") as f:
    pickle.dump(model, f)

print("   ✔ Model berhasil disimpan!")
print("\n============================================")
print("🚀 Training HRFLM selesai!")
print("============================================")
