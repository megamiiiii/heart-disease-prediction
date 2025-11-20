from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "ubah_dengan_rahasia_anda"

# ============================
# Load model (sesuai instruksi)
# ============================
model = pickle.load(open("model_hrflm.pkl", "rb"))
rf = model["rf"]
lr = model["lr"]
scaler = model["scaler"]

# Helper: rentang input dan info (dipakai di template)
INPUT_SPEC = [
    {"name": "age", "label": "Usia (tahun)", "min": 18, "max": 100, "step": 1, "type": "number"},
    {"name": "sex", "label": "Jenis Kelamin (0=Perempuan, 1=Laki-laki)", "min": 0, "max": 1, "step": 1, "type": "number"},
    {"name": "cp", "label": "Chest Pain Type (0–3)", "min": 0, "max": 3, "step": 1, "type": "number"},
    {"name": "trestbps", "label": "Tekanan Darah Istirahat (mm Hg)", "min": 80, "max": 220, "step": 1, "type": "number"},
    {"name": "chol", "label": "Kolesterol (mg/dl)", "min": 100, "max": 600, "step": 1, "type": "number"},
    {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl (0/1)", "min": 0, "max": 1, "step": 1, "type": "number"},
    {"name": "restecg", "label": "Resting ECG (0–2)", "min": 0, "max": 2, "step": 1, "type": "number"},
    {"name": "thalach", "label": "Maximum Heart Rate Achieved", "min": 60, "max": 220, "step": 1, "type": "number"},
    {"name": "exang", "label": "Exercise Induced Angina (0/1)", "min": 0, "max": 1, "step": 1, "type": "number"},
    {"name": "oldpeak", "label": "ST depression induced by exercise (oldpeak)", "min": 0.0, "max": 10.0, "step": 0.1, "type": "number"},
    {"name": "slope", "label": "Slope of peak exercise ST segment (0–2)", "min": 0, "max": 2, "step": 1, "type": "number"},
    {"name": "ca", "label": "Number of major vessels (0–3) colored by flourosopy", "min": 0, "max": 3, "step": 1, "type": "number"},
    {"name": "thal", "label": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", "min": 1, "max": 3, "step": 1, "type": "number"},
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", inputs=INPUT_SPEC)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil dan konversi input
        features = []
        for spec in INPUT_SPEC:
            key = spec["name"]
            val = request.form.get(key, "").strip()
            if val == "":
                flash(f"Field {spec['label']} wajib diisi.", "danger")
                return redirect(url_for("index"))
            # tipe numeric: float atau int
            if spec["type"] == "number":
                # allow float for oldpeak
                if "." in val or spec["step"] != 1:
                    v = float(val)
                else:
                    v = int(val)
            else:
                v = float(val)
            # opsi validasi rentang minimal
            if v < spec["min"] or v > spec["max"]:
                flash(f"Nilai {spec['label']} harus antara {spec['min']} dan {spec['max']}.", "danger")
                return redirect(url_for("index"))
            features.append(v)

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Prediksi probabilitas (RF + LR)
        try:
            prob_rf = rf.predict_proba(X_scaled)[0][1]
        except Exception:
            prob_rf = float(rf.predict(X_scaled)[0])
        try:
            prob_lr = lr.predict_proba(X_scaled)[0][1]
        except Exception:
            prob_lr = float(lr.predict(X_scaled)[0])

        # Ensemble sederhana: rata-rata probabilitas
        final_prob = float((prob_rf + prob_lr) / 2.0)
        threshold = 0.5
        prediction = 1 if final_prob >= threshold else 0

        # Interpretasi hasil untuk klinisi/pasien (ringkas & hati-hati)
        if prediction == 1:
            label = "Kemungkinan Penyakit Jantung: TINGGI"
            advice = ("Hasil menunjukkan kemungkinan penyakit jantung cukup tinggi. "
                      "Saran: pertimbangkan evaluasi lanjutan seperti EKG, ekokardiografi, "
                      "atau rujukan ke kardiolog. Keputusan akhir harus digabungkan dengan pemeriksaan klinis.")
        else:
            label = "Kemungkinan Penyakit Jantung: RENDAH"
            advice = ("Hasil menunjukkan kemungkinan penyakit jantung rendah menurut model. "
                      "Jika ada keluhan atau faktor risiko lain, lakukan tindak lanjut klinis sesuai protokol.")
        
        # kembalikan ke halaman hasil
        return render_template("result.html",
                               probability=round(final_prob, 4),
                               label=label,
                               advice=advice,
                               raw_rf=round(prob_rf,4),
                               raw_lr=round(prob_lr,4),
                               inputs=dict(zip([s["name"] for s in INPUT_SPEC], features)))
    except Exception as e:
        flash("Terjadi kesalahan pada server: " + str(e), "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
