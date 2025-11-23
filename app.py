import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io

# --------------------------
# Paths
# --------------------------
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\plant-app\plant_model (1).keras"
BG_PATH = "assets/bgimage.jpg"
IMG_SIZE = 224

# --------------------------
# Background helper
# --------------------------
def add_bg(local_image_path):
    try:
        with open(local_image_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .block-container {{
                background: rgba(0,0,0,0.60);
                padding: 25px;
                border-radius: 12px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        # if background fails, ignore silently
        pass

add_bg(BG_PATH)

# --------------------------
# Load model
# --------------------------
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False, compile=False)

# --------------------------
# Class labels (same order used during training)
# --------------------------
class_names = [
    'Pepper_bell__Bacterial_spot',
    'Pepper_bell__healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# --------------------------
# Utility: parse label -> (Plant, Disease)
# --------------------------
def split_prediction(label):
    # unify underscores, split, and build readable strings
    label = label.replace("__", "_")
    parts = label.split("_")
    if parts[-1].lower() == "healthy":
        plant = parts[0].replace("_", " ").capitalize()
        return plant, "Healthy"
    plant = parts[0].replace("_", " ").capitalize()
    disease = " ".join(parts[1:]).replace("_", " ").strip()
    return plant, disease.capitalize()

# --------------------------
# Treatment database (normalized keys)
# --------------------------
treatments = {
    "bacterial spot": [
        "Spray Copper Oxychloride 50% WP",
        "Use Streptocycline (100 ppm)",
        "Avoid overhead irrigation",
        "Remove infected leaves"
    ],
    "early blight": [
        "Spray Mancozeb 75% WP",
        "Use Chlorothalonil",
        "Improve air circulation"
    ],
    "late blight": [
        "Use Metalaxyl + Mancozeb (Ridomil Gold)",
        "Spray Dimethomorph",
        "Destroy infected plants"
    ],
    "leaf mold": [
        "Spray Copper Fungicide",
        "Improve airflow"
    ],
    "septoria leaf spot": [
        "Apply Mancozeb or Chlorothalonil",
        "Remove infected leaves"
    ],
    "spider mites two spotted spider mite": [
        "Spray Abamectin 1.9% EC",
        "Use Neem Oil 0.5%"
    ],
    "target spot": [
        "Spray Difenoconazole",
        "Remove infected leaves"
    ],
    "tomato yellowleaf curl virus": [
        "Use Imidacloprid to control whiteflies",
        "Remove infected plants immediately"
    ],
    "tomato mosaic virus": [
        "Remove infected plants",
        "Disinfect tools regularly"
    ],
    "healthy": [
        "No treatment needed ✔ The plant is healthy."
    ]
}
# ensure keys lowercased (defensive)
treatments = {k.lower(): v for k, v in treatments.items()}

# --------------------------
# Fertilizer suggestions
# --------------------------
fertilizer = {
    "Tomato": "Apply 15-15-15 NPK or organic compost every 2 weeks.",
    "Pepper": "Use 5-10-10 fertilizer every 15 days.",
    "Potato": "Apply 10-20-20 NPK at planting stage."
}

# --------------------------
# Health metric functions
# --------------------------
def water_stress_score(image: Image.Image) -> float:
    arr = np.array(image.resize((100, 100)))
    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    score = (np.mean(gray) / 255.0) * 100.0
    return round(float(score), 2)

def disease_severity(image: Image.Image) -> float:
    arr = np.array(image.resize((100, 100)))
    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    damaged = np.sum(gray < 100)
    severity = (damaged / gray.size) * 100.0
    return round(float(severity), 2)

# --------------------------
# PDF generator
# --------------------------
def generate_pdf(plant: str, disease: str, conf: float, treat_list: list) -> io.BytesIO:
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(595, 842))  # A4-ish
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "Plant Disease Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, 775, f"Plant: {plant}")
    c.drawString(40, 755, f"Disease: {disease}")
    c.drawString(40, 735, f"Confidence: {conf:.2f}%")
    c.drawString(40, 710, "Recommended Treatment:")
    y = 690
    for t in treat_list:
        c.drawString(60, y, f"- {t}")
        y -= 18
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌿 Smart Plant Disease Detection System")
st.write("Upload a leaf image (JPG/PNG) and click Predict.")

uploaded = st.file_uploader("📤 Upload leaf image", type=["jpg", "jpeg", "png"], key="upload_leaf")

if uploaded:
    # load and normalize image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=320)

    if st.button("🔍 Predict Disease"):
        # preprocess to model input
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized)  # shape (224,224,3)
        img_array = np.expand_dims(img_array, axis=0)  # shape (1,224,224,3)

        # model predict
        preds = model.predict(img_array)
        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds) * 100.0)
        label = class_names[idx]

        # parse label -> plant, disease
        plant_name, disease_name = split_prediction(label)

        # UI output
        st.subheader("🔍 Prediction Result")
        st.write("🌱 **Plant:**", plant_name)
        st.write("🦠 **Disease:**", disease_name)
        st.write("📊 **Confidence:**", f"{confidence:.2f}%")
        st.progress(min(int(confidence), 100))

        # health metrics
        sev = disease_severity(image)
        ws = water_stress_score(image)
        st.write(f"🔥 **Disease Severity:** {sev}%")
        st.write(f"💧 **Water Stress Score:** {ws}%")

        # treatment lookup (normalized)
        disease_key = disease_name.lower().strip()
        treat_list = treatments.get(disease_key)

        st.subheader("💊 Recommended Treatment")
        if treat_list:
            for t in treat_list:
                st.write("✔", t)
        else:
            # try fallback fuzzy matches (simple contains)
            matched = None
            for k in treatments.keys():
                if k in disease_key or disease_key in k:
                    matched = treatments[k]
                    break
            if matched:
                for t in matched:
                    st.write("✔", t)
                treat_list = matched
            else:
                st.write("⚠ No treatment data found for this exact label.")
                # still provide generic advice
                st.write("General advice: isolate affected plants, avoid overhead irrigation, maintain airflow, and disinfect tools.")
                treat_list = ["General advice: isolate affected plants, improve airflow, disinfect tools"]

        # fertilizer suggestion
        st.subheader("🌱 Fertilizer Recommendation")
        st.write(fertilizer.get(plant_name, "Use balanced NPK fertilizer."))

        # PDF download
        pdf = generate_pdf(plant_name, disease_name, confidence, treat_list)
        st.download_button("📥 Download Report as PDF", data=pdf, file_name="Plant_Report.pdf", mime="application/pdf")

