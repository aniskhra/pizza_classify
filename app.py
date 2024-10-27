import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page title and icon
st.set_page_config(page_title="Pizza Classification", page_icon="🍕")

# --- CSS ---
st.markdown(
    """
    <style>
    .stHeader {
        background-color: #ffffff; 
        padding: 1rem;
        border-bottom: 1px solid;
    }
    .stTitle {
        text-align: center;
        font-size: 3rem;
        font-weight: 600; 
        margin-bottom: 0;
    }
    .stMarkdown {
        font-size: 1.1rem;
        line-height: 1.6; 
    }
    .stImage {
        border: 1px solid #e0e0e0;
        border-radius: 8px; 
        box-shadow: 2px 2px 5px; 
    }
    .stSuccess {
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px; 
    }
    /* Style for tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 4rem; /* Adjust the font size as needed */
        padding: 1rem 1.5rem; 
    }
    .stTabs [aria-selected="true"] { 
        border-bottom: 2px solid #4a90e2; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load model ---
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    model.fuse()  # Prepare model for inference
    return model

# --- Classify pizza ---
def classify_pizza(image, model):
    results = model.predict(image)
    for result in results:
        probs = result.probs.data.tolist()
        predicted_class = result.names[np.argmax(probs)]
        return predicted_class
    
# --- Function to display pizza description ---
def display_pizza_description(pizza_type):
    if pizza_type == "Pepperoni_Pizza":
        st.markdown("## Pepperoni Pizza")
        st.write(
            """
            **Asal:** Amerika Serikat (Awal abad ke-20)\n
            **Bahan-bahan:** Saus tomat, keju, pepperoni.\n
            **Cara memasak:**
            1. Panaskan oven hingga 450 ° F (230 ° C).
            2. Gulung adonan pizza dan letakkan di atas loyang.
            3. Oleskan saus tomat secara merata di atas adonan.
            4. Taburi dengan keju.
            5. Susun irisan pepperoni di atasnya.
            6. Panggang selama 12-15 menit, atau sampai adonan berwarna cokelat keemasan dan keju meleleh dan berbuih.
            """
        )
        
    elif pizza_type == "Margherita_Pizza":
        st.markdown("## Margherita Pizza")
        st.write(
            """
            **Asal:** Napoli, Italia (1889)\n
            **Bahan:** Saus tomat, keju mozzarella, daun basil segar, minyak zaitun.\n
            **Persiapan:**
            1. Panaskan oven hingga 450 ° F (230 ° C).
            2. Gulung adonan pizza dan letakkan di atas loyang.
            3. Oleskan saus tomat secara merata di atas adonan.
            4. Taburi dengan keju mozzarella.
            5. Susun daun basil segar di atasnya.
            6. Hujani dengan minyak zaitun.
            7. Panggang selama 12-15 menit, atau sampai adonan berwarna cokelat keemasan dan keju meleleh dan berbuih.
            """
        )
    
    elif pizza_type == "Hawaiian_Pizza":
        st.markdown("## Hawaiian Pizza")
        st.write(
            """
            **Asal:** Kanada (1962) oleh Sam Panopoulos\n
            **Bahan-bahan:** Saus tomat, keju, nanas, ham, atau bacon Kanada.\n
            **Cara memasak:**
            1. Panaskan oven hingga 450 ° F (230 ° C).
            2. Gulung adonan pizza dan letakkan di atas loyang.
            3. Oleskan saus tomat secara merata di atas adonan.
            4. Taburi dengan keju.
            5. Susun nanas dan ham/bacon Kanada di atasnya.
            6. Panggang selama 12-15 menit, atau sampai adonan berwarna cokelat keemasan dan keju meleleh dan berbuih.
            """
        )

# --- Main app ---
def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Home", "About"])

    # --- Home Page ---
    with tab1:
        st.title("🍕 Pizza Classification")
        st.write("Unggah gambar pizza untuk mengklasifikasikannya.")

        model_path = "runs/classify/train/weights/best.pt"  # Ganti dengan path model Anda
        model = load_model(model_path)

        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            predicted_class = classify_pizza(image, model)
            st.success(f"**Prediksi:** {predicted_class}")

            # Display pizza description
            display_pizza_description(predicted_class)

    # --- About Page ---
    with tab2:
        st.title("📋 About Pizza Classification")
        st.write("Aplikasi ini menggunakan model YOLOv8 untuk mengklasifikasikan gambar pizza menjadi tiga kategori: Pepperoni Pizza, Margherita Pizza, dan Hawaiian Pizza.")
        st.write("Aplikasi ini dibuat menggunakan Streamlit, kerangka kerja Python untuk membuat aplikasi web interaktif.")

if __name__ == "__main__":
    main()