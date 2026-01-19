import streamlit as st
import tempfile
from ocr_engine import UltraPreciseOCR

st.set_page_config(page_title="Business Card OCR", layout="wide")
st.title("Business Card Text Extraction")

uploaded = st.file_uploader(
    "Upload a business card image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        image_path = tmp.name

    ocr = UltraPreciseOCR(gpu=False)

    original, processed = ocr.preprocess_image(image_path)
    results = ocr.extract_text(processed)
    unique = ocr.remove_duplicates(results)
    organized = ocr.organize(unique)
    detected = ocr.draw_boxes(original, unique)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_column_width=True)

    with col2:
        st.subheader("Detected Image")
        st.image(detected, use_column_width=True)

    st.subheader("Extracted Text")
    st.text(organized["exact_text"])

    st.subheader("Text with Confidence")
    st.table(organized["lines_with_confidence"])

    st.subheader("Categorized Information")

    category_table = []
    for category, values in organized["categorized"].items():
        for value in values:
            category_table.append({
                "Category": category,
                "Value": value
            })

    st.table(category_table)

else:
    st.info("Upload a business card image to start extraction")
