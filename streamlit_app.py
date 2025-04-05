import streamlit as st
import numpy as np
import cv2
import io
import base64
from io import BytesIO
import re
from difflib import SequenceMatcher

# Set page config
st.set_page_config(
    page_title="OCR Enhancement Demo",
    page_icon="📝",
    layout="wide"
)

# Constants
HF_TOKEN = "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"

# Sidebar
st.sidebar.title("OCR Enhancement Demo")
st.sidebar.info("This demo simulates the improvements from applying post-processing to OCR results.")

# Image preprocessing options
st.sidebar.subheader("Image Preprocessing")
apply_deblur = st.sidebar.checkbox("Apply Deblurring", value=True)
apply_sharpen = st.sidebar.checkbox("Apply Sharpening", value=True)
apply_binarization = st.sidebar.checkbox("Apply Binarization", value=False)
apply_contrast = st.sidebar.checkbox("Enhance Contrast", value=False)

# Helper functions
def normalize_text(text):
    """Normalize text for soft matching"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text
    
def calculate_similarity(text1, text2):
    """Calculate text similarity ratio"""
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()

def preprocess_image(image_array):
    """Apply preprocessing to image"""
    # Convert to grayscale if it's a color image
    if len(image_array.shape) > 2 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_array
    
    # Apply preprocessing based on user selection
    processed = gray_image.copy()
    
    if apply_deblur:
        blurred = cv2.GaussianBlur(processed, (0, 0), 3)
        processed = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)
        
    if apply_sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
        
    if apply_contrast:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)
        
    if apply_binarization:
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    return processed

def get_image_download_link(img, filename="processed_image.jpg", text="Download Processed Image"):
    """Generate a download link for an image"""
    buffered = BytesIO()
    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode()
    href = f'<a href="data:image/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def simulate_ocr_results(img_complexity):
    """Simulate OCR results based on image complexity"""
    # Raw OCR simulation with common errors
    raw_texts = {
        "low": "This is a sirnple text with sorne errors.",
        "medium": "This docurnent contains information about the OCR systern and how it can be irnproved using post-processing techniques.",
        "high": "The Enhanced OCR Systern uses TrOCR and PaLI-Gemrna to recognize text in irnages with varying degrees of cornplexity and quality."
    }
    
    # Refined OCR simulation with corrections
    refined_texts = {
        "low": "This is a simple text with some errors.",
        "medium": "This document contains information about the OCR system and how it can be improved using post-processing techniques.",
        "high": "The Enhanced OCR System uses TrOCR and PaLI-Gemma to recognize text in images with varying degrees of complexity and quality."
    }
    
    return raw_texts[img_complexity], refined_texts[img_complexity]


# Main app
st.title("OCR Enhancement Demo")
st.write("""
This application demonstrates how post-processing can significantly improve OCR results.
Upload an image or use one of our examples to see the simulated results.
""")

# Image source selection
image_source = st.radio(
    "Select image source:",
    ["Upload your own", "Use example image"]
)

image_array = None

if image_source == "Upload your own":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Read the image as a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_complexity = "medium"  # Default complexity for uploaded images
        
else:  # Use example image
    example_option = st.selectbox(
        "Select example image:",
        ["Clear printed text", "Blurry text", "Handwritten text"]
    )
    
    # Provide example images (these are just blank images with different dimensions for the demo)
    if example_option == "Clear printed text":
        image_array = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Add some sample text to the image
        cv2.putText(image_array, "This is clear printed text", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image_complexity = "low"
    elif example_option == "Blurry text":
        image_array = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Add some sample text and blur it
        cv2.putText(image_array, "This is blurry text", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
        image_complexity = "medium"
    else:  # Handwritten text
        image_array = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Add some sample "handwritten-like" text (just using a different font)
        cv2.putText(image_array, "Handwritten text sample", (50, 150), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)
        image_complexity = "high"

# Process the image if we have one
if image_array is not None:
    # Process the image
    processed_image = preprocess_image(image_array)
    
    # Display images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_column_width=True, channels="GRAY")
        st.markdown(get_image_download_link(processed_image), unsafe_allow_html=True)
    
    # Get simulated OCR results
    raw_ocr, refined_ocr = simulate_ocr_results(image_complexity)
    
    # Display OCR results
    st.header("Simulated OCR Results")
    
    # TrOCR Results
    st.subheader("TrOCR Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Without Post-Processing:**")
        st.text_area("Raw OCR Output", raw_ocr, height=100)
    
    with col2:
        st.write("**With Post-Processing:**")
        st.text_area("Refined OCR Output", refined_ocr, height=100)
        
        similarity = calculate_similarity(raw_ocr, refined_ocr)
        st.write(f"**Text Similarity:** {similarity:.2f}")
    
    # PaLI-Gemma Results
    st.subheader("PaLI-Gemma Results")
    
    # Simulate better base results for PaLI
    pali_raw = raw_ocr.replace("sirnple", "simple").replace("docurnent", "document")
    pali_refined = refined_ocr
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Without Post-Processing:**")
        st.text_area("PaLI Raw Output", pali_raw, height=100)
    
    with col2:
        st.write("**With Post-Processing:**")
        st.text_area("PaLI Refined Output", pali_refined, height=100)
        
        similarity = calculate_similarity(pali_raw, pali_refined)
        st.write(f"**Text Similarity:** {similarity:.2f}")
    
    # Comparison section
    st.header("Error Analysis")
    
    # Create a simple table showing the errors and corrections
    errors = []
    
    for i, (raw_char, refined_char) in enumerate(zip(raw_ocr, refined_ocr)):
        if raw_char != refined_char:
            errors.append({
                "Position": i,
                "Original": raw_ocr[max(0, i-5):min(len(raw_ocr), i+6)],
                "Error": raw_char,
                "Correction": refined_char
            })
    
    if errors:
        st.write("**Detected Errors:**")
        
        # Create a custom HTML table for better visibility
        error_html = "<table style='width:100%'><tr><th>Position</th><th>Context</th><th>Error</th><th>Correction</th></tr>"
        
        for error in errors:
            # Highlight the error character in the context
            context = error["Original"]
            highlighted_context = context[:5] + f"<mark>{context[5]}</mark>" + context[6:]
            
            error_html += f"<tr><td>{error['Position']}</td><td>{highlighted_context}</td><td>{error['Error']}</td><td>{error['Correction']}</td></tr>"
        
        error_html += "</table>"
        
        st.markdown(error_html, unsafe_allow_html=True)
    else:
        st.write("No errors detected between raw and refined outputs.")

else:
    # Instructions when no image is selected
    st.info("Please upload an image or select an example to see the OCR enhancement demo.")

# Tips section
st.header("OCR Enhancement Tips")
st.write("""
1. **Image Quality**: Clear, high-resolution images generally produce better OCR results.
2. **Preprocessing**: Deblurring and sharpening can significantly improve OCR accuracy for blurry images.
3. **Post-processing**: Using language models to refine initial OCR results can correct common errors.
4. **Common Error Patterns**: OCR engines often confuse similar-looking characters:
   - 'm' vs 'rn' (as in 'modern' vs 'modem')
   - '0' vs 'O' (zero vs capital O)
   - 'l' vs 'I' (lowercase L vs capital i)
""")

# How it works section
with st.expander("How Post-Processing Works"):
    st.write("""
    ### The Enhanced OCR Pipeline
    
    1. **Image Preprocessing**:
       - Deblurring removes blur from images
       - Sharpening enhances text edges
       - Binarization converts the image to black and white
       - Contrast enhancement improves readability
    
    2. **Base OCR**:
       - TrOCR: A transformer-based OCR model specialized for printed text
       - PaLI-Gemma: A vision-language model that can understand images and text
    
    3. **Post-Processing**:
       - Language model-based refinement corrects common OCR errors
       - Context-aware corrections improve accuracy
       - Grammar and spelling corrections for final output
    
    This approach significantly improves OCR accuracy, especially for challenging images with low quality, unusual fonts, or poor lighting.
    """)

# Sidebar - additional info
st.sidebar.subheader("About this Demo")
st.sidebar.info("""
This is a simplified simulation of an OCR enhancement system. In a real implementation, we would use:

1. TrOCR for base text recognition
2. PaLI-Gemma for advanced recognition and refinement
3. Custom post-processing with contextual fixes

For full functionality, you would need:
```
transformers
torch
opencv-python
streamlit
```
""")
