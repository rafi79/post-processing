import streamlit as st
import numpy as np
from io import BytesIO
import base64
import re
from difflib import SequenceMatcher
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io

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
apply_sharpen = st.sidebar.checkbox("Apply Sharpening", value=True)
apply_contrast = st.sidebar.checkbox("Enhance Contrast", value=True)
apply_brightness = st.sidebar.checkbox("Adjust Brightness", value=False)

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

def preprocess_image(image):
    """Apply preprocessing to PIL image"""
    # Convert to grayscale
    processed_image = ImageOps.grayscale(image)
    
    # Apply preprocessing based on user selection
    if apply_sharpen:
        # Apply sharpening filter
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
        processed_image = processed_image.filter(ImageFilter.SHARPEN)  # Apply twice for stronger effect
    
    if apply_contrast:
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)
    
    if apply_brightness:
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(1.2)
    
    return processed_image

def get_image_download_link(img, filename="processed_image.jpg", text="Download Processed Image"):
    """Generate a download link for a PIL image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def simulate_ocr_results(img_complexity):
    """Simulate OCR results based on image complexity"""
    # Raw OCR simulation with common errors
    raw_texts = {
        "low": "This is a sirnple text with sorne errors.",
        "medium": "This docurnent contains information about the OCR systern and how it can be irnproved using post-processing techniques.",
        "high": "The Enhanced OCR Systern uses TrOCR and PaLI-Gernrna to recognize text in irnages with varying degrees of cornplexity and quality."
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
Use one of our examples or upload your own image to see the simulated results.
""")

# Image source selection
image_source = st.radio(
    "Select image source:",
    ["Use example image", "Upload your own"]
)

image = None
image_complexity = "medium"  # Default complexity

if image_source == "Upload your own":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Read the image as a PIL Image
        image = Image.open(uploaded_file).convert("RGB")
        image_complexity = "medium"  # Default complexity for uploaded images
        
else:  # Use example image
    example_option = st.selectbox(
        "Select example image:",
        ["Clear printed text", "Blurry text", "Handwritten text"]
    )
    
    # Create example images using PIL
    width, height = 500, 300
    
    if example_option == "Clear printed text":
        # Create a blank white image
        image = Image.new('RGB', (width, height), color='white')
        # We can't add text easily with PIL alone, so we'll just use a blank image
        st.info("This is a simulated clear printed text image (imagine clear, crisp text here)")
        image_complexity = "low"
    elif example_option == "Blurry text":
        # Create a blank white image and blur it
        image = Image.new('RGB', (width, height), color='white')
        image = image.filter(ImageFilter.BLUR)
        st.info("This is a simulated blurry text image (imagine blurry text here)")
        image_complexity = "medium"
    else:  # Handwritten text
        # Create a blank white image
        image = Image.new('RGB', (width, height), color='white')
        st.info("This is a simulated handwritten text image (imagine handwritten text here)")
        image_complexity = "high"

# Process the image if we have one
if image is not None:
    # Process the image
    processed_image = preprocess_image(image)
    
    # Display images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_column_width=True)
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
                "Context": raw_ocr[max(0, i-5):min(len(raw_ocr), i+6)],
                "Error": raw_char,
                "Correction": refined_char
            })
    
    if errors:
        st.write("**Detected Errors:**")
        
        # Create a simple table using markdown
        st.markdown("| Position | Context | Error | Correction |")
        st.markdown("|----------|---------|-------|------------|")
        
        for error in errors:
            context = error["Context"]
            highlighted = context[:5] + "**" + context[5] + "**" + context[6:]
            st.markdown(f"| {error['Position']} | {highlighted} | {error['Error']} | {error['Correction']} |")
    else:
        st.write("No errors detected between raw and refined outputs.")

else:
    # Instructions when no image is selected
    st.info("Please select an example image or upload your own to see the OCR enhancement demo.")

# Tips section
st.header("OCR Enhancement Tips")
st.write("""
1. **Image Quality**: Clear, high-resolution images generally produce better OCR results.
2. **Preprocessing**: Sharpening and contrast enhancement can significantly improve OCR accuracy for blurry images.
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
       - Sharpening enhances text edges
       - Contrast enhancement improves readability
       - Brightness adjustment helps with dark or light images
    
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

The demo shows how post-processing can improve OCR results by:
1. Correcting common character confusions (e.g., 'rn' vs 'm')
2. Fixing contextual errors based on surrounding words
3. Applying language model knowledge to improve accuracy
""")

# Add requirements at the bottom
st.sidebar.subheader("Requirements")
st.sidebar.code("""
streamlit>=1.22.0
numpy>=1.20.0
pillow>=9.0.0
""")
