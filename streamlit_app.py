import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import re
from difflib import SequenceMatcher
import pandas as pd
import io
import matplotlib.pyplot as plt

# Minimal imports to reduce dependency issues
try:
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        AutoProcessor,
        AutoModelForVision2Seq,
        AutoTokenizer
    )
    transformers_available = True
except ImportError:
    transformers_available = False
    st.error("Transformers library not available. Installing required packages...")
    st.code("pip install transformers torch")

# Set page config
st.set_page_config(
    page_title="OCR Enhancement Demo",
    page_icon="📝",
    layout="wide"
)

# Constants
HF_TOKEN = "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"  # Using the token from your code
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar
st.sidebar.title("OCR Enhancement Demo")
st.sidebar.info("Upload an image to see how post-processing improves OCR results.")

# OCR Models selection (simplified)
ocr_models = st.sidebar.multiselect(
    "Select OCR Models",
    ["TrOCR", "PaLI-Gemma"],
    default=["TrOCR"]
)

# Image preprocessing options
st.sidebar.subheader("Image Preprocessing")
apply_deblur = st.sidebar.checkbox("Apply Deblurring", value=True)
apply_sharpen = st.sidebar.checkbox("Apply Sharpening", value=True)
apply_binarization = st.sidebar.checkbox("Apply Binarization", value=False)
apply_contrast = st.sidebar.checkbox("Enhance Contrast", value=False)

# Post-processing options
st.sidebar.subheader("Post-processing")
refine_with_pali = st.sidebar.checkbox("Refine with PaLI-Gemma", value=True)

# Main class with reduced functionality to ensure it works
class OCRSystem:
    def __init__(self, hf_token):
        self.device = DEVICE
        self.hf_token = hf_token
        self.models = {}
        
        # Progress indicator for model loading
        if transformers_available:
            with st.spinner("Loading OCR models... This may take a moment."):
                self._init_models()
        else:
            st.warning("Models cannot be loaded until transformers package is installed.")

    def _init_models(self):
        """Initialize selected OCR models"""
        # Initialize TrOCR if selected
        if "TrOCR" in ocr_models:
            try:
                self.models["TrOCR"] = {
                    "processor": TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed"),
                    "model": VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(self.device)
                }
                
                # Set TrOCR configuration
                model = self.models["TrOCR"]["model"]
                processor = self.models["TrOCR"]["processor"]
                
                model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
                model.config.pad_token_id = processor.tokenizer.pad_token_id
                model.config.vocab_size = model.config.decoder.vocab_size
                model.config.max_length = 64
                model.config.early_stopping = True
                model.config.no_repeat_ngram_size = 3
                model.config.length_penalty = 2.0
                model.config.num_beams = 4
                
                st.sidebar.success("✅ TrOCR model loaded")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load TrOCR: {str(e)}")
        
        # Initialize PaLI-Gemma if selected
        if "PaLI-Gemma" in ocr_models or refine_with_pali:
            try:
                model_name = "google/paligemma-3b-pt-224"
                self.models["PaLI-Gemma"] = {
                    "processor": AutoProcessor.from_pretrained(model_name, token=self.hf_token),
                    "model": AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        token=self.hf_token
                    ),
                    "tokenizer": AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
                }
                
                self.models["PaLI-Gemma"]["tokenizer"].pad_token = self.models["PaLI-Gemma"]["tokenizer"].eos_token
                st.sidebar.success("✅ PaLI-Gemma model loaded")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load PaLI-Gemma: {str(e)}")

    def normalize_text(self, text):
        """Normalize text for soft matching"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
        
    def calculate_similarity(self, text1, text2):
        """Calculate text similarity ratio"""
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()

    def preprocess_image(self, image):
        """Apply selected preprocessing techniques to the image"""
        # Convert to CV2 format if it's a PIL image
        if isinstance(image, Image.Image):
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv2 = image.copy()
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
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
        
        # Convert back to PIL and RGB
        pil_image = Image.fromarray(processed)
        rgb_image = Image.merge('RGB', [pil_image, pil_image, pil_image])
        
        return rgb_image, processed

    def process_with_trocr(self, image):
        """Process image with TrOCR model"""
        if "TrOCR" not in self.models:
            return "TrOCR model not loaded"
        
        processor = self.models["TrOCR"]["processor"]
        model = self.models["TrOCR"]["model"]
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return prediction

    def process_with_pali(self, image, task="read the text"):
        """Process image with PaLI-Gemma model"""
        if "PaLI-Gemma" not in self.models:
            return "PaLI-Gemma model not loaded"
        
        processor = self.models["PaLI-Gemma"]["processor"]
        model = self.models["PaLI-Gemma"]["model"]
        tokenizer = self.models["PaLI-Gemma"]["tokenizer"]
        
        prompt = f"{task}"
        
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the prediction
        prediction = prediction.replace(prompt, "").strip()
        
        return prediction

    def refine_prediction(self, image, initial_pred):
        """Refine the prediction using PaLI-Gemma"""
        if "PaLI-Gemma" not in self.models:
            return initial_pred + " (refinement failed - PaLI-Gemma not loaded)"
        
        prompt = f"Correct this OCR text from an image: '{initial_pred}'"
        return self.process_with_pali(image, prompt)

    def process_image(self, image):
        """Process a single image through the full pipeline with all selected models"""
        # Preprocess image
        processed_image, processed_gray = self.preprocess_image(image)
        
        results = {}
        
        # Process with each selected model
        if "TrOCR" in ocr_models and "TrOCR" in self.models:
            try:
                trocr_pred = self.process_with_trocr(processed_image)
                results["TrOCR"] = {
                    "raw": trocr_pred,
                    "refined": self.refine_prediction(processed_image, trocr_pred) if refine_with_pali else None
                }
            except Exception as e:
                results["TrOCR"] = {"raw": f"Error: {str(e)}", "refined": None}
        
        if "PaLI-Gemma" in ocr_models and "PaLI-Gemma" in self.models:
            try:
                pali_pred = self.process_with_pali(processed_image, "Read and transcribe all the text in this image:")
                results["PaLI-Gemma"] = {
                    "raw": pali_pred,
                    "refined": None  # PaLI doesn't need refinement with itself
                }
            except Exception as e:
                results["PaLI-Gemma"] = {"raw": f"Error: {str(e)}", "refined": None}
        
        return results, processed_image, processed_gray


# Let's create a simplified demo mode for when transformers aren't available
def demo_mode(image):
    """Run a demonstration with mock data when models can't be loaded"""
    # Apply basic preprocessing
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Apply simple preprocessing
    processed = gray_image.copy()
    
    if apply_deblur:
        blurred = cv2.GaussianBlur(processed, (0, 0), 3)
        processed = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)
        
    if apply_sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    # Convert back to PIL
    processed_image = Image.fromarray(processed)
    processed_rgb = Image.merge('RGB', [processed_image, processed_image, processed_image])
    
    # Mock results
    results = {
        "TrOCR": {
            "raw": "This is a simulated OCR result with potential errors.",
            "refined": "This is a simulated refined OCR result with corrections." if refine_with_pali else None
        }
    }
    
    if "PaLI-Gemma" in ocr_models:
        results["PaLI-Gemma"] = {
            "raw": "This is a simulated PaLI-Gemma result with improved recognition.",
            "refined": None
        }
    
    return results, processed_rgb, processed


# Main app
st.title("OCR Enhancement Demo")
st.write("""
This application demonstrates how post-processing can significantly improve OCR results. 
Upload an image containing text, and see the difference with and without our enhancement pipeline.
""")

# Installation instructions if transformers not available
if not transformers_available:
    st.warning("""
    The Transformers library is not installed. For full functionality, please install:
    ```
    pip install transformers torch
    ```
    Running in demo mode with simulated results.
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Process the image
    with st.spinner("Processing image..."):
        if transformers_available:
            # Initialize the OCR system if not already done
            if 'ocr_system' not in st.session_state:
                st.session_state.ocr_system = OCRSystem(HF_TOKEN)
            
            # Process with actual models
            results, processed_image, processed_gray = st.session_state.ocr_system.process_image(image)
        else:
            # Run demo mode with simulated results
            results, processed_image, processed_gray = demo_mode(image)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_column_width=True)
    
    # Display OCR results
    st.header("OCR Results")
    
    for model_name, model_results in results.items():
        st.subheader(f"{model_name} Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Without Post-Processing:**")
            st.text_area(f"{model_name} Raw Output", model_results["raw"], height=150)
        
        with col2:
            if model_results["refined"] is not None:
                st.write("**With Post-Processing:**")
                st.text_area(f"{model_name} Refined Output", model_results["refined"], height=150)
                
                if transformers_available:
                    # Calculate similarity for comparison
                    similarity = st.session_state.ocr_system.calculate_similarity(
                        model_results["raw"], 
                        model_results["refined"]
                    )
                    st.write(f"**Text Similarity:** {similarity:.2f}")
                else:
                    st.write("**Text Similarity:** 0.85 (demo)")
            else:
                st.write("**Post-Processing Not Applied**")
    
    # Tips section
    st.header("OCR Enhancement Tips")
    st.write("""
    1. **Image Quality**: Clear, high-resolution images generally produce better OCR results.
    2. **Preprocessing**: Deblurring and sharpening can significantly improve OCR accuracy for blurry images.
    3. **Post-processing**: Using language models to refine initial OCR results can correct common errors.
    4. **Model Selection**: Different OCR engines perform better on different types of text.
    """)

else:
    # Display sample images when no file is uploaded
    st.info("👆 Upload an image to start. Images with text at various angles, fonts, and backgrounds work best for demonstrating OCR enhancement.")
    
    # Sample section
    st.header("What This Demo Does")
    st.write("""
    This application demonstrates how post-processing can significantly improve OCR (Optical Character Recognition) results:
    
    1. **Image Processing**: Applies techniques like deblurring and sharpening to enhance image quality
    2. **OCR Models**: Uses state-of-the-art models like TrOCR and PaLI-Gemma (when available)
    3. **Post-Processing**: Refines initial OCR results using advanced language models
    4. **Comparison**: Shows results with and without enhancement for easy comparison
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Upload Image")
        st.write("Upload any image containing text")
    
    with col2:
        st.subheader("2. Process Image")
        st.write("Apply preprocessing and OCR")
    
    with col3:
        st.subheader("3. View Results")
        st.write("Compare results with and without post-processing")


# Requirements
st.sidebar.subheader("Requirements")
st.sidebar.code("""
streamlit
torch
transformers
numpy
opencv-python
pandas
pillow
""")

# Run instructions
st.sidebar.subheader("How to Run")
st.sidebar.code("streamlit run app.py")

# Add a FAQ section to help with troubleshooting
st.sidebar.subheader("Troubleshooting")
with st.sidebar.expander("Common Issues"):
    st.write("""
    - **ModuleNotFoundError**: Install missing packages with `pip install package-name`
    - **CUDA/GPU errors**: The app will fall back to CPU if GPU is not available
    - **Memory errors**: Reduce image size or close other applications
    - **Model download failures**: Check internet connection and try again
    """)
