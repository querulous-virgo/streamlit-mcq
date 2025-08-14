import streamlit as st
import base64
from together import Together
import io
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 OCR Text Extractor")
st.markdown("Upload an image to extract text using AI-powered OCR")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Together API Key", type="password", help="Enter your Together AI API key")
    
    if not api_key:
        st.warning("Please enter your Together API key to proceed")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image containing text to extract"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

def encode_image_from_bytes(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_text_from_image(api_key, image_bytes):
    """Extract text from image using Together API"""
    try:
        client = Together(api_key=api_key)
        
        getDescriptionPrompt = "Extract all the text in the image and write it. Ignore the webpages url and the buttons. Focus on the text"
        
        # Encode image
        base64_image = encode_image_from_bytes(image_bytes)
        
        # Make API call
        stream = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            temperature=0.2,  # Lower = more accurate, less hallucination
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR reader for coding platforms."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": getDescriptionPrompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            stream=True,
        )
        
        # Collect streaming response
        extracted_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                extracted_text += chunk.choices[0].delta.content
        
        return extracted_text, None
        
    except Exception as e:
        return None, str(e)

with col2:
    st.header("Extracted Text")
    
    if uploaded_file is not None and api_key:
        # Extract text button
        if st.button("🔍 Extract Text", type="primary"):
            with st.spinner("Extracting text from image..."):
                # Get image bytes
                image_bytes = uploaded_file.getvalue()
                
                # Extract text
                extracted_text, error = extract_text_from_image(api_key, image_bytes)
                
                if error:
                    st.error(f"Error extracting text: {error}")
                else:
                    # Store extracted text in session state
                    st.session_state.extracted_text = extracted_text
                    st.success("Text extracted successfully!")
    
    # Text editor
    if hasattr(st.session_state, 'extracted_text'):
        st.subheader("Edit Extracted Text")
        
        # Text area for editing
        edited_text = st.text_area(
            "You can edit the extracted text below:",
            value=st.session_state.extracted_text,
            height=300,
            help="Make any necessary corrections to the extracted text"
        )
        
        # Update session state with edited text
        st.session_state.edited_text = edited_text
        
        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📋 Copy to Clipboard"):
                # Note: This requires additional JavaScript, so we'll show the text to copy
                st.code(edited_text, language=None)
                st.info("Text displayed above for copying")
        
        with col_b:
            if st.button("📁 Download as TXT"):
                # Create download button
                st.download_button(
                    label="Download Text File",
                    data=edited_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
        
        with col_c:
            if st.button("🔄 Reset"):
                if hasattr(st.session_state, 'extracted_text'):
                    st.session_state.edited_text = st.session_state.extracted_text
                    st.rerun()

# Instructions
if not uploaded_file:
    st.info("👆 Upload an image to get started")
elif not api_key:
    st.warning("👈 Please enter your Together API key in the sidebar")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Together AI")

# Additional features section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Enter API Key**: Add your Together AI API key in the sidebar
    2. **Upload Image**: Choose an image file containing text you want to extract
    3. **Extract Text**: Click the 'Extract Text' button to process the image
    4. **Edit**: Make any necessary corrections in the text editor
    5. **Save**: Copy or download the final text
    
    **Supported formats**: PNG, JPG, JPEG, GIF, BMP
    """)

with st.expander("⚙️ API Configuration"):
    st.markdown("""
    This app uses the Together AI API with the following settings:
    - **Model**: meta-llama/Llama-Vision-Free
    - **Temperature**: 0.2 (for more accurate, less creative responses)
    - **Max Tokens**: 1024
    - **System Prompt**: Configured as an expert OCR reader
    """)
