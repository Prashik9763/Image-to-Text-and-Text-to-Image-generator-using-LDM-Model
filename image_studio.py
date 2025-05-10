import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import time
import random
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="AI Image Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .image-container {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 10px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
    .caption-container {
        padding: 10px;
        background-color: #f1f8fe;
        border-radius: 5px;
        margin-top: 10px;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .prompt-suggestions {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f1f8fe;
        border-radius: 5px;
    }
    .tabs-container {
        margin-bottom: 20px;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎨 AI Image Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Generate and Caption Images with AI</p>", unsafe_allow_html=True)

# Initialize session state
if 'text_prompt' not in st.session_state:
    st.session_state.text_prompt = ""
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'selected_image_index' not in st.session_state:
    st.session_state.selected_image_index = None

# API Keys (hidden in code)
hf_api_key = "Your API Key"  # Hugging Face API key
gemini_api_key = "Your API Key"  # Gemini API key

# Sidebar for settings
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    
    # App mode selection (simplified to 2 options)
    app_mode = st.radio("Choose Mode", ["Generate Images", "Caption Images"])
        
    # Settings based on mode
    if app_mode == "Generate Images":
        st.markdown("### 🖼️ Image Generation Settings")
        
        # Image generation models
        MODELS = {
            "FLUX.1": "black-forest-labs/FLUX.1-dev",
            "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
            "Playground v2": "playgroundai/playground-v2-1024px-aesthetic"
        }
        selected_model = st.selectbox("Select Model", list(MODELS.keys()))
        
        # Image settings
        num_images = st.slider("Number of Images", 1, 4, 1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
            num_inference_steps = st.slider("Inference Steps", 20, 100, 50, 5)
        
        # Example prompts
        st.markdown("<div class='prompt-suggestions'>", unsafe_allow_html=True)
        st.markdown("### ✨ Prompt Ideas")
        example_prompts = [
            "A serene lake at sunset with mountains in the background",
            "Cyberpunk cityscape with neon lights and flying cars",
            "A magical forest with glowing mushrooms and fairy lights",
            "An astronaut riding a horse on Mars, digital art"
        ]
        
        for prompt in example_prompts:
            if st.button(f"{prompt[:25]}...", key=f"prompt_{example_prompts.index(prompt)}"):
                st.session_state.text_prompt = prompt
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Image Captioning Settings
    elif app_mode == "Caption Images":
        st.markdown("### 📝 Image Captioning Settings")
        
        # Caption style selection
        caption_style = st.selectbox(
            "Caption Style:",
            ["Detailed description", "Short caption", "Technical analysis", "Creative description"]
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to generate images using Hugging Face API
def generate_images_with_huggingface(text_prompt, model_name, num_images, guidance_scale, num_inference_steps):
    try:
        # Initialize InferenceClient with API key
        client = InferenceClient(
            provider="nebius",
            api_key=hf_api_key,
        )
        
        generated_images = []
        
        for i in range(num_images):
            # Add some randomness to each generation for variety
            seed = random.randint(1, 10000)
            
            # Generate image
            image = client.text_to_image(
                text_prompt,
                model=MODELS[model_name],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            generated_images.append(image)
        
        return generated_images, None
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Function to generate captions using Gemini API
def generate_caption_with_gemini(image, prompt_style):
    """Generate a caption for an image using Google Gemini."""
    try:
        if not gemini_api_key:
            return "Please enter your Gemini API key in the sidebar to generate captions."
        
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Set up the model - using Gemini 1.5 Flash which is the recommended replacement
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt based on selected style
        if prompt_style == "Detailed description":
            prompt = "Provide a detailed description of this image. Include all significant elements, colors, actions, and context."
        elif prompt_style == "Short caption":
            prompt = "Create a short, concise caption for this image in one sentence."
        elif prompt_style == "Technical analysis":
            prompt = "Analyze this image from a technical perspective. Describe composition, lighting, focus, and any technical elements."
        else:  # Creative description
            prompt = "Create a creative, engaging description of this image as if it were part of a story."
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        # Return the generated caption
        return response.text
        
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# Main content area based on selected mode
if app_mode == "Generate Images":
    # User Input for Text Prompt
    col1, col2 = st.columns([4, 1])
    with col1:
        text_prompt = st.text_input("Enter your prompt:", value=st.session_state.text_prompt, key="prompt_input")
    with col2:
        generate_button = st.button("Generate", use_container_width=True)
    
    # Store the prompt in session state when changed
    if text_prompt != st.session_state.text_prompt:
        st.session_state.text_prompt = text_prompt
    
    # Generate images when button is clicked or prompt is submitted
    if (generate_button or text_prompt) and text_prompt:
        with st.spinner(f"Creating your masterpiece with {selected_model}..."):
            images, error = generate_images_with_huggingface(
                text_prompt, 
                selected_model, 
                num_images, 
                guidance_scale, 
                num_inference_steps
            )
            
            if error:
                st.error(error)
            else:
                st.session_state.generated_images = images
                
                # Display images
                if num_images > 1:
                    cols = st.columns(min(num_images, 2))
                    for i, image in enumerate(images):
                        with cols[i % 2]:
                            st.markdown(f"<div class='image-container'>", unsafe_allow_html=True)
                            st.image(image, caption=f"Variation {i+1}", use_container_width=True)
                            
                            # Add download button for each image
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            st.download_button(
                                label="Download",
                                data=img_byte_arr.getvalue(),
                                file_name=f"ai_image_{int(time.time())}_{i}.png",
                                mime="image/png",
                                key=f"download_{i}"
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='image-container'>", unsafe_allow_html=True)
                    st.image(images[0], caption="Generated Image", use_container_width=True)
                    
                    # Add download button
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='PNG')
                    st.download_button(
                        label="Download Image",
                        data=img_byte_arr.getvalue(),
                        file_name=f"ai_image_{int(time.time())}.png",
                        mime="image/png"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display the prompt used
                st.markdown(f"**Prompt used:** {text_prompt}")

elif app_mode == "Caption Images":
    # Image upload
    st.markdown("### 📤 Upload an Image to Caption")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if API key is provided
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key in the sidebar to generate captions.")
        else:
            # Add a generate button
            if st.button("Generate Caption", key="generate_caption"):
                # Generate caption
                with st.spinner("Generating caption with Gemini..."):
                    caption = generate_caption_with_gemini(image, caption_style)
                
                # Display the caption
                st.markdown("<div class='caption-container'>", unsafe_allow_html=True)
                st.markdown("### 🔍 Generated Caption:")
                st.markdown(f"<p style='font-size: 1.2rem; padding: 10px;'>{caption}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # If the caption doesn't contain an error message, show the download button
                if not caption.startswith("Error") and not caption.startswith("Please enter"):
                    # Add download button for the caption
                    caption_text = f"Caption for uploaded image:\n{caption}\n\nGenerated using Google Gemini with style: {caption_style}"
                    st.download_button(
                        label="Download Caption",
                        data=caption_text,
                        file_name="image_caption.txt",
                        mime="text/plain",
                        key="download_caption"
                    )
    else:
        st.info("👆 Upload an image to generate a caption")

# Removed the Generate & Caption mode as requested

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    AI Image Studio | Powered by Hugging Face & Google Gemini
</div>
""", unsafe_allow_html=True)
