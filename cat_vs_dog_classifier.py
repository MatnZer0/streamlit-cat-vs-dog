import time
from io import BytesIO
import traceback
from PIL import Image
import streamlit as st
import keras

# Increased file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Max dimensions for processing
MAX_IMAGE_SIZE = 2000  # pixels

model = keras.saving.load_model("cats-and-dogs.keras")

# Resize image while maintaining aspect ratio
def resize_image(image):
    # width, height = image.size
    # if width <= max_size and height <= max_size:
    #     return image
    
    # if width > height:
    #     new_width = max_size
    #     new_height = int(height * (max_size / width))
    # else:
    #     new_height = max_size
    #     new_width = int(width * (max_size / height))
    
    # return image.resize((new_width, new_height), Image.LANCZOS)
    image_size = (180, 180)
    return image.resize(image_size, Image.LANCZOS)

def process_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        # Resize large images to prevent memory issues
        resized = resize_image(image)
        return resized
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None
    
def run_model(image):
    img_array = keras.utils.img_to_array(image)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    st.success(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


def classify_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        # Read image bytes
        image_bytes = upload.getvalue()
        
        status_text.text("Processing image...")
        progress_bar.progress(30)
        
        # Process image
        image = process_image(image_bytes)
        if image is None:
            return
        
        progress_bar.progress(80)
        status_text.text("Displaying results...")

        run_model(image)
        
        # Display image
        st.image(image)
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        st.error("Failed to process image")
        st.error(f"An error occurred: {str(e)}")
        # Log the full error for debugging
        print(f"Error in classify_image: {traceback.format_exc()}")

st.write("## Classify a image by having a cat or dog")

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        # st.info("Processsing image")
        classify_image(my_upload)
else:
    st.info("Please upload an image to get started!")