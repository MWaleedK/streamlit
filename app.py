import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained Generator (after training the model above)
generator = tf.keras.models.load_model('generator_model.h5')  # Replace with your saved generator model

# Function to generate 5 images based on selected digit
def generate_images(digit, num_images=5):
    latent_dim = 100
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    
    return generated_images

# Streamlit UI
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) to generate 5 images of that digit.")

# Digit selection
digit = st.selectbox("Select Digit", range(10))

# Generate and display 5 images of the selected digit
if st.button("Generate Images"):
    generated_images = generate_images(digit)

    # Display the generated images
    for i, image in enumerate(generated_images):
        plt.figure(figsize=(2, 2))
        plt.imshow(image.squeeze(), cmap="gray")
        plt.axis("off")
        st.pyplot(plt)
