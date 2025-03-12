# Install the Hugging Face diffusers library
!pip install diffusers transformers accelerate                                                                                                                                                                                                                            from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Check if GPU is available; otherwise, fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

# User-defined prompt input
prompt = input("Enter your prompt for image generation: ")

# Generate an image
image = pipeline(prompt).images[0]

# Save the image with prompt in the filename (replace spaces with underscores)
filename = f"generated_image_{prompt.replace(' ', '_')}.png"
image.save(filename)

print(f"Generated image for prompt '{prompt}' saved as '{filename}'")
