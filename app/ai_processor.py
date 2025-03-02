import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
from .config import Config

class StoryProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        
    def process_images(self, image_paths, style):
        processed_images = []
        style_config = Config.STYLE_CONFIGS[style]
        
        for img_path in image_paths:
            # Load and preprocess image
            img = Image.open(img_path)
            img = self.preprocess_image(img)
            
            # Apply style transfer
            styled_img = self.apply_style_transfer(
                img, 
                style_config['prompt'],
                style_config['negative_prompt']
            )
            
            processed_images.append(styled_img)
            
        return processed_images
    
    def preprocess_image(self, image):
        # Resize and normalize image
        image = image.resize((512, 512))
        return image
    
    def apply_style_transfer(self, image, prompt, negative_prompt):
        # Generate styled image using Stable Diffusion
        result = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        return result