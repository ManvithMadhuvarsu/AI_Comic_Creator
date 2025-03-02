from flask import Flask, render_template, request, send_file, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from fpdf import FPDF
import random
import time
import cv2
import numpy as np

# Flask app setup
app = Flask(__name__, 
           template_folder='../templates',  # Point to templates directory
           static_folder='../static')       # Point to static directory

# Configuration
UPLOAD_FOLDER = '../uploads'
OUTPUT_FOLDER = '../static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Story templates
STORY_TEMPLATES = {
    'Manga': [
        "As {name} walked home from school, a mysterious light enveloped them, "
        "awakening hidden powers that would change their life forever...",
        "In the prestigious martial arts academy, {name} discovers an ancient "
        "technique that sets them apart from all other students..."
    ],
    'Comic': [
        "When {name} witnesses a strange meteor crash, they gain extraordinary "
        "abilities. Now they must learn to use these powers for good...",
        "By day, {name} leads a normal life, but when danger threatens the city, "
        "they transform into a powerful defender of justice..."
    ],
    'Animated Story': [
        "On their birthday, {name} receives a magical gift that opens the door "
        "to a world of endless adventures and new friendships...",
        "During a school field trip, {name} stumbles upon an enchanted artifact "
        "that grants them the ability to communicate with magical creatures..."
    ]
}

# Style configurations
STYLE_CONFIGS = {
    'Manga': {
        'prompt': "manga style character, same person, detailed face, maintaining facial features, "
                 "Japanese anime protagonist, dynamic pose, expressive eyes, "
                 "black and white manga illustration, detailed line art, clean linework",
        'negative_prompt': "different face, multiple people, deformed face, bad anatomy, "
                         "extra limbs, blurry, low quality, watermark, signature",
        'parameters': {
            'num_inference_steps': 75,
            'guidance_scale': 7.5,
            'strength': 0.65  # Lower strength to preserve more of original image
        }
    },
    'Comic': {
        'prompt': "comic book superhero portrait, same person, maintaining facial features, "
                 "dramatic superhero pose, detailed face, comic book style, "
                 "professional illustration, bold colors, dynamic lighting",
        'negative_prompt': "different person, multiple people, deformed, bad anatomy, "
                         "blurry, low quality, watermark, signature",
        'parameters': {
            'num_inference_steps': 75,
            'guidance_scale': 7.5,
            'strength': 0.7
        }
    },
    'Animated Story': {
        'prompt': "Pixar style 3D character, same person, maintaining facial features, "
                 "animated movie protagonist, expressive face, high quality 3D render, "
                 "professional character design, family friendly",
        'negative_prompt': "different person, multiple people, deformed, unrealistic, "
                         "bad anatomy, blurry, low quality, 2d animation",
        'parameters': {
            'num_inference_steps': 75,
            'guidance_scale': 7.5,
            'strength': 0.68
        }
    }
}

class StoryProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        self.pipe.enable_attention_slicing()
        self.pipe.to(self.device)
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def process_images(self, image_paths, style):
        processed_images = []
        style_config = STYLE_CONFIGS[style]
        
        # First image should be a clear face shot
        reference_face = self.extract_face_features(image_paths[0])
        
        for idx, img_path in enumerate(image_paths):
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                
                # Different processing for first (reference) image vs action shots
                if idx == 0:
                    # For reference image, focus on face
                    img = self.preprocess_portrait(img)
                else:
                    # For action shots, prepare full body composition
                    img = self.preprocess_action_shot(img)
                
                # Enhance prompt with reference face features
                enhanced_prompt = self.enhance_prompt_with_features(
                    style_config['prompt'], 
                    reference_face
                )
                
                # Apply style transfer
                with torch.inference_mode():
                    styled_img = self.apply_style_transfer(
                        img,
                        enhanced_prompt,
                        style_config['negative_prompt'],
                        style_config['parameters']
                    )
                
                # Post-process image based on style
                styled_img = self.postprocess_image(styled_img, style)
                processed_images.append(styled_img)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                
        return processed_images
    
    def extract_face_features(self, image_path):
        # Convert PIL to CV2 format
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face = img[y:y+h, x:x+w]
            # Extract features (you could add more sophisticated feature extraction here)
            return {
                'face_shape': self.detect_face_shape(face),
                'skin_tone': self.detect_skin_tone(face),
                'hair_color': self.detect_hair_color(face)
            }
        return {}
    
    def enhance_prompt_with_features(self, base_prompt, face_features):
        if not face_features:
            return base_prompt
            
        # Add detected features to prompt
        feature_prompt = f", {face_features.get('face_shape', '')} face"
        feature_prompt += f", {face_features.get('skin_tone', '')} skin tone"
        feature_prompt += f", {face_features.get('hair_color', '')} hair"
        
        return base_prompt + feature_prompt
    
    def preprocess_portrait(self, image):
        # Focus on face for reference shot
        target_size = (512, 512)
        return self.center_and_crop_face(image, target_size)
    
    def preprocess_action_shot(self, image):
        # Prepare image for action poses
        target_size = (512, 512)
        return self.maintain_aspect_ratio_resize(image, target_size)
    
    def center_and_crop_face(self, image, target_size):
        # Convert PIL to CV2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            # Add padding around face
            padding = int(w * 0.5)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_cv.shape[1] - x, w + 2 * padding)
            h = min(img_cv.shape[0] - y, h + 2 * padding)
            
            face = img_cv[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            return face_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def apply_style_transfer(self, image, prompt, negative_prompt, parameters):
        try:
            result = self.pipe(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=parameters['num_inference_steps'],
                guidance_scale=parameters['guidance_scale'],
                strength=parameters['strength']
            ).images[0]
            return result
        except Exception as e:
            print(f"Style transfer error: {str(e)}")
            return image
            
    def postprocess_image(self, image, style):
        if style == 'Manga':
            # Convert to black and white for manga style
            return image.convert('L').convert('RGB')
        return image

class StoryBookGenerator:
    def create_storybook(self, images, story_text, style):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add cover page
        self._add_cover_page(pdf, style, story_text)
        
        # Add story pages
        self._add_story_pages(pdf, images)
        
        # Save PDF
        output_path = os.path.join(OUTPUT_FOLDER, f'storybook_{int(time.time())}.pdf')
        pdf.output(output_path)
        return output_path
    
    def _add_cover_page(self, pdf, style, story_text):
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, f'{style} Storybook', 0, 1, 'C')
        
        # Add story introduction
        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, story_text)
        
    def _add_story_pages(self, pdf, images):
        for idx, image in enumerate(images, 1):
            pdf.add_page()
            
            # Save image with high quality
            temp_path = os.path.join(OUTPUT_FOLDER, f'temp_{idx}.jpg')
            image.save(temp_path, 'JPEG', quality=95)
            
            # Calculate optimal image dimensions
            img_w = 190  # Max width
            img_h = 250  # Max height
            aspect = image.size[0] / image.size[1]
            
            if aspect > img_w/img_h:
                w = img_w
                h = img_w / aspect
            else:
                h = img_h
                w = img_h * aspect
            
            # Center image on page
            x = (210 - w) / 2
            y = (297 - h) / 3
            
            # Add image
            pdf.image(temp_path, x=x, y=y, w=w)
            os.remove(temp_path)  # Clean up
            
            # Add page number
            pdf.set_y(-15)
            pdf.set_font('Arial', 'I', 8)
            pdf.cell(0, 10, f'Page {idx}', 0, 0, 'C')

# Initialize processors
story_processor = StoryProcessor()
pdf_generator = StoryBookGenerator()

@app.route('/')
def index():
    return render_template('index.html', 
                         story_templates=STORY_TEMPLATES,
                         styles=STYLE_CONFIGS)

@app.route('/get_random_story')
def get_random_story():
    story_type = request.args.get('format_type', 'Manga')
    if story_type in STORY_TEMPLATES:
        random_story = random.choice(STORY_TEMPLATES[story_type])
        return jsonify({'story': random_story})
    return jsonify({'story': ''})

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        images = request.files.getlist('images')
        story_text = request.form.get('story_text', '').strip()
        format_type = request.form.get('format_type', 'Manga')
        character_name = request.form.get('character_name', 'our hero')
        
        # Validate inputs
        if len(images) < 4 or len(images) > 5:
            return jsonify({'error': 'Please upload 4-5 images'})
        
        # Save and process images
        saved_images = []
        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image.save(filepath)
                saved_images.append(filepath)
        
        # If using a template story, personalize it
        if story_text in STORY_TEMPLATES[format_type]:
            story_text = story_text.format(name=character_name)
        
        # Process images with AI
        processed_images = story_processor.process_images(saved_images, format_type)
        
        # Generate storybook
        pdf_path = pdf_generator.create_storybook(processed_images, story_text, format_type)
        
        return jsonify({
            'success': True,
            'pdf_url': url_for('static', filename=f'output/{os.path.basename(pdf_path)}')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(OUTPUT_FOLDER, filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
