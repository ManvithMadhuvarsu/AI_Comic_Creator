import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Directory configuration
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, '../uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, '../static/output')
    
    # File configuration
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model configuration
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    
    STORY_TEMPLATES = {
        'Manga': [
            "A young hero discovers their hidden powers...",
            "In a world of martial arts..."
        ],
        'Comic': [
            "Our hero becomes a masked vigilante...",
            "A regular person gains supernatural abilities..."
        ],
        'Animated Story': [
            "A magical adventure begins...",
            "Through a series of whimsical events..."
        ]
    }
    
    STYLE_CONFIGS = {
        'Manga': {
            'prompt': "manga style, anime artwork, detailed line art, black and white",
            'negative_prompt': "realistic, photographic, color, western style",
            'background': 'static/backgrounds/manga_bg.png'
        },
        'Comic': {
            'prompt': "comic book style, superhero art, bold colors, action scene",
            'negative_prompt': "realistic, photograph, anime, sketch",
            'background': 'static/backgrounds/comic_bg.png'
        },
        'Animated Story': {
            'prompt': "disney pixar style, 3D animated, colorful, cheerful",
            'negative_prompt': "realistic, photograph, anime, sketch",
            'background': 'static/backgrounds/animated_bg.png'
        }
    }
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)