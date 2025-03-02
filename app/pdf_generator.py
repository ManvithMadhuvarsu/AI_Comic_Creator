from fpdf import FPDF
import os
from PIL import Image
from .config import Config

class StoryBookGenerator:
    def __init__(self):
        self.page_width = 210  # A4 width in mm
        self.page_height = 297  # A4 height in mm
        
    def create_storybook(self, images, story_text, style):
        pdf = FPDF()
        style_config = Config.STYLE_CONFIGS[style]
        
        # Create cover
        self.create_cover(pdf, style, story_text)
        
        # Create story pages
        for i, image in enumerate(images):
            self.create_story_page(pdf, image, story_text, style_config, page_num=i+1)
        
        # Save PDF
        output_path = os.path.join(Config.OUTPUT_FOLDER, f'storybook_{style.lower()}.pdf')
        pdf.output(output_path)
        return output_path
    
    def create_cover(self, pdf, style, title):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 30, f'{style} Storybook', 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, title)
    
    def create_story_page(self, pdf, image, text, style_config, page_num):
        pdf.add_page()
        
        # Add styled background
        if style_config['background']:
            pdf.image(style_config['background'], 0, 0, self.page_width)
        
        # Add main image
        img_w = self.page_width * 0.8
        img_h = self.page_height * 0.6
        x = (self.page_width - img_w) / 2
        pdf.image(image, x, 30, img_w)
        
        # Add text
        pdf.set_font('Arial', '', 12)
        pdf.set_y(img_h + 40)
        pdf.multi_cell(0, 10, text)