import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO
from fake_news_detector import SimpleFakeNewsDetector

# Interactive Jupyter widget-based interface
class InteractiveFakeNewsDetector:
    def __init__(self):
        self.detector = SimpleFakeNewsDetector()
        self.create_examples()
        self.build_interface()
    
    def create_examples(self):
        """Create example images for the interface"""
        os.makedirs('examples', exist_ok=True)
        
        self.example_texts = [
            "BREAKING: President announces sudden resignation after secret meetings with foreign leaders.",
            "Scientists discover miracle fruit that cures all diseases. Big pharma doesn't want you to know about this.",
            "Famous celebrity secretly marries in private ceremony. Exclusive photos reveal the truth."
        ]
        
        self.example_images = []
        
        for i, text in enumerate(self.example_texts):
            # Create a simple image for each example
            img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add some text to the image
            text_to_draw = f"Example {i+1}"
            draw.text((10, 10), text_to_draw, fill="black")
            
            # Add a visual element
            draw.rectangle([50, 50, 350, 150], outline="red", fill="yellow")
            
            # Save image and add to list
            img_path = f"examples/example_{i+1}.jpg"
            img.save(img_path)
            self.example_images.append(img)
    
    def build_interface(self):
        """Build the IPython widget interface"""
        # Create style
        style = HTML("""
        <style>
        .widget-label { min-width: 20% !important; }
        .widget-text { width: 100% !important; }
        .widget-textarea { width: 100% !important; }
        </style>
        """)
        display(style)
        
        # Create widgets
        self.title = widgets.HTML("<h1>Fake News Detection Demo</h1>")
        self.description = widgets.HTML("<p>Upload text and an image to analyze potential fake news</p>")
        
        # Input widgets
        self.text_input = widgets.Textarea(
            value='',
            placeholder='Enter news text here...',
            description='News Text:',
            disabled=False,
            layout=widgets.Layout(width='100%', height='100px')
        )
        
        self.image_upload = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='Upload Image:'
        )
        
        # Example selection
        self.example_dropdown = widgets.Dropdown(
            options=[f'Example {i+1}' for i in range(len(self.example_texts))],
            value='Example 1',
            description='Examples:',
            disabled=False,
        )
        
        # Analyze button
        self.analyze_button = widgets.Button(
            description='Analyze',
            disabled=False,
            button_style='success',
            tooltip='Click to analyze',
            icon='check'
        )
        
        # Output area
        self.output_area = widgets.Output()
        
        # Create the layout
        self.app = widgets.VBox([
            self.title,
            self.description,
            widgets.HTML("<h2>Input</h2>"),
            self.text_input,
            widgets.HBox([self.image_upload, self.example_dropdown]),
            self.analyze_button,
            widgets.HTML("<h2>Results</h2>"),
            self.output_area
        ])
        
        # Set up event handlers
        self.analyze_button.on_click(self.on_analyze_click)
        self.example_dropdown.observe(self.on_example_change, names='value')
        
        # Load the first example
        self.on_example_change(None)
    
    def on_example_change(self, change):
        """Handle example selection change"""
        example_num = int(self.example_dropdown.value.split()[1]) - 1
        self.text_input.value = self.example_texts[example_num]
        
        # Clear the file uploader - not possible with ipywidgets, so we just note it
        if hasattr(self, 'example_image_loaded'):
            pass
        else:
            with self.output_area:
                print("Using example image. You can also upload your own image.")
            self.example_image_loaded = True
    
    def on_analyze_click(self, b):
        """Handle analyze button click"""
        with self.output_area:
            clear_output()
            
            # Get the text
            text = self.text_input.value
            if not text:
                print("Please enter some text.")
                return
            
            # Get the image
            image = None
            
            # If a file was uploaded, use that
            if self.image_upload.value:
                image_name = list(self.image_upload.value.keys())[0]
                image_data = self.image_upload.value[image_name]['content']
                image = Image.open(BytesIO(image_data))
                print(f"Using uploaded image: {image_name}")
            else:
                # Use the example image
                example_num = int(self.example_dropdown.value.split()[1]) - 1
                image = self.example_images[example_num]
                print(f"Using example image {example_num + 1}")
            
            # Process the input
            result = self.detector.process(text, image)
            
            # Display results
            html_output = f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;">
                <h3>Prediction: <span style="color: {'red' if result['prediction'] == 'Fake' else 'green'}">{result['prediction']}</span></h3>
                <h4>Confidence: {result['confidence']:.2f}%</h4>
                <p><strong>Explanation:</strong> {result['explanation']}</p>
            </div>
            """
            
            if result['visualization_base64']:
                html_output += f"""
                <div style="text-align: center; margin-top: 20px;">
                    <h3>Visualization:</h3>
                    <img src="data:image/png;base64,{result['visualization_base64']}" style="max-width: 100%;" />
                </div>
                """
            
            display(HTML(html_output))
    
    def display(self):
        """Display the interface"""
        display(self.app)
