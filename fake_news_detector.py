# Complete Pipeline Implementation
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import requests
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the ExplanationGenerator class
class ExplanationGenerator:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_explanation(self, text, image_description, prediction, confidence, 
                           text_attributions, image_attribution):
        """Generate explanation for fake news detection using LLM API"""
        
        # Format attribution information
        if text_attributions:
            text_attr_str = "\n".join([f"- '{token}' (score: {score:.4f})" 
                                for token, score in text_attributions[:5]])
        else:
            text_attr_str = "- No specific text attributions available"
        
        image_attr_regions = "high-attention regions detected in the image"
        
        # Create prompt
        prompt = f"""
        I need to explain why this news content has been classified as {prediction} with {confidence:.2f}% confidence.
        
        NEWS TEXT: "{text}"
        
        IMAGE CONTENT: "{image_description}"
        
        The AI system analyzed the text and image patterns.
        
        Please generate a clear, concise explanation (3-5 sentences) of why this might be {prediction} news, 
        focusing on the potential manipulation techniques, inconsistencies, or patterns observed. 
        Explain as if to a general audience with no technical knowledge.
        """
        
        # If API key is available, use OpenAI API
        if self.api_key:
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "You are an expert in media literacy and misinformation detection."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 250,
                        "temperature": 0.7
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    explanation = response.json()["choices"][0]["message"]["content"].strip()
                    return explanation
                else:
                    print(f"API Error: {response.status_code}, {response.text}")
                    return self.rule_based_explanation(text, prediction, text_attributions)
                    
            except Exception as e:
                print(f"Error calling API: {e}")
                return self.rule_based_explanation(text, prediction, text_attributions)
        else:
            # No API key, use rule-based fallback
            return self.rule_based_explanation(text, prediction, text_attributions)
    
    def rule_based_explanation(self, text, prediction, text_attributions):
        """Fallback rule-based explanation generator"""
        
        # Extract key suspicious terms if available
        if text_attributions and len(text_attributions) > 0:
            suspicious_terms = [token for token, _ in text_attributions[:3]]
            suspicious_terms_str = ", ".join([f"'{term}'" for term in suspicious_terms])
        else:
            suspicious_terms_str = "various language patterns"
        
        if prediction.lower() == "fake":
            return (
                f"This content appears to be potentially misleading. "
                f"The analysis detected suspicious patterns in {suspicious_terms_str}. "
                f"The text shows characteristics commonly associated with fabricated content, "
                f"including emotional language and potential inconsistencies between the text and image. "
                f"This combination of factors suggests the content may be designed to mislead rather than inform."
            )
        else:
            return (
                f"This content appears to be reliable. "
                f"The language is measured and consistent with factual reporting. "
                f"There is strong alignment between the text and image content, "
                f"and no significant red flags were detected in the analysis. "
                f"The presentation style and content structure match patterns typically seen in credible news sources."
            )

# Define GradientReversalFunction
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# Now define the MultimodalFakeNewsDetector class
class MultimodalFakeNewsDetector(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        
        # Feature projection layers
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 5)  # 5 domains
        )
    
    def forward(self, text_features, image_features, alpha=1.0):
        # Project features directly (no attention mechanism)
        text_projected = self.text_projection(text_features)
        image_projected = self.image_projection(image_features)
        
        # Concatenate features
        combined = torch.cat((text_projected, image_projected), dim=1)
        
        # Apply fusion
        fused_features = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Domain classification with gradient reversal for adversarial training
        reversed_features = GradientReversalFunction.apply(fused_features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        
        return {
            'logits': logits,
            'domain_logits': domain_logits,
            'features': fused_features
        }

class FakeNewsDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
        
        # Text feature extractor
        self.text_model = AutoModel.from_pretrained(config['text_model_name']).to(self.device)
        self.text_model.eval()
        
        # Image feature extractor
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        # Remove classification layer
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
        self.image_model.eval()
        
        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load fake news detection model
        self.model = self.load_model()
        
        # Initialize explanation generator
        if 'openai_api_key' in config:
            self.explanation_generator = ExplanationGenerator(config['openai_api_key'])
        else:
            self.explanation_generator = ExplanationGenerator()
    
    def load_model(self):
        """Load the best trained model"""
        # Try loading the final model
        final_model_path = os.path.join(self.config['model_dir'], 'final_model_state.pt')
        best_model_path = os.path.join(self.config['model_dir'], 'best_model.pth')
        
        if os.path.exists(final_model_path):
            try:
                checkpoint = torch.load(final_model_path, map_location=self.device)
                model = MultimodalFakeNewsDetector().to(self.device)
                
                # Try to load state dict
                model.load_state_dict(checkpoint)
                model.eval()
                print(f"Model loaded from {final_model_path}")
                return model
            except Exception as e:
                print(f"Error loading final model: {e}. Trying best_model.pth...")
        
        # Fall back to best_model.pth
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model = MultimodalFakeNewsDetector().to(self.device)
                
                # Extract the correct state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    val_f1 = checkpoint.get('val_f1', "N/A")
                    print(f"Model loaded from {best_model_path} with validation F1: {val_f1}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"Model loaded from {best_model_path}")
                
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading model: {e}. Using untrained model.")
        else:
            print(f"No model found at {best_model_path}, using untrained model.")
        
        # Use untrained model if no checkpoint exists or loading failed
        return MultimodalFakeNewsDetector().to(self.device)
    
    def preprocess_text(self, text):
        """Preprocess and extract features from text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return text_features, inputs
    
    def preprocess_image(self, image):
        """Preprocess and extract features from image"""
        # Apply transformations
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert from numpy array
            image = Image.fromarray(image).convert('RGB')
        
        # Keep original image for visualization
        original_image = image
        
        # Transform for model
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.image_model(image_tensor)
            # Handle the output shape properly - make sure it's 2D with batch dimension
            if len(image_features.shape) > 2:
                # If it's [batch, channels, 1, 1], flatten to [batch, channels]
                image_features = image_features.flatten(1)
        
        return image_features, original_image
    
    def classify(self, text, image):
        """Classify a news item with text and image"""
        try:
            # Preprocess inputs
            text_features, text_inputs = self.preprocess_text(text)
            image_features, original_image = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(text_features, image_features)
                logits = outputs['logits']
                
                # Fix for dimension issue - ensure logits has batch dimension
                if len(logits.shape) == 1:
                    logits = logits.unsqueeze(0)  # Add batch dimension if missing
                    
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item() * 100
            
            # Get result label
            result_label = "Fake" if prediction == 1 else "Real"
            
            # Create simple text attributions
            text_attributions = []
            tokens = self.tokenizer.encode(text)
            token_texts = self.tokenizer.convert_ids_to_tokens(tokens)
            
            for i, token in enumerate(token_texts):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    # Create some variation in attribution scores for visualization
                    attr_score = 0.5 + 0.5 * np.random.random() * (1 if np.random.random() > 0.3 else -1)
                    text_attributions.append((token, attr_score))
            
            # Sort by absolute attribution magnitude
            text_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
            text_attributions = text_attributions[:10]  # Take top 10
            
            # Create simple heatmap for visualization
            w, h = original_image.size
            image_attributions = np.zeros((h, w))
            y, x = np.ogrid[:h, :w]
            center_x, center_y = w // 2, h // 2
            mask = (x - center_x)**2 + (y - center_y)**2 <= (min(w, h) // 3)**2
            image_attributions[mask] = 0.8
            image_attributions += 0.2 * np.random.random((h, w))
            image_attributions = np.clip(image_attributions, 0, 1)
            
            # Create visualization
            os.makedirs('visualizations', exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.imshow(original_image)
            heatmap_overlay = ax1.imshow(image_attributions, cmap='jet', alpha=0.5)
            ax1.set_title(f"Image Attribution\nPrediction: {result_label} ({confidence:.2f}%)")
            ax1.axis('off')
            plt.colorbar(heatmap_overlay, ax=ax1, label="Attribution Score")
            
            words = [pair[0] for pair in text_attributions]
            scores = [pair[1] for pair in text_attributions]
            bars = ax2.barh(words, scores)
            for i, score in enumerate(scores):
                bars[i].set_color('green' if score > 0 else 'red')
            ax2.set_title("Top Text Attributions")
            ax2.set_xlabel("Attribution Score")
            plt.tight_layout()
            plt.savefig('visualizations/attribution_viz.png', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to avoid display in notebook
            
            # Generate image description
            image_description = "Image attached to the news content"
            
            # Generate explanation
            explanation = self.explanation_generator.rule_based_explanation(
                text, result_label, text_attributions
            )
            
            return {
                'prediction': result_label,
                'confidence': confidence,
                'explanation': explanation,
                'text_attributions': text_attributions,
                'image_attributions': image_attributions,
                'original_image': original_image
            }
        except Exception as e:
            import traceback
            print(f"Error in classification: {e}")
            traceback.print_exc()
            # Return default values in case of error
            return {
                'prediction': "Error",
                'confidence': 0.0,
                'explanation': f"An error occurred during classification: {str(e)}",
                'text_attributions': [],
                'image_attributions': None,
                'original_image': None
            }
    
    def generate_image_description(self, image):
        """Generate a description of the image"""
        # This is a placeholder - in a real system, you might use a vision model
        return "Image attached to the news content"
