# Interactive Fake News Detection System

This system detects potential fake news by analyzing both text and images using deep learning techniques, presented in an interactive Jupyter interface.

## Features

- Joint text and image analysis for improved fake news detection
- Attention-based visualization of suspicious elements
- Explanation generation for model decisions
- Interactive IPython widgets interface for analysis
- Example news items for demonstration

## Installation

```bash
# Clone the repository
git clone https://github.com/ali-abdullah9/fake-news-detection.git
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the interactive interface in Jupyter or Google Colab:

```python
from interactive_interface import InteractiveFakeNewsDetector

# Create and display the interface
demo = InteractiveFakeNewsDetector()
demo.display()
```

## Model Architecture

The system uses a simplified multimodal architecture:
- Text features: BERT encoder
- Image features: ResNet50
- Simple feature fusion for binary classification
- Attention visualization for interpretability

## Example News Items

The interface comes with pre-loaded example news items for demonstration:
1. Political news example
2. Health/science news example
3. Celebrity news example

## Performance

This demonstration version generates random predictions. The full model achieves:
- Accuracy: 69.70%
- F1 Score: 66.92%

## Contributors

- Ali Abdullah
