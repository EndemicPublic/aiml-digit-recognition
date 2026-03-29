# Handwritten Digit Recognition

A simple AI/ML project for B.Tech first-year students that recognizes handwritten digits (0-9) using the MNIST dataset.

## What You'll Learn

- Basic neural networks
- Loading and preprocessing image data
- Training a machine learning model
- Evaluating model accuracy

## Prerequisites

- Python 3.8+
- VS Code with Python extension

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd aiml-digit-recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python digit_recognition.py
```

## Project Structure

```
aiml-digit-recognition/
├── digit_recognition.py    # Main code
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## How It Works

1. Load MNIST dataset (70,000 handwritten digit images)
2. Preprocess images (normalize pixel values)
3. Build a simple neural network
4. Train the model
5. Test accuracy on unseen data

## Expected Output

- Training accuracy: ~92-95%
- Test accuracy: ~92-94%
