# Pneumonia Detection System

AI-powered chest X-ray analysis for pneumonia detection using DenseNet121.

## Features
- Upload chest X-ray images
- Real-time pneumonia detection
- Confidence score visualization
- Modern medical-themed UI

## Setup

1. Install dependencies:
```bash
pip install flask tensorflow pillow numpy
```

2. **Add your trained model:**
   - Place your trained model file in the project root
   - Name it: `best_densenet_feature_extraction (3).keras`
   - Or update the model path in `app.py` line 38

3. Run the application:
```bash
python app.py
```

4. Open browser: http://127.0.0.1:5000

## Project Structure
```
pneumonia_app/
├── app.py                          # Flask backend
├── templates/
│   └── index.html                  # Frontend UI
├── static/
│   └── css/
│       └── style.css              # Styles (optional)
└── best_densenet_feature_extraction (3).keras  # Model (not included)
```

## Note
The trained model file is not included in this repository due to its large size. You need to train your own model or obtain one separately.

## Technologies
- Flask
- TensorFlow/Keras
- Tailwind CSS
- DenseNet121
