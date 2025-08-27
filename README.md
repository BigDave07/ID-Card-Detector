# ID Card Detector

A production-ready starter for classifying uploaded images as Real vs Fake ID cards using transfer learning (MobileNetV2) and a practical business rule:

# Decision policy: if the model’s probability that an image is Real is below 60%, the result is Fake.

# ⚠️ Disclaimer: 
This project is for learning and prototyping. It is not a certified ID verification system and does not address regulatory, privacy, or security requirements for production use.

# Table of Contents
Features
Architecture
Project Structure
Quickstart
Training
Decision Policy (60% Rule)
API
Deployment
Large Files (Model) & Git LFS
Troubleshooting
Roadmap

# Features

# Transfer Learning with MobileNetV2

🧠 Two-phase training: head training (frozen backbone) → targeted fine-tuning

🎛️ Data Augmentation for robustness

📊 Validation metrics (accuracy/AUC) & saved metadata for inference

🧩 Clean Flask UI with live image preview

🧱 Clear project structure, ready for CI/CD & containerization

# Architecture

Data: Train/ (and optional Test/) with folder-per-class (e.g., Fake/, Real/).

Training (train.py):

Build datasets + augmentation

Train classifier head (backbone frozen)

Unfreeze top N layers → fine-tune with low LR

Save model.h5 and metadata.json (class names, image size, thresholds)

Inference (app.py):

Preprocess upload → model predicts P(Real)

Apply 60% rule → label & confidence

Render result in index.html

# Project Structure
├─ app.py                 # Flask server (uses model.h5 + metadata.json)

├─ train.py               # Training / fine-tuning script

├─ metadata.json          # Class names, image size, thresholds

├─ model.h5               # (large; consider Git LFS or ignoring in git)

├─ templates/

│  └─ index.html

└─ static/

     ├─ style.css
   
     └─ script.js

# Quickstart
1) Install dependencies
pip install -r requirements.txt
minimal alt:
pip install flask pillow numpy tensorflow

2) Ensure a trained model

If model.h5 and metadata.json already exist in the repo root, proceed.

Otherwise, see Training

3) Run the app
python app.py

Open http://127.0.0.1:5000
 and upload an image.

# Training
The default setup expects two classes (binary): Fake/ and Real/.

# Configure & run

Edit paths in train.py:
TRAIN_DIR = r"C:\path\to\dataset\Train"
TEST_DIR  = r"C:\path\to\dataset\Test" 

# Train:
python train.py


# Outputs:

model.h5 — trained weights & architecture

metadata.json — class names, image size, (and learned threshold for reference)

Need to reject non-ID images explicitly? Extend to a 3-class setup (Fake/, NotID/, Real/). The app can be adapted to softmax and a confidence-based reject rule.

Decision Policy (60% Rule)

The app enforces a fixed threshold: if P(Real) < 0.60 ⇒ Fake.

Adjust in app.py:

FIXED_DECISION_THRESHOLD = 0.60

# Deployment

Production server: use a WSGI server (e.g., gunicorn on Linux/macOS or waitress on Windows) behind Nginx/Apache.

Environment: pin dependencies in requirements.txt; consider Docker for reproducibility.

Security: add file-size limits, content-type checks, and proper logging before exposing publicly.

Example (waitress on Windows):

pip install waitress
waitress-serve --host=0.0.0.0 --port=8000 app:app

# Large Files (Model) & Git LFS

GitHub blocks files >100 MB by default. Choose one:

Option A: ignore model artifacts

# .gitignore
*.h5
*.hdf5
*.tflite
*.onnx
models/
checkpoints/


If already added:

git rm --cached model.h5
git add .gitignore
git commit -m "chore: ignore model artifacts"
git push


Option B: use Git LFS

git lfs install
git lfs track "*.h5" "*.hdf5" "*.tflite" "*.onnx"
git add .gitattributes model.h5
git commit -m "chore: track model with Git LFS"
git push
