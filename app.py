from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np
import json

MODEL_PATH = "model.h5"
META_PATH  = "metadata.json"

app = Flask(__name__)

# Load model + metadata once
model = tf.keras.models.load_model(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

CLASS_NAMES = meta["class_names"]
IMG_SIZE    = tuple(meta["img_size"])
MODE        = meta.get("mode", "binary")  # 'binary' or 'multiclass'

# ---- Decision policy ----
# Always use a fixed 0.60 rule for binary: if P(Real) < 0.60 -> Fake
FIXED_DECISION_THRESHOLD = 0.60

# Binary metadata (still loaded, but we override with FIXED_DECISION_THRESHOLD)
THRESH_META = float(meta.get("threshold", 0.50))
POS_IDX     = int(meta.get("positive_class_index", 1))  # by default index 1 is usually 'Real'

# Multiclass fallback (only used if you trained 3+ classes)
REJECT_T = float(meta.get("reject_threshold", 0.60))

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template("index.html", result="No file uploaded")

        # Load & preprocess
        file_bytes = request.files['file'].read()
        img = Image.open(BytesIO(file_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        if MODE == 'binary':
            # prob_pos is P(class index 1) which (with typical folders ['Fake','Real']) = P(Real)
            prob_pos = float(model.predict(x, verbose=0)[0][0])

            thr = FIXED_DECISION_THRESHOLD  # force 0.60 rule here
            if prob_pos >= thr:
                label = CLASS_NAMES[POS_IDX]      # e.g., 'Real'
                conf  = prob_pos
            else:
                neg_idx = 1 - POS_IDX
                label = CLASS_NAMES[neg_idx]      # e.g., 'Fake'
                conf  = 1.0 - prob_pos

            result_text = f"{label} ({conf*100:.1f}% confidence, thr={thr:.2f})"

        else:
            # If you trained 3 classes (e.g., ['Fake','NotID','Real']):
            probs = model.predict(x, verbose=0)[0]     # softmax
            # If 'Real' exists, apply the same 60% rule to 'Real' confidence
            if "Real" in CLASS_NAMES:
                real_idx = CLASS_NAMES.index("Real")
                prob_real = float(probs[real_idx])
                if prob_real >= FIXED_DECISION_THRESHOLD:
                    result_text = f"Real ({prob_real*100:.1f}% confidence)"
                else:
                    # below 60 → call it Fake per your policy (even if NotID is higher)
                    result_text = f"Fake ({(1.0 - prob_real)*100:.1f}% inferred)"
            else:
                # Fallback: just argmax if 'Real' not in class names
                idx = int(np.argmax(probs))
                label = CLASS_NAMES[idx]
                conf  = float(probs[idx])
                # Optional reject if low confidence
                if conf < REJECT_T:
                    result_text = f"Uncertain ({label}, {conf*100:.1f}%)"
                else:
                    result_text = f"{label} ({conf*100:.1f}% confidence)"

        return render_template("index.html", result=result_text)

    except Exception:
        import traceback; traceback.print_exc()
        return render_template("index.html", result="Internal error during prediction.")
        
if __name__ == '__main__':
    app.run(debug=True)
