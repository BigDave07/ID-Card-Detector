import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("GCP_API_KEY")
if not API_KEY:
    raise RuntimeError("GCP_API_KEY not set. Put it in .env or set env var.")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
LANGUAGES_URL = f"{TRANSLATE_URL}/languages"

@app.get("/")
def root():
    return send_from_directory("static", "index.html")

@app.get("/api/languages")
def get_languages():
    params = {"key": API_KEY, "target": "en"}
    r = requests.get(LANGUAGES_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    return jsonify(data)

@app.post("/api/translate")
def translate():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    target = payload.get("target", "en")
    source = payload.get("source")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    params = {
        "key": API_KEY,
        "q": text,
        "target": target,
        "format": "text"
    }
    if source and source.lower() != "auto":
        params["source"] = source

    try:
        r = requests.post(TRANSLATE_URL, params=params, timeout=20)
        r.raise_for_status()
        translations = r.json()["data"]["translations"]
        result = translations[0]
        return jsonify({
            "translatedText": result.get("translatedText"),
            "detectedSourceLanguage": result.get("detectedSourceLanguage")
        })
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Translation API error: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
