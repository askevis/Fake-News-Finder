from flask import Flask, request, jsonify
#from flask_cors import CORS
from setfit import SetFitModel

model = SetFitModel.from_pretrained("Anthony246346/fake-news-setfit-e5")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return 'Fake News API is running!'

@app.route("/health", methods=["GET"])
def health_check():

    if model is None:
        return jsonify({
            "status": "DOWN",
            "message": "Model failed to load."
        }), 503

    return jsonify({
        "status": "UP",
        "message": "Fake News API is running!"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data['title']
    text = data['text']

    if not text or not isinstance(text, str):
        return jsonify({"error": "No valid article text provided"}), 400
    if not title or not isinstance(title, str):
        return jsonify({"error": "No valid article title provided"}), 400

    combined_input = f"query: {title} {text}"

    try:
        pred = model.predict([combined_input])[0]
        probs = model.predict_proba([combined_input])[0].tolist()

        return jsonify({
            "input_text": combined_input,
            "prediction_label_index": int(pred),
            "prediction_probabilities": probs
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)