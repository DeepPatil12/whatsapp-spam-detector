from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model and vectorizer
classifier = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON data from request
    message = data.get("message", "")  # Extract message
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Transform message using vectorizer
    message_tfidf = vectorizer.transform([message])

    # Predict
    prediction = classifier.predict(message_tfidf)[0]

    # Return result as JSON
    return jsonify({"message": message, "prediction": "spam" if prediction == 1 else "ham"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get port from environment variable
    app.run(host="0.0.0.0", port=port, debug=True)  # Ensure app runs on the right port
