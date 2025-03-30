from flask import Flask, request, jsonify
import joblib

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
    app.run(debug=True)
