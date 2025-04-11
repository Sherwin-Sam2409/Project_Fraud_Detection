from flask import Flask, request, jsonify
from flask_cors import CORS
from model import detect_fraud, train_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "message": "Fraud Detection API is running!",
        "endpoints": {
            "POST /predict": "Make fraud prediction for a transaction",
            "GET /train": "Retrain the model using CSV dataset"
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction_data = request.get_json()

        if not transaction_data:
            return jsonify({"error": "No transaction data provided"}), 400

        result = detect_fraud(transaction_data)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/train', methods=['GET'])
def train():
    try:
        train_model("financial_transactions.csv")
        return jsonify({"message": "Model retrained successfully!"}), 200
    except FileNotFoundError:
        return jsonify({"error": "Training dataset not found."}), 404
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
