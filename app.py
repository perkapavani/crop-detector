from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Helper functions for data preprocessing and classification
def preprocess_data(df, method):
    if method == 'normalization':
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif method == 'missing_data':
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'discretization':
        # Convert numerical values into bins (example for first column only)
        df.iloc[:, 0] = pd.cut(df.iloc[:, 0], bins=3, labels=["Low", "Medium", "High"])
    return df

def classify_data(df, classifier):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable

    if classifier == 'id3':
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X, y)
        return {"accuracy": clf.score(X, y)}
    elif classifier == 'apriori':
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
        return frequent_itemsets.to_dict()
    elif classifier == 'fp_growth':
        frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
        return frequent_itemsets.to_dict()
    else:
        return {"error": "Classifier not supported"}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check file extension and read file accordingly
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading file: {e}"}), 400

    # Preprocess data
    preprocessing_option = request.form.get('preprocessing')
    if preprocessing_option:
        df = preprocess_data(df, preprocessing_option)

    # Classify data
    classifier_option = request.form.get('classifier')
    if classifier_option:
        result = classify_data(df, classifier_option)
        return jsonify({"result": result})

    return jsonify({"message": "File processed successfully"})

if __name__ == '__main__':
    app.run(debug=True)
