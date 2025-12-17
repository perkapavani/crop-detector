# crop-detector
"Flask-based API for crop disease detection and data preprocessing with ID3, Apriori, and FP-Growth support."
# Crop Detector 🌾

This is a Flask-based application for **crop disease detection** and **data preprocessing** using machine learning techniques like:

- ID3 Decision Tree
- Apriori Algorithm
- FP-Growth

## 📁 Project Structure

## 📈 Association Rule Mining Output

This diagram shows the output of the Apriori algorithm applied to the crop dataset.

![Apriori Result](https://github.com/perkapavani/crop-detector/blob/main/apriori_result.png?raw=true)


## 🚀 Features

- Crop disease prediction based on user input
- Data preprocessing: normalization, discretization
- Uses ML algorithms for classification and association rule mining

## ⚙️ Technologies Used

- Python
- Flask
- Pandas, NumPy
- ID3, Apriori, FP-Growth
- HTML/CSS (for UI)

## Project Structure
- app.py : Main Flask application
- model/ : Trained ML model files
- templates/ : HTML files
- static/ : CSS and assets

## How to Run the Project
1. Clone the repository
2. Install required packages:
   pip install -r requirements.txt
3. Run the Flask app:
   python app.py
4. Open browser and go to:
   http://127.0.0.1:5000/

## Features
- Predicts crop disease based on input
- Simple and user-friendly interface
- Fast response using trained ML model

## Future Improvements
- Improve UI
- Add more crop datasets
- Better error handling

```bash

# Step 1: Activate virtual environment (if any)
# Step 2: Run the app
python app.py
Then open your browser at: http://localhost:5000

