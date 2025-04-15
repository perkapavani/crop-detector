import requests

# Server URL
url = "http://127.0.0.1:5000/upload"

# File to upload
file_path = "C:/Users/prave/OneDrive/Desktop/1207Dmcbp/cropdetection.csv"

# Form data
form_data = {
    'preprocessing': 'normalization',  # Options: 'normalization', 'missing_data', 'discretization'
    'classifier': 'id3'               # Options: 'id3', 'apriori', 'fp_growth'
}

# Open the file in binary mode and send the POST request
with open(file_path, 'rb') as f:
    files = {'file': (file_path, f)}
    try:
        # Sending the request
        response = requests.post(url, files=files, data=form_data)
        
        # Print the server's response
        if response.status_code == 200:
            print("Response from server:", response.json())
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("An error occurred:", e)
