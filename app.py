from flask import Flask, render_template, request, jsonify
import json
import joblib
import pandas as pd
from custom_preprocessor import CustomPreprocessor

app = Flask(__name__, static_folder='static')

# Define an empty DataFrame at the beginning
prediction_df = pd.DataFrame(columns=['booking_id', 'predicted_cancellation'])

@app.route('/')
def home():
    # Load analysis results from the JSON file
    with open('analysis_results.json') as infile:
        analysis_results = json.load(infile)
    
    # Pass the results to the template
    return render_template('index.html', analysis_results=analysis_results, prediction_df=prediction_df)

# Load the model
model = joblib.load("model_pipe.joblib")
@app.route('/predict', methods=['POST'])
def predict():
    global prediction_df
    # Prediction logic
    data_df = pd.read_csv('data_to_predict.csv')

    # Apply preprocessing steps
    #preprocessor = CustomPreprocessor()
    #data_df_preprocessed = preprocessor.transform(data_df)

    # Apply the same filter as in the model pipeline
    filter = (data_df['children'] == 0) & (data_df['adults'] == 0) & (data_df['babies'] == 0)
    data_df_filtered = data_df[~filter]

    # Extract booking IDs
    booking_ids = data_df_filtered['booking_id']

    # Predict cancellation
    prediction_results = model.predict(data_df_filtered)

    # Create a DataFrame with booking IDs and predicted cancellation
    prediction_df = pd.DataFrame({'booking_id': booking_ids, 'predicted_cancellation': prediction_results})

    # Save prediction results as CSV
    prediction_csv_path = 'prediction_results.csv'
    prediction_df.to_csv(prediction_csv_path, index=False)

    # Render the prediction results as a table in the template
    return render_template('index.html', prediction_df=prediction_df)

if __name__ == '__main__':
    app.run(debug=True)
