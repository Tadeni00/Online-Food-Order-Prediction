from flask import Flask, request, render_template
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictionPipeline
from src.components.data_validation import DataValidation
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract form data based on columns
            data = CustomData(
                gender=request.form.get('Gender'),
                marital_status=request.form.get('Marital Status'),
                occupation=request.form.get('Occupation'),
                educational_qualifications=request.form.get('Educational Qualifications'),
                feedback=request.form.get('Feedback'),
                age=int(request.form.get('Age')),
                monthly_income=int(request.form.get('Monthly Income')),
                family_size=int(request.form.get('Family size')),
                latitude=float(request.form.get('latitude')),
                longitude=float(request.form.get('longitude')),
                pin_code=int(request.form.get('Pin code'))
            )

            # Convert the data to a DataFrame
            pred_df = data.get_data_as_dataframe()

            # Initialize DataValidation (dummy paths here since we're only validating DataFrame)
            data_validation = DataValidation(None, None)
            try:
                data_validation.validate_data_format(pred_df)
            except CustomException as e:
                # Handle validation exceptions
                return render_template('home.html', results=f"Validation error: {e}")

            # Predict using the pipeline
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        
        except ValueError as ve:
            # Handle value errors (e.g., conversion issues)
            print(f"Value error during prediction: {ve}")
            return render_template('home.html', results="Invalid input data")
        except Exception as e:
            # Handle other exceptions
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error during prediction")

    # Render the index.html page for GET requests
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
