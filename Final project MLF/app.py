from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Load fitted label encoders from their respective pickle files
label_encoders = {
    'Ever_Married': joblib.load('Ever_Married_label_encoder.pkl'),
    'Graduated': joblib.load('Graduated_label_encoder.pkl'),
    'Profession': joblib.load('Profession_label_encoder.pkl'),
    'Var_1': joblib.load('Var_1_label_encoder.pkl')
}

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    
    # Create DataFrame from input data
    input_data = pd.DataFrame(data)

    # Define the expected feature set (must match what was used for training)
    expected_features = ['Ever_Married', 'Age', 'Graduated', 
                         'Profession', 'Work_Experience', 
                         'Spending_Score', 'Family_Size', 'Var_1']

    # Ensure input_data only contains expected features
    input_data = input_data[expected_features]

    # Define the mapping for Spending_Score
    spending_score_mapping = {'Low': 0, 'Average': 1, 'High': 2}

    # Apply the mapping to Spending_Score
    input_data['Spending_Score'] = input_data['Spending_Score'].map(spending_score_mapping)

    # Check for any NaN values after mapping
    if input_data['Spending_Score'].isnull().any():
        return jsonify({"error": "Invalid value in Spending_Score"}), 400

    # Debugging: Print input data to check its structure
    print("Input Data:\n", input_data)

    # Preprocess input data (similar to your training preprocessing)
    
    # Encode categorical features using the loaded encoders
    for column in label_encoders.keys():
        if column in input_data.columns:
            try:
                # Debugging: Print original values before transformation
                print(f"Transforming column '{column}' with values: {input_data[column].unique()}")
                
                # Transform using the fitted encoder
                input_data[column] = label_encoders[column].transform(input_data[column])
                
                # Debugging: Print transformed values to verify encoding
                print(f"Encoded values for column '{column}': {input_data[column].unique()}")
            except ValueError as e:
                return jsonify({"error": f"Value error for column '{column}': {str(e)}"}), 400

    # Impute missing values for categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    input_data[list(label_encoders.keys())] = cat_imputer.fit_transform(input_data[list(label_encoders.keys())])

    # Define numerical columns for imputation
    numeric_columns = ['Work_Experience', 'Family_Size']  # Add any other numerical columns as needed

    # Impute missing values for numerical columns
    num_imputer = SimpleImputer(strategy='median')
    input_data[numeric_columns] = num_imputer.fit_transform(input_data[numeric_columns])
    
    # Convert numerical columns to integer type if necessary
    input_data[numeric_columns] = input_data[numeric_columns].astype(int)

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Return predictions as JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
