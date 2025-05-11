# Predict_CGPA

## Project Overview
This project predicts the first-semester CGPA of international master's students to identify at-risk students early and provide timely academic support. By using machine learning models, the goal is to improve student success and optimize institutional processes.

## Project Structure
The project is organized into the following directories and files:

- **`model/`**: Contains the trained models and scripts related to model training and saving.
- **`static/`**: This folder contains static files like images, CSS, and JavaScript for the front-end of the web application.
- **`templates/`**: Contains HTML templates for rendering dynamic content in the web application.
- **`app/`**: This folder contains the Python source code for the web application logic, handling the predictions and user interaction.
- **`CP_Project_pipeline_final.ipynb`**: The Jupyter notebook that holds the final machine learning pipeline, including data preprocessing, model training, and evaluation.
- **`student-2018-2023_NEW.xlsx`**: The dataset containing historical data for students from 2018 to 2023, used to train and test the model.
- **`requirement.txt`**: The file lists all the necessary Python libraries for this project. It can be used to set up the project environment easily.
  
## Dataset Splitting
To enhance the accuracy of predictions, the dataset was split into three subsets based on the school:
- **SET**
- **SOM**
- **SERD**

Each dataset was used to train separate models, allowing the prediction to be more accurate by accounting for school-specific variations. The best-performing model for each school dataset was chosen based on predictive accuracy.

## Models Used
The following machine learning models were applied to predict first-semester CGPA using the three school-specific datasets:
- **Linear Regression**
- **Ridge, Lasso, Elastic Net** (Regularization-based models)
- **Support Vector Regression (SVR)**
- **K-Nearest Neighbors**
- **Decision Tree Regressor**
- **Random Forest Regressor**

## Comparative Model Performance
  - **SET**: Random Forest Regressor got  MSE = 0.0543, R² = 0.7825
  - **SERD**: Random Forest Regressor got MSE = 0.0506, R² = 0.6473
  - **SOM**: K-Nearest Neighbors got MSE = 0.1346, R² =  0.0198

## Feature Importance
The most important features for prediction varied depending on the school, highlighting the unique factors that influence CGPA outcomes at each institution.

## Key Benefits
- **For Universities**: Helps optimize processes like resource allocation, student enrollment, and retention.
- **Early Intervention**: Identifies at-risk students for timely academic support.
- **Institutional Improvement**: Supports data-driven decision-making to improve rankings, retention, and graduation rates.

## How to Use
1. **Install Dependencies**: Run `pip install -r requirements.txt`.
2. **Start the Web App**: Launch the application to access the interface and enter student data to predict CGPA.

## Web Platform
The prediction model is deployed on a custom web platform with an interactive interface. The platform enables real-time predictions and provides actionable insights to support students.

### Example Pages:

#### 1. **Welcome Page**
   - Users select their school (SET, SERD, or SOM) to proceed with predictions.
![image](https://github.com/user-attachments/assets/7c013d7a-3b57-4daf-8f0e-2bea0324f98d)


#### 2. **Prediction Page**
   - Users input their data (ex.CGPA for Midterm, English Score, Previous Degree, etc.) to predict their first-semester CGPA.
![image](https://github.com/user-attachments/assets/2e3895c9-eb40-409e-8993-6f2a66db1443)


#### 3. **Result Page**
   - After entering the details, the platform predicts the CGPA and provides a risk assessment (e.g., low, medium, or high risk).
![image](https://github.com/user-attachments/assets/da969f0f-a3c8-4159-afdb-1375a6914f47)

