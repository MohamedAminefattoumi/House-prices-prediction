#Importing the libraries 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor
from sklearn.metrics import r2_score , mean_squared_error


#Importing the dataset
df = pd.read_csv('Housing_clean.csv')

X = df.drop("price", axis=1)  
y = df["price"].values                


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Numerical columns
num_columns = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Categorical columns
cat_columns = X.select_dtypes(include=['object']).columns.tolist()

# Encode all categorical variables
le = LabelEncoder()

# Encode binary variables (yes/no to 1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    X_train[col] = X_train[col].map({'yes': 1, 'no': 0})
    X_test[col] = X_test[col].map({'yes': 1, 'no': 0})

# Encode furnishingstatus with LabelEncoder
X_train['furnishingstatus'] = le.fit_transform(X_train['furnishingstatus'])
X_test['furnishingstatus'] = le.transform(X_test['furnishingstatus'])

# Update column lists after encoding
# After encoding, all columns become numerical, so we only need the scaler
numeric_processor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# ColumnTransformer - only numerical columns after encoding
processor = ColumnTransformer(
    transformers=[
        ('num', numeric_processor, num_columns + ['furnishingstatus'] + ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])
    ]
)

#Creating the pipeline
pipeline = Pipeline (steps=[
    ('processor ', processor),
    ('regressor', LinearRegression())    
])

# Parameters grid to tune the regressor 
params = [
    {
        'regressor': [RandomForestRegressor(random_state=0)],
        'regressor__n_estimators': [10,20,30,40,50,60,70,80,90,100],
        'regressor__criterion': ['squared_error','absolute_error','poisson'],
        'regressor__bootstrap': [True, False]
    },
    {
        'regressor': [GradientBoostingRegressor(random_state=0)],
        'regressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'regressor__n_estimators': [10,20,30,40,50,60,70,80,90,100],
        'regressor__criterion': ['friedman_mse', 'squared_error']
    },
    {
        'regressor': [SVR()],
        'regressor__kernel': ['linear', 'rbf', 'poly'],
        'regressor__C': [0.1, 1, 10, 100],
        'regressor__epsilon': [0.01, 0.1, 0.2, 0.5],
        'regressor__gamma': ['scale', 'auto']
    }

]

# GridSearchCV with 10-fold CV to find best hyperparameters
print("\nStarting GridSearchCV...")
grid = GridSearchCV(
    pipeline, 
    param_grid=params, 
    cv=10, 
    scoring='r2', 
    n_jobs=-1,
    verbose=1
)

# Training the GridSearchCV
print("Training models with GridSearchCV...")
grid.fit(X_train, y_train)

print(f"\nBest model: {grid.best_estimator_}")
print(f"Best hyperparameters: {grid.best_params_}")
print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

# Predicting on test set using the best model
y_pred = grid.best_estimator_.predict(X_test)

# Evaluation
print("R² score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Function to predict house price
def predict_house_price():
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTION")
    print("="*50)
    
    # Get user input for house features
    print("\nPlease enter the house characteristics:")
    
    # Numerical features
    area = float(input("Area (in square feet): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    stories = int(input("Number of stories: "))
    parking = int(input("Number of parking spaces: "))
    
    # Categorical features
    print("\nFeatures (yes/no):")
    mainroad = input("Near main road (yes/no): ").lower()
    while mainroad not in ['yes', 'no']:
        mainroad = input("Please enter 'yes' or 'no': ").lower()
    
    guestroom = input("Guest room (yes/no): ").lower()
    while guestroom not in ['yes', 'no']:
        guestroom = input("Please enter 'yes' or 'no': ").lower()
    
    basement = input("Basement (yes/no): ").lower()
    while basement not in ['yes', 'no']:
        basement = input("Please enter 'yes' or 'no': ").lower()
    
    hotwaterheating = input("Hot water heating (yes/no): ").lower()
    while hotwaterheating not in ['yes', 'no']:
        hotwaterheating = input("Please enter 'yes' or 'no': ").lower()
    
    airconditioning = input("Air conditioning (yes/no): ").lower()
    while airconditioning not in ['yes', 'no']:
        airconditioning = input("Please enter 'yes' or 'no': ").lower()
    
    prefarea = input("Preferred area (yes/no): ").lower()
    while prefarea not in ['yes', 'no']:
        prefarea = input("Please enter 'yes' or 'no': ").lower()
    
    print("\nFurnishing status:")
    furnishingstatus = input("furnished/semi-furnished/unfurnished: ").lower()
    while furnishingstatus not in ['furnished', 'semi-furnished', 'unfurnished']:
        furnishingstatus = input("Please enter 'furnished', 'semi-furnished' or 'unfurnished': ").lower()
    
    # Create input data
    input_data = {
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Apply the same preprocessing as training data
    # Encode binary variables (yes/no to 1/0) - same as in training
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
    
    # Encode furnishingstatus using the same label encoder from training
    input_df['furnishingstatus'] = le.transform(input_df['furnishingstatus'])
    
    # Ensure all columns are in the correct order and type
    input_df = input_df.astype(float)
    
    # Make prediction
    predicted_price = grid.best_estimator_.predict(input_df)[0]
    
    # Display result
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Predicted price: {predicted_price:,.0f} ₹")
    

# Run prediction function
if __name__ == "__main__":
    predict_house_price()
