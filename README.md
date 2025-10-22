# 🏠 House Price Prediction System

A comprehensive machine learning system for predicting house prices based on various property characteristics. This project implements a complete data science pipeline from data exploration to model deployment.

## 📊 Project Overview

This project uses machine learning algorithms to predict house prices based on features like area, number of bedrooms, bathrooms, location amenities, and furnishing status. The system achieves a **60.69% cross-validation accuracy** with an **R² score of 49.77%**.

## 🚀 Features

- **Complete Data Pipeline**: From raw data to predictions
- **Multiple ML Models**: Random Forest, Gradient Boosting, SVR, Linear Regression
- **Hyperparameter Optimization**: GridSearchCV for best model selection
- **Interactive Prediction**: User-friendly prediction interface
- **Data Visualization**: Comprehensive EDA with plots and correlation analysis
- **Robust Preprocessing**: Outlier detection, encoding, and normalization

## 📁 Project Structure

```
House prediction 2/
├── Dataoverview.py          # Data exploration and preprocessing
├── train.py                 # Model training and hyperparameter tuning
├── main.py                  # Prediction interface
├── Housing.csv              # Original dataset
├── Housing_clean.csv        # Preprocessed dataset
├── housing_model.pkl        # Trained model (generated)
├── label_encoder.pkl        # Label encoder (generated)
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## 📋 Usage

### 1. Data Exploration and Preprocessing
```bash
python Dataoverview.py
```
This script performs:
- Data quality assessment
- Outlier detection and treatment
- Categorical variable encoding
- Correlation analysis
- Data visualization

### 2. Model Training
```bash
python train.py
```
This script:
- Trains multiple ML models
- Performs hyperparameter optimization
- Saves the best model and encoder
- Provides interactive prediction interface

### 3. Price Prediction
```bash
python main.py
```
This script provides:
- Interactive prediction interface
- Batch prediction from CSV files
- Model loading and inference

## 📊 Dataset Information

The dataset contains **545 house records** with the following features:

### Numerical Features
- **area**: House area in square feet
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **stories**: Number of stories
- **parking**: Number of parking spaces

### Categorical Features
- **mainroad**: Connected to main road (yes/no)
- **guestroom**: Guest room available (yes/no)
- **basement**: Basement available (yes/no)
- **hotwaterheating**: Hot water heating (yes/no)
- **airconditioning**: Air conditioning (yes/no)
- **prefarea**: Preferred area (yes/no)
- **furnishingstatus**: furnished/semi-furnished/unfurnished

### Target Variable
- **price**: House price in Indian Rupees (₹)

## 🤖 Model Performance

### Best Model: Random Forest Regressor
- **Cross-validation Score**: 60.69%
- **R² Score**: 49.77%
- **RMSE**: ₹1,245,095
- **Hyperparameters**:
  - `n_estimators`: 70
  - `criterion`: absolute_error
  - `bootstrap`: True

### Model Comparison
| Model | R² Score | RMSE | Cross-validation |
|-------|----------|------|------------------|
| Random Forest | 0.4977 | 1,245,095 | 0.6069 |
| Gradient Boosting | 0.6034 | 1,129,851 | 0.6034 |
| SVR | - | - | - |
| Linear Regression | - | - | - |

## 📈 Data Insights

### Key Findings
1. **Area** has the strongest correlation with price
2. **Air conditioning** significantly affects house value
3. **Preferred area** location increases property value
4. **Furnishing status** impacts price prediction

### Data Quality
- **No missing values** in the dataset
- **Outliers detected and treated** in area feature
- **All categorical variables properly encoded**

## 🔧 Technical Implementation

### Data Preprocessing
1. **Outlier Detection**: IQR method for area feature
2. **Encoding**: Binary encoding for yes/no variables, LabelEncoder for furnishing status
3. **Normalization**: StandardScaler for numerical features

### Model Selection
- **GridSearchCV** with 10-fold cross-validation
- **236 hyperparameter combinations** tested
- **Multiple algorithms** compared simultaneously

### Pipeline Architecture
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Hyperparameter Tuning → Best Model Selection
```

## 📱 Usage Examples

### Interactive Prediction
```python
# Example input
Area: 7420 square feet
Bedrooms: 4
Bathrooms: 2
Stories: 3
Main road: yes
Guest room: no
Basement: no
Hot water heating: no
Air conditioning: yes
Parking: 2
Preferred area: yes
Furnishing status: furnished

# Predicted output
Predicted price: ₹13,300,000
```

### Batch Prediction
```python
# Load CSV with house features
df = pd.read_csv('houses_to_predict.csv')
# Run batch prediction
batch_prediction('houses_to_predict.csv')
```

## 🔍 Code Structure

### Dataoverview.py
- Data loading and initial exploration
- Statistical analysis and visualization
- Outlier detection and treatment
- Categorical variable encoding
- Correlation analysis

### train.py
- Data splitting and preprocessing
- Model training with GridSearchCV
- Hyperparameter optimization
- Model evaluation and selection
- Interactive prediction function

### main.py
- Model loading and inference
- User interface for predictions
- Batch prediction capabilities
- Error handling and validation

## 🚀 Future Improvements

1. **Feature Engineering**: Create new features from existing ones
2. **Advanced Models**: Implement XGBoost, LightGBM
3. **Deep Learning**: Neural networks for complex patterns
4. **Web Interface**: Flask/Django web application
5. **API Development**: REST API for model serving
6. **Real-time Data**: Integration with real estate APIs

## 📊 Performance Metrics

### Model Evaluation
- **Mean Absolute Error (MAE)**: ₹1,245,095
- **Root Mean Square Error (RMSE)**: ₹1,245,095
- **R² Score**: 0.4977 (49.77% variance explained)
- **Cross-validation Score**: 0.6069 (60.69%)

### Business Impact
- **Prediction Accuracy**: 60.69% cross-validation accuracy
- **Error Range**: ±₹1.2M average prediction error
- **Model Reliability**: Consistent performance across different house types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Dataset source: [Housing Dataset](https://www.kaggle.com/datasets/housing)
- Scikit-learn documentation
- Python data science community



⭐ **Star this repository if you found it helpful!**
