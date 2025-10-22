#Importing libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 

#Importing the dataset 
df = pd.read_csv ('Housing.csv')

#General information about the dataset
print (df.info())

#Shape of the dataset
print (f"Dataset contains (rows,columns): {df.shape} ")

#Statistical description
print("\n ====Statistical description==== ")
print(df.describe())


#Checking the null values
print ("\n ====Null values==== ")
print (df.isna().sum()) #Shows the total number of null values per each column


#Display unique values in each column to detect abnormal values
print ("\n ====Unique values==== ")
for col in df.columns :
    print (f"Column {col}: ", end=" ")
    print(df[col].unique())

# Boxplot showing the distribution of area and detect outliers

plt.figure (figsize=(12,6))
plt.title (" The dispersion of area")
plt.boxplot (df['area'])
plt.ylabel ("Area in Square feet (1 ft^2 = 0.09 m^2)")
plt.show ()

# Replacing outliers in 'area' column with median value 
print("\n ====Outlier Treatment==== ")
q1 = df['area'].quantile(0.25) #1st quantile
q3 = df['area'].quantile(0.75) #2nd quantile
iqr = q3 - q1 #Interquantile Range

lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5*iqr #Upper fence (Upper Whisker)

print(f"Outliers detected: {len(df[(df['area'] < lower_whisker) | (df['area'] > upper_whisker)])}")

# Replace outliers with median
df.loc[df['area'] > upper_whisker, 'area'] = df['area'].median()
df.loc[df['area'] < lower_whisker, 'area'] = df['area'].median()

print(f"Outliers after treatment: {len(df[(df['area'] < lower_whisker) | (df['area'] > upper_whisker)])}")

# Encoding categorical variables
print("\n ====Variable Encoding==== ")
df_encoded = df.copy()

# Binary encoding for yes/no variables
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})

# Encoding for furnishingstatus (3 categories)
le = LabelEncoder()
df_encoded['furnishingstatus'] = le.fit_transform(df_encoded['furnishingstatus'])

print("Variables encoded successfully")
print("\nEncoded data preview:")
print(df_encoded.head())

# Boxplot after outlier treatment
print("\n ====Boxplot Visualization==== ")
plt.figure(figsize=(12,6))
plt.title("Area Distribution After Outlier Treatment")
plt.boxplot(df['area'])
plt.ylabel("Area in Square feet (1 ft^2 = 0.09 m^2)")
plt.show()

# Correlation matrix visualization
print("\n ====Correlation Analysis==== ")
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Price correlation analysis
price_corr = correlation_matrix['price'].sort_values(ascending=False)
print("\nCorrelation with price:")
print(price_corr)

#Saving the clean data in a new dataset
df.to_csv("Housing_clean.csv", index=False)

print("\n ====SUMMARY==== ")
print("✅ Outlier treatment (lower and upper)")
print("✅ Categorical variable encoding")
print("✅ Boxplot visualization")
print("✅ Correlation matrix analysis")
print("✅ Clean data saved successfully")
print("✅ Data ready for next steps")


