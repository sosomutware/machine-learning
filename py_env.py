# # Import the following libraries and give them alias names:
# import pandas as pd  # Data manipulation
# # import numpy as np  # Data manipulation
# import matplotlib.pyplot as plt  # Visualization
# import seaborn as sns  # Visualization
#
# # Assuming the dataset is a CSV file named 'insurance.csv'
# insurance_data = pd.read_csv(f"C:/Users/user/PycharmProjects/assignment/insurance.csv")
#
# # Print the first 5 rows of the dataset
# print(insurance_data.head())
#
# # Assuming you want to plot a scatter plot of 'age' against 'charges'
# plt.scatter(insurance_data['age'], insurance_data['sex'])
# plt.xlabel('Age')
# plt.ylabel('sex')
# plt.title('Scatter Plot of Age vs sex')
# plt.show()
#
# # Display basic statistics of the dataset
# print(insurance_data.describe())
#
# # Check for missing values in the dataset
# print(insurance_data.isnull().sum())
#
# # Handle categorical variables using one-hot encoding
# insurance_data_encoded = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'])
#
# # Plot the correlation matrix
#
# correlation_matrix = insurance_data_encoded.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()
#
# # Check for duplicate rows in the dataset
# duplicate_rows = insurance_data_encoded[insurance_data_encoded.duplicated()]
# print("Duplicate rows:\n", duplicate_rows)







# Import the following libraries and give them alias names:
import pandas as pd  # Data manipulation
from sklearn.preprocessing import OneHotEncoder  # For encoding categorical variables
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Visualization

# Assuming the dataset is a CSV file named 'insurance.csv'
insurance_data = pd.read_csv(f"C:/Users/user/PycharmProjects/assignment/insurance.csv")

# Print the first 5 rows of the dataset
print(insurance_data.head())

# Assuming you want to plot a scatter plot of 'age' against 'sex'
plt.scatter(insurance_data['region'], insurance_data['charges'])
plt.xlabel('region')
plt.ylabel('charges')
plt.title('Scatter Plot of Age vs Sex')
plt.show()

# Display basic statistics of the dataset
print(insurance_data.describe())

# Check for missing values in the dataset
print(insurance_data.isnull().sum())

# Extract categorical columns
categorical_columns = insurance_data.select_dtypes(include=['object']).columns

# Encode categorical variables using OneHotEncoder
#encoder = OneHotEncoder(sparse_output=True)
#encoder = OneHotEncoder(sparse=False, drop='first') this was giving me an error
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(insurance_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))


# Concatenate the encoded data with the original data
insurance_data_encoded = pd.concat([insurance_data, encoded_df], axis=1)

# Drop the original categorical columns
insurance_data_encoded.drop(categorical_columns, axis=1, inplace=True)

# Plot the correlation matrix
correlation_matrix = insurance_data_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Check for duplicate rows in the dataset
duplicate_rows = insurance_data_encoded[insurance_data_encoded.duplicated()]
print("Duplicate rows:\n", duplicate_rows)
