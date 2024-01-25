\# # Import the following libraries and give them alias names:
# import pandas as pd  # Data manipulation
# import numpy as np  # Data manipulation
# import matplotlib.pyplot as plt  # Visualization
# import seaborn as sns  # Visualization
#
# # Assuming the dataset is a CSV file named 'insurance.csv'
# df = pd.read_csv(f"C:/Users/user/PycharmProjects/assignment/insurance.csv")
# print(df.shape)
# print(df.head())
# print(df.info())
# print('NULL VALUES ARE:', df.isnull())
# print('NULL VALUES ARE:', df.isnull().sum())
# print('DUPLICATES ARE FREQUENCY:', df.duplicated().sum())
# categorical = [var for var in df.columns if df[var].dtype == 'O']
# print('non numeric features are:', categorical)
# numerical = [vr for vr in df.columns if df[vr].dtype != 'O']
# print('numerical features are:', numerical)
# cat_columns = ['sex', 'smoker', 'region']
# # corr=df.corr()
# # sns.heatmap(corr,cmap='Viridis')
# df_encoded = pd.get_dummies(data=df, columns=cat_columns)
# corr1 = df_encoded.corr()
# sns.heatmap(corr1, cmap='viridis', annot=True)
# plt.show()

# Import the following libraries and give them alias names:
import pandas as pd  # Data manipulation
import numpy as np  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Visualization

# Assuming the dataset is a CSV file named 'insurance.csv'
df = pd.read_csv(f"C:/Users/user/PycharmProjects/assignment/insurance.csv")
print(df.shape)
print(df.head())
print(df.info())
print('NULL VALUES ARE:', df.isnull())
print('NULL VALUES ARE:', df.isnull().sum())
print('DUPLICATES ARE FREQUENCY:', df.duplicated().sum())
categorical = [var for var in df.columns if df[var].dtype == 'O']
print('non numeric features are:', categorical)
numerical = [vr for vr in df.columns if df[vr].dtype != 'O']
print('numerical features are:', numerical)
cat_columns = ['sex', 'smoker', 'region']
# corr=df.corr()
# sns.heatmap(corr,cmap='Viridis')
df_encoded = pd.get_dummies(data=df, columns=cat_columns)
corr1 = df_encoded.corr()
'''sns.heatmap(corr1, cmap='viridis', annot=True, cbar=True)
plt.show()'''
# sns.scatterplot(data=df, x='sex', y='charges')
# sns.violinplot(data=df, x='sex', y='charges')
# sns.boxplot(data=df, x='sex', y='charges')
# sns.histplot(data=df, x='sex', y='charges')
# sns.histplot(df['charges'], bins=30, color='r')
charges_nor = np.log10(df['charges'])
f = plt.figure(figsize=(16, 5))
ax = f.add_subplot(121)
sns.histplot(df['charges'], bins=30, color='r', ax=ax)
ax = f.add_subplot(122)
sns.histplot(charges_nor, bins=30, color='b')
df['charges'] = np.log10(df['charges'])
plt.show()
