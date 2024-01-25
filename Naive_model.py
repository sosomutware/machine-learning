import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import category_encoders as ce

# ... (previous code)

# Renaming column names
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']

# Read the CSV file
df = pd.read_csv("C:/Users/user/Downloads/adult.csv")
print(df.shape)
print(df.head())

# Assign column names to the DataFrame
df.columns = col_names

# Display the DataFrame
print(df.head())

# view summary of dataset
df.info()
# find categorical variables
categorical=[var for var in df.columns if df[var].dtype=='O']
print('Ther are {} categorical values'.format(len(categorical)))
print('The categorica are :\n\n',categorical)
print(df[categorical].head())
# check missing values in categorical variables
df[categorical].isnull().sum()
# view frequency counts of values in categorical variables
for var in categorical:
 print(df[var].value_counts())
# view frequency counts of values in categorical variables
for var in categorical:
 print(df[var].value_counts())
for var in categorical:
 print(df[var].value_counts() / np.float64(len(df)))
 # check labels in workclass variable
 print(df.workclass.unique())
 # check frequency distribution of values in workclass variable
 print(df.workclass.value_counts())
 # replace '?' values in workclass variable with `NaN`
 df['workclass'].replace('?', np.NaN, inplace=True)
 # replace '?' values in workclass variable with `NaN`
 df['workclass'].replace('?', np.NaN, inplace=True)
 # again check the frequency distribution of values in workclass variable
 print(df['workclass'].value_counts())
 # again check the frequency distribution of values in occupation variable
 print(df.occupation.value_counts())
 df['occupation'].replace('?', np.NaN, inplace=True)
 print(df.occupation.value_counts())
 # check labels in native_country variable
 df.native_country.unique()
 # check labels in native_country variable
 df.native_country.unique()
 # check frequency distribution of values in native_country variable
 print(df.native_country.value_counts())
 df['native_country'].replace('?', np.NaN, inplace=True)
 # again check the frequency distribution of values in native_country variable
 print(df.native_country.value_counts())
 print(df[categorical].isnull().sum())
 # check for cardinality in categorical variables
 for var in categorical:
     print(var, 'contains', len(df[var].unique()), 'Labels')
numerical=[var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical values '.format(len(numerical)))
print('numerical are:',numerical)
# view the numerical variables
print(df[numerical].head())
# check missing values in numerical variables
# check missing values in numerical variables
print(df[numerical].isnull().sum())
X = df.drop(['income'], axis=1)
y = df['income']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)
categorical=[var for var in X_train.columns if X_train[var].dtype=='O']
print('categrical :\n\n',categorical,)
# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical
# print percentage of missing values in the categorical variables in training set
print(X_train[categorical].isnull().mean())
# print categorical variables with missing data
for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print('missing variable are :', col, X_train[col].isnull().mean())
for df2 in [X_train, X_test]:
 df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
 df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
 df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)
# check missing values in categorical variables in X_train
print(X_train[categorical].isnull().sum())
# check missing values in categorical variables in X_test
X_test[categorical].isnull().sum()
# check missing values in X_train
print(X_train.isnull().sum())
# check missing values in X_test
print(X_test.isnull().sum())
# print categorical variables
print(categorical)
print(X_train[categorical].head())
# import category encoders
import category_encoders as ce
# encode remaining variables with one-hot encoding
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation',
'relationship',
 'race', 'sex', 'native_country'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_test.head()
X_test.shape()
cols = X_train.columns
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = gnb.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)
print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
# check class distribution in test set
print(y_test.value_counts())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
# Specificity : 0.5740
# print the first 10 predicted probabilities of two classes- 0 and 1
y_pred_prob = gnb.predict_proba(X_test)[0:10]
print(y_pred_prob)
# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Probof - >50K'])
print(y_pred_prob_df)
# print the first 10 predicted probabilities for class 1 - Probability of >50K
gnb.predict_proba(X_test)[0:10, 1]
# plot histogram of predicted probabilities
# adjust the font size
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')
# compute ROC AUC
from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
# calculate cross-validated ROC AUC

from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
# Applying 10-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# compute Average cross-validation score
print('Average cross-validation score: {:.4f}'.format(scores.mean()))