#!/usr/bin/env python
# coding: utf-8

# # Interconnect Churn Predictions
# Interconnect is a telecom company exploring predictions of when a clients would leave. Discovering the at risk of leaving clients the company plans to ffer them incentives to stay.  
# 
# Interconnect mainly provides two types of services:
# $1. Landline communication. The telephone can be connected to several lines simultaneously.
# $2. Internet. The network can be set up via a telephone line (DSL, digital subscriber line) or through a fiber optic cable.
# 
# Some other services the company provides include:
# 
# Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
# A dedicated technical support line (TechSupport)
# Cloud file storage and data backup (OnlineBackup)
# TV streaming (StreamingTV) and a movie directory (StreamingMovies)
# 
# The clients can choose either a monthly payment or sign a 1- or 2-year contract. They can use various payment methods and receive an electronic invoice after a transaction.Data being evaluated is prior to 1 Feb. 2020. 
# 
# The data source files contain contract information, client's personal data, client options of Internet services, and information about telephone services.  Each customer has a unique ID assigned. 

# In[1]:


#import libraries to be used 

#general
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

#visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

#machine learning model requirements
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve 
from sklearn import metrics
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


#models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# !pip install shap
# import shap

# In[2]:


#import files 
contract = pd.read_csv('/datasets/final_provider/contract.csv')
personal = pd.read_csv('/datasets/final_provider/personal.csv')
internet = pd.read_csv('/datasets/final_provider/internet.csv')
phone = pd.read_csv('/datasets/final_provider/phone.csv') 


# # EDA Exploration 

# In[3]:


def evaluate_file(file_path, duplicate_column=None):
    """
    Evaluates a given CSV or Excel file by:
    - Printing general file information
    - Checking for duplicate values in a specified column
    - Searching for zero values
    - Searching for empty (NaN) cells
    """

    # Print file information
    print("\n=== File Info ===")
    print(file_path.info())

    # Check for duplicates in the specified column
    if duplicate_column:
        duplicate_count = file_path.duplicated(subset=[duplicate_column]).sum()
        print(f"\n=== Duplicates in '{duplicate_column}' ===")
        print(f"Total duplicate values: {duplicate_count}")
    
    # Check for zero values
    zero_values = (file_path == 0).sum().sum()
    print(f"\n=== Zero Values ===")
    print(f"Total zero values: {zero_values}")

    # Check for empty (NaN) cells
    missing_values = file_path.isnull().sum().sum()
    print(f"\n=== Empty (NaN) Cells ===")
    print(f"Total empty cells: {missing_values}")
    
    #print a sample 
    display(file_path.sample(n=5))


# In[4]:


#Run pipeline for each of the files to evaluate issues 
evaluate_file(contract, duplicate_column="customerID")  


# In[5]:


#check for earliest date
contract['BeginDate'].min()


# In[6]:


#check for most recent customer 
begindatemax=contract['BeginDate'].max()
print(begindatemax)


# In[7]:


non_numeric_values = contract[~contract['TotalCharges'].str.replace('.', '', 1).str.isnumeric()]
non_numeric_values


# In[8]:


#there are 17 rows where the 'TotalCharges' have nonnumeric.  That is only .002% of data.  small enough to drop 
contract = contract[pd.to_numeric(contract['TotalCharges'], errors='coerce').notna()]


# In[9]:


#replace object columns with numerics
#replace yes/no with boolean numeric representation 
contract['PaperlessBilling']=contract['PaperlessBilling'].replace('No','0')
contract['PaperlessBilling']=contract['PaperlessBilling'].replace('Yes','1')

print(contract['PaperlessBilling'].value_counts())

#convert the gender datatype to an integer 
contract['PaperlessBilling'] = contract['PaperlessBilling'].astype('int')


# In[10]:


#replace 'Type' and 'PaymentMethod' with numeric representation 

#find answers in each column to be able to replace. 
print(contract['Type'].value_counts())
print(contract['PaymentMethod'].value_counts())


# In[11]:


contract['Type']=contract['Type'].replace('Month-to-month','1')
contract['Type']=contract['Type'].replace('Two year','2')
contract['Type']=contract['Type'].replace('One year','3')

contract['PaymentMethod']=contract['PaymentMethod'].replace('Electronic check','1')
contract['PaymentMethod']=contract['PaymentMethod'].replace('Mailed check','2')
contract['PaymentMethod']=contract['PaymentMethod'].replace('Bank transfer (automatic)','3')
contract['PaymentMethod']=contract['PaymentMethod'].replace('Credit card (automatic)','4')

contract['EndDate']=contract['EndDate'].replace('No','0')

#Ensure converion was correct 
print('Type values post edit', contract['Type'].value_counts())
print('Payment Method post edit', contract['PaymentMethod'].value_counts())



# In[12]:


#convert datatypes 
contract['Type']=contract['Type'].astype('int')
contract['PaymentMethod']=contract['PaymentMethod'].astype('int')
contract['TotalCharges'].fillna(0, inplace=True)  # Fill with 0
contract['TotalCharges'] = contract['TotalCharges'].astype(float)


# In[13]:


# Convert to datetime format
contract['BeginDate'] = pd.to_datetime(contract['BeginDate'])
contract['EndDate'] = pd.to_datetime(contract['EndDate'], errors='coerce')
#count how many customers have not ended 
contract['EndDate'].isna().sum()


# In[14]:


#replace all NA in 'EndDate' with beginning date
#contract['EndDate'].replace('NaT',0, inplace=True)  # Fill with 0
contract['EndDate'].fillna(begindatemax, inplace=True)

contract['EndDate'].isna().sum()


# In[15]:


# Extract useful features
contract['beginyear'] = contract['BeginDate'].dt.year
contract['beginmonth'] =contract['BeginDate'].dt.month
contract['beginday'] = contract['BeginDate'].dt.day

contract['endyear'] = contract['EndDate'].apply(lambda x: x.year if isinstance(x, pd.Timestamp) else 0)
contract['endmonth'] = contract['EndDate'].apply(lambda x: x.month if isinstance(x, pd.Timestamp) else 0)
contract['endday'] = contract['EndDate'].apply(lambda x: x.day if isinstance(x, pd.Timestamp) else 0)


# In[16]:


# Convert BeginDate and EndDate to datetime, coercing errors to NaT
contract['BeginDate'] = pd.to_datetime(contract['BeginDate'], errors='coerce')
contract['EndDate'].fillna(0, inplace=True)
contract['EndDate'] = pd.to_datetime(contract['EndDate'], errors='coerce')

# Calculate the difference in months
contract['user_timeframe'] = ((contract['EndDate'] - contract['BeginDate']) / np.timedelta64(1, 'M')).round().astype(int)

# Convert to integer (round first)
contract['user_timeframe'] = contract['user_timeframe'].round().astype(int)

# Fill NaN values in user_timeframe with a zero timedelta
contract['user_timeframe'].fillna(pd.Timedelta(seconds=0), inplace=True)
contract.head()


# In[17]:


average = np.mean(contract['user_timeframe'])
print(average)  


# In[18]:


#confirm types
print(contract.dtypes)
contract.head(3)


# In[19]:


#visualization 
# Create bins (brackets) for user_length (10 bins max)
contract['user_timeframe_bins'] = pd.cut(contract['user_timeframe'], bins=10)

# Group by bins and calculate 'total charges' $ per bin
grouped = contract.groupby('user_timeframe_bins')['TotalCharges'].mean()

# Plot
plt.figure(figsize=(10, 6))
grouped.plot(kind='bar', color='skyblue', edgecolor='black')

# Labels and Title
plt.xlabel('User Length Brackets')
plt.ylabel('Total Charges ($)')
plt.title('User Spending vs User Length')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

plt.show()


# In[20]:


contract.columns = contract.columns.str.lower()
contract.head(3)


# # Evaluate Personal file

# None of the files have duplicates, zeros, or empty cells.  Many of the column datatypes are objects rather than integers or floats which are needed for machine learning.  

# In[21]:


evaluate_file(personal, duplicate_column="customerID")  


# In[22]:


#confirm only two genders are listed
personal['gender'].value_counts()


# In[23]:


#visualization

# Group by bins and calculate 'total charges' $ per bin
partner_count = personal['Partner'].value_counts()
dependent_count=personal['Dependents'].value_counts()
gender_count=personal['gender'].value_counts()
# Create a subplot for the three bar charts
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Partner bar plot
axes[0].bar(partner_count.index, partner_count.values, color='skyblue')
axes[0].set_title('Partner Count')
axes[0].set_xlabel('Partner')
axes[0].set_ylabel('Count')

# Dependents bar plot
axes[1].bar(dependent_count.index, dependent_count.values, color='lightgreen')
axes[1].set_title('Dependents Count')
axes[1].set_xlabel('Dependents')
axes[1].set_ylabel('Count')

# Gender bar plot
axes[2].bar(gender_count.index, gender_count.values, color='lightcoral')
axes[2].set_title('Gender Count')
axes[2].set_xlabel('Gender')
axes[2].set_ylabel('Count')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[24]:


#replace yes/no with boolean numeric representation 

#replace 'Gender', 'Partner', 'Dependents' with numeric representation
#convert gender column onto a binary.  Male=1 and Female=2
personal['gender']=personal['gender'].replace('Female','2')
personal['gender']=personal['gender'].replace('Male', '1')

personal[['Partner', 'Dependents']] = personal[['Partner', 'Dependents']].replace('No', '0')
personal[['Partner', 'Dependents']] = personal[['Partner', 'Dependents']].replace('Yes', '1')

#convert the gender datatype to an integer 
personal[['Partner', 'Dependents']] = personal[['Partner', 'Dependents']].astype('int')
personal['gender']=personal['gender'].astype('int')

#confirm types
print(personal.dtypes)


# In[25]:


personal.columns = personal.columns.str.lower()
personal.head(3)


# # Review Internet File

# In[26]:


evaluate_file(internet, duplicate_column="customerID")  


# In[27]:


#tech support more important to online or streaming customers 
internet['online_tech'] = internet.apply(lambda row: 'Yes' if row['OnlineSecurity'] == 'Yes' and row['OnlineBackup'] == 'Yes' else 'No', axis=1)
internet['streaming_tech'] = internet.apply(lambda row: 'Yes' if row['StreamingTV'] == 'Yes' and row['StreamingMovies'] == 'Yes' else 'No', axis=1)
online=internet['online_tech'].value_counts()
streaming=internet['streaming_tech'].value_counts()


# In[28]:


# Convert to lists for plotting
labels = list(online.index)  # Use .index instead of .keys() if it's a DataFrame
online_values = list(online.values)  # Remove parentheses
streaming_values = list(streaming.values)  # Remove parentheses


# Define positions for bars
x = np.arange(len(labels))  # Get x locations
width = 0.4  # Bar width

plt.figure(figsize=(10, 6))

# Plot bars side by side
plt.bar(x - width/2, online_values, width=width, label='Online', color='skyblue', edgecolor='black')
plt.bar(x + width/2, streaming_values, width=width, label='Streaming', color='purple', edgecolor='black')

# Labels and title
plt.xlabel('Tech Support Users')
plt.ylabel('Number of Users')
plt.title('Users Investing in Tech Support per Service Type')
plt.xticks(ticks=x, labels=labels, rotation=45)  # Set x-axis labels
plt.legend()  # Show legend

plt.show()


# Note that the majority of users do not choose to have tech support.  The choice of online or streaming options do not effect the issue.  Potentially this is an indication for lack of trust of the company.  

# In[29]:


#replace yes/no with boolean numeric representation 
for col in internet.columns:
    internet[col] = internet[col].replace({'Yes': 1, 'No': 0})

internet.head(3)


# In[30]:


print(internet['InternetService'].value_counts())


# In[31]:


internet=internet.drop('online_tech',axis=1)


# In[32]:


internet=internet.drop('streaming_tech',axis=1)


# In[33]:


#replace 'InernetService' with numeric representation 
internet['InternetService']=internet['InternetService'].replace('Fiber optic','1')
internet['InternetService']=internet['InternetService'].replace('DSL','2')

#confirm types
print(internet.dtypes)


# In[34]:


internet['InternetService']=internet['InternetService'].astype(int)
internet.dtypes


# In[35]:


#make all columns be lower case
internet.columns = internet.columns.str.lower()
internet.head(3)


# # Evaluate Phone File

# In[36]:


evaluate_file(phone, duplicate_column="customerID") 


# In[37]:


#replace MultipleLines with numeric representation 
phone['MultipleLines']=phone['MultipleLines'].replace('Yes','1')
phone['MultipleLines']=phone['MultipleLines'].replace('No','0')

#force all columns IDs to be lowercase
phone.columns = phone.columns.str.lower()

#confirm types
print(phone.dtypes)


# In[38]:


#make some visualizations 
line_research=phone['multiplelines'].value_counts()
plt.figure(figsize=(10, 6))
line_research.plot(kind='bar', color='green', edgecolor='green')

# Labels and Title
plt.xlabel('Multiple Lines')
plt.ylabel('Number of Users')
plt.title('Do Users Have Multiple Lines? ')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

plt.show()


# ## Project Plan 

# # Merge DF

# In[39]:


# Randomly sample 5,000 rows from each dataframe (if they have at least 5k rows)
internet_sample = internet.sample(n=5000, random_state=42, replace=False)
contract_sample = contract.sample(n=5000, random_state=42, replace=False)
phone_sample = phone.sample(n=5000, random_state=42, replace=False)
personal_sample =personal.sample(n=5000, random_state=42, replace=False)


# In[40]:


#entire = internet_sample.merge(contract_sample, on='customerid', how='outer').merge(phone_sample, on='customerid', how='outer').merge(personal_sample, on='customerid', how='outer')
entire = internet.merge(contract, on='customerid', how='outer').merge(personal, on='customerid', how='outer').merge(phone, on='customerid', how='outer')


# In[41]:


#look for any changes post merge
evaluate_file(entire, duplicate_column="customerid") 


# In[42]:


#drop columns which will not be needed any longer 
entire=entire.drop(['user_timeframe_bins'], axis=1)


# In[43]:


#ther are 11529 NaN values
#Replace NaN values with 0
entire = entire.fillna(0)


# In[44]:


evaluate_file(entire, duplicate_column="customerid") 


# In[45]:


# Ensure 'enddate' is in datetime format
entire['enddate'] = pd.to_datetime(entire['enddate'], errors='coerce')

# converting to int
entire['multiplelines'] = entire['multiplelines'].astype(int)


# In[46]:


#use feature engineering to create new useful columns. 

#trust (if they have techsupport, deviceprotection, and online security. are still customer?)
entire['trust']=((entire['techsupport'] == '1') & 
              (entire['deviceprotection'] == '1') & 
              (entire['onlinesecurity'] == '1') & 
              (entire['endyear'] == '0')).astype(int)  # Convert to 1/0

entire['trust'].value_counts()


# In[47]:


#determine avg. of spending. rank if high/low spend.  see relationship to churn
av_spend=entire['totalcharges'].mean()
print(av_spend)
entire['type_of_spender']=(entire['totalcharges']>av_spend).astype(int)
print(entire['type_of_spender'].value_counts())


# In[48]:


#create with total spent and customers index 
sns.set_style("white")
sns.kdeplot(x=entire.index, y=entire['totalcharges'], cmap="Reds", fill=True)
plt.xlabel('Users')
plt.ylabel('Total Charges')
plt.title('Payment Ranges of Customers')
plt.show()


# In[49]:


#drop columns which will not be needed any longer 
entire=entire.drop(['begindate','customerid','beginyear', 'beginmonth', 'beginday','endmonth', 'endday','type_of_spender', 'trust'], axis=1)


# In[50]:


#examine class balance. everything counted in '0' indiates still current customer. 
entire['endyear'].value_counts()


# In[51]:


#add 'churn' column 
entire['churn'] = ((entire['endyear'] != 0) & (entire['endyear'] != 2025)).astype(int)


# In[52]:


entire.head()


# In[53]:


# Sort by 'enddate' (ascending) and then by 'monthly charges' (ascending)
entire = entire.sort_values(by=['enddate', 'monthlycharges'], ascending=[True, True])


# In[54]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Set 'enddate' as index (if not already)
entire.set_index('enddate', inplace=True)

# Select the column representing churn counts over time (change 'Churn_Count' to the actual column name)
if 'churn' in entire.columns:
    decomposed = seasonal_decompose(entire['churn'], model='additive', period=12)  

    # Plot results
    plt.figure(figsize=(6, 8))
    plt.subplot(311)
    plt.plot(decomposed.trend, label="Trend")
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(312)
    plt.plot(decomposed.seasonal, label="Seasonality")
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(313)
    plt.plot(decomposed.resid, label="Residuals")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("Error: Column 'Churn_Count' not found in dataset.")


# In[55]:


entire.columns


# In[56]:


entire = entire.dropna().reset_index(drop=True)


# In[57]:


#split source data into test, training and validation set of 6:2:2
#create split of 60% to training and 40% assigned as temp 
entire_train, entire_temp=train_test_split(entire, test_size=0.4, random_state=54321)
#create split from beta_temp to _validation and _test dataframes. Sources 20% of data to each. 
entire_valid, entire_test=train_test_split(entire_temp, test_size=0.5, random_state=54321)


# In[58]:


#define variables for training 
features_train = entire_train.drop(['churn','user_timeframe'],axis=1)
target_train = entire_train['churn']
#define variables for testing
features_test = entire_test.drop(['churn','user_timeframe'],axis=1)
target_test = entire_test['churn']
#define variables for validation 
features_valid = entire_valid.drop(['churn','user_timeframe'],axis=1)
target_valid = entire_valid['churn']


# In[59]:


from sklearn.utils import shuffle

def upsample_entire(df, column, values, num_rows_to_add):

    # Ensure the column is numeric 
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Filter rows where the column has the target values (2019 or 2020)
    df_target = df[df[column].isin(values)]

    # Check if df_target is empty
    if df_target.empty:
        raise ValueError(f"No rows found in '{column}' with values {values}. Upsampling cannot proceed.")

    # Calculate how many times to duplicate the rows
    num_target_rows = len(df_target)
    repeat_factor = num_rows_to_add // num_target_rows  # Integer division
    remainder = num_rows_to_add % num_target_rows  # Extra rows needed

    # Duplicate rows
    df_upsampled = pd.concat([df_target] * repeat_factor, ignore_index=True)

    # If remainder exists, sample additional rows
    if remainder > 0:
        df_extra = df_target.sample(n=remainder, replace=True, random_state=12345)
        df_upsampled = pd.concat([df_upsampled, df_extra], ignore_index=True)

    # Combine with original dataset
    df_final = pd.concat([df, df_upsampled], ignore_index=True)

    # Shuffle to maintain randomness
    df_final = shuffle(df_final, random_state=12345)

    return df_final


# In[60]:


# Add exactly 3,294 rows where 'endyear' is 2019 or 2020
entire = upsample_entire(features_train, 'endyear', [2019, 2020], 3294)
print(f"Features_train upsampled DataFrame has {len(entire)} rows.")


# In[61]:


features_train = features_train.drop(['endyear'], axis=1)
features_test = features_test.drop(['endyear'], axis=1)
features_valid = features_valid.drop(['endyear'], axis=1)


# In[62]:


scaler = StandardScaler()  # Initialize the scaler

# Fit and transform training data
features_train = scaler.fit_transform(features_train)

# Only transform test data (do NOT fit again!)
features_test = scaler.transform(features_test)


# In[63]:


features_train.shape


# <div class="info">
# <b>Student comment V1</b> <a class="tocSkip"></a>
# 
# original n of 1226 was the len of limiting factor of target_train.  Features_train will need to be sampled since it was  upsampled to account for the class imbalance 
# </div>

# In[64]:


# Define models and their respective parameter grids
models_and_params = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'random_state': [42]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(),
        'param_grid': {
            'max_depth': [3, 5, 10],
            'random_state': [42]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'random_state': [42]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(),
        'param_grid': {}
    },
    'XGBClassifier': {
        'model': xgb.XGBClassifier(),
        'param_grid': {} 
    },
    'LGBMClassifier': {
        'model': lgb.LGBMClassifier(),
        'param_grid': {}
    },
    'CatBoost': {
        'model': CatBoostClassifier(),
        'param_grid': {}
    },
}

def train_models(models_and_params, features_train, target_train):
    trained_models = {}

    for model_name, config in models_and_params.items():
        print(f"\nRunning GridSearchCV for {model_name}...")

        # Hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['param_grid'],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(features_train, target_train)

        # Store the best model
        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model
        print(f"Best {model_name} Parameters: {grid_search.best_params_}")

    return trained_models

def evaluate_models(trained_models, features_train, target_train, features_test, target_test):
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        evaluate_classification_model(model_name, model, features_train, target_train, features_test, target_test)

def evaluate_classification_model(model_name, model, features_train, target_train, features_test, target_test):
    eval_stats = {}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for dataset_type, features, target in [('train', features_train, target_train), ('test', features_test, target_test)]:
        eval_stats[dataset_type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]  # Use the correct dataset

        # Compute metrics
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [f1_score(target, pred_proba >= threshold) for threshold in f1_thresholds]
        fpr, tpr, _ = roc_curve(target, pred_proba)
        precision, recall, _ = precision_recall_curve(target, pred_proba)

        # Aggregate results
        eval_stats[dataset_type]['Accuracy'] = accuracy_score(target, pred_target)
        eval_stats[dataset_type]['F1'] = f1_score(target, pred_target)
        eval_stats[dataset_type]['ROC AUC'] = roc_auc_score(target, pred_proba)
        eval_stats[dataset_type]['APS'] = average_precision_score(target, pred_proba)

        color = 'blue' if dataset_type == 'train' else 'green'

        # F1 Score Plot
        ax = axs[0]
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{dataset_type}, max={max(f1_scores):.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.set_title(f'F1 Score ({model_name})')

        # ROC Curve
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{dataset_type}, AUC={roc_auc_score(target, pred_proba):.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.set_title(f'ROC Curve ({model_name})')

        # Precision-Recall Curve
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{dataset_type}, AP={average_precision_score(target, pred_proba):.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.set_title(f'Precision-Recall Curve ({model_name})')

    df_eval_stats = pd.DataFrame(eval_stats).round(2)
    df_eval_stats = df_eval_stats.reindex(index=['Accuracy', 'F1', 'APS', 'ROC AUC'])

    print(df_eval_stats)
    plt.show()

# Run training and evaluation separately
trained_models = train_models(models_and_params, features_train, target_train)
evaluate_models(trained_models, features_train, target_train, features_test, target_test)


# In[65]:


def find_best_model_by_auc(trained_models, features_test, target_test):
    """Find the model with the highest ROC-AUC score on the test set."""
    best_model = None
    best_model_name = None
    best_auc = 0

    for model_name, model in trained_models.items():
        pred_proba = model.predict_proba(features_test)[:, 1]
        auc = roc_auc_score(target_test, pred_proba)

        print(f"{model_name} Test ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_auc:.4f}")
    return best_model, best_model_name

# Find the best model based on ROC-AUC
best_model, best_model_name = find_best_model_by_auc(trained_models, features_test, target_test)

# Compute ROC-AUC on the validation set using the best model
best_val_pred_proba = best_model.predict_proba(features_valid)[:, 1]
auc_roc_val = roc_auc_score(target_valid, best_val_pred_proba)

print(f"AUC-ROC Score for Best Model ({best_model_name}) on Validation Set: {auc_roc_val:.4f}")


# # Conclusion: 
# Interconnect now has a successful model to discover the users at risk of leaving.  
# 
# Analyzing the data provided some interesting insights. Data provided was from earliest customer on 2013-10-01 to 202-02-01. Average time of a customer was 32.42 months. Average user spent $2,036 in total charges.  However, the longer a user was a customer the more they paid in total charges.User data provided was equally split between male and female.  In given informatio majority of users (around 5k) do not have dependents.  Yet nearly half of users have multiple lines on their accounts.  Perhaps an indication that majority of users have business lines.  Overall most users (around 8k) did not elect for any type of tech support (security or backup).  Streaming customrs are more likely to invest in tech support over online customers.  
# 
# Following models were evaluated: Random Forest, Decision Tree, Gradient Boosting, Logistic Regression,XGB, LGBM, and CatBoost. Results listed below are arranged per model listed above on the test model.  AUC-ROC was .84,.84,.85,.83,.83,.84,.85.  Precision was .62,.62,.68,.66,.65,.66,.69.  Accuracy was .80,.80,.82,.80,.79,.80,.81.  F1 score was .58,.58,.62,.57,.58,.60,.60.  Overall ranking of models to predict based on ROC would be GradientBoosting and CatBoost tied as the best.  Second place is a three way tie with Random Forest, LGBM, and Decision Treel.  Tied for third most effective is XGB and Logisitc Regression.  
# 
# Chose to evaluate with AUC-ROC, F1, Accurary, and Precision-Recall to see which would be the best measurment. F1 Score balances precision and recall by penalizing extreme values.  F1 does not differentiate between false positives and false negatives at an alarmingly high rate.  So we would not want to use. In our data we balanced between customers who had churned and customers who had remained. Precision-Recall Curve measures the trade-off between Precision and Recall at various thresholds. Works best on highly imbalanced datasets so it is not good for our data. Accuracy measures the percentage of correctly classified instances. Best for a balanced dataset (such as ours). ROC Curve measures the trade-off between True Positive Rate (TPR, Sensitivity/Recall) and False Positive Rate (FPR) at various classification thresholds.  The closer to 1 indicates a perfect model (highest achieved by Gradient Boosting at .85).  Either accuracy or AUC-ROC could be used to measure the success of the model.  Ranking of models is mostly same between accuracy or AUC-ROC.  Only note is that in using accuracy XBG joins second place to make a four way tie. 
# 

# Final Report
# 
# All steps described in the plan were performed.  All four source files were uploaded. A function was crafted to check each file for duplicates, zero values, empty cells, and print a sample.  Some exploratory research was done to learn more regarding each data frame.  Finding the earliest date of customer aquisition and most recent date.  Confirming the number and types of genders being listed.  Determie the spending of average user.  Search for nonnumric results in the priority column of 'totalcharge.  Once it was discovered there was only .0002% of nonumeric data a decision was made to delete those.  There were many rows with NA listed for 'enddate' but decided to replace with the max value of enddate.  Preparation for machine learning began.  Many columns contained information with datatypes of object or boolean values. These were replaced with numerics so the datatypes could be acceptable for machine learning (int64 or float). 
# 
# All cleaned EDA files were combined into one dataframe on common column of 'customerid'.  Performing this step created lots of zero and Nan values to be adressed.  Choose to replace all Nan values with zereos.  Performed some feature engineering. A column was added of boolean values to represent whether or not a customr had churned.  Anothr column was created to show the determinant of each user as a high/low spender compared to the average.  Data was checked for seasonal trends but none were discovered.  
# 
# Machine Learning began.  The entire dataframe was split into training, test, and validation dataframes.  The features and target were identified for each of the seperated dataframes.  This step showed that there was a large class imbalance between churned customers and present customers. The next step was to upample rows in features_train for users who had churned to create class balance.  The train and test data were scaled. To prdict whether or not users would churn was identified as a classifiction problem.  The models were chosen with parameters to evaluate (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression,XGB, LGBM, and CatBoost).  A loop was created to use GridSearch to find best hyperparameters (for applicable models).  The defined loop fit the training and test data to each of the models.  Each model was evaluated for precision, f1 score, accuracy, and auc-roc.  
# 
# Difficulties were encountred preforming the project. Some I encountered are listed below:
# Initially I incorrectly identified the issue as a regression problem.  This generated poor results when testing on regression models. 
# 
# I chose to fill in NAN of 'enddate'originally with a timestamp for the current data.  My idea was that then I could sort to say that all the rows with todays date were still customers.  However that overly complicated the task since the date wouold change everytime the project was run and need to have code changed in the later part of the project.  This also caused data leakage so original models were overfit. Eventually chose to fill them with the maximum value of the 'begindate'.  
# 
# Mistakenly scaled and upsampled entire dataframe of combined files rather than only the training dataframe.This was corrected to only be the features_train df after train_test_split had been performed.  
# 
# Initially took that the 5,000 rows present in one of the files was the limiting factor of data to include.  Adjusted so that all data was used and addressed zeros and NANs created by utilizing all of the data provided. 
# 
# It was difficult to create a loop to evauate model parameters and perform classification taks with graphs that did not take hours.  Eventually accomplished by splitting into multiple functions.   
# 
# I thought a trend would be obviously by seasonality.  At first the graphs looked indecipherable once evaluated with seasonal_decompose.  This was because the churn date had been bouncing back and forth between actual churn dates and when I had imposed 2025 as being a churn date for users who were still customers.  As described above that was replaced with the 'begindate'.  Then I chose to sort the combined dataframe in acending values by 'enddate' and then by 'monthly charges'.  This made the graph not have static of values bouncing back and forth.  The graphs continue to look odd though.  Unsure how to correct or any insights which could be gained from them.  
# 
# Once models were run some of my results were .95 or 1.0 ROC-AUC.  This was rare and not to be expected.  It took the help of a tutor in review to realize the issue.  I performed train_test_split. However when I defined features/targt for train/test/validation I was assigning it from the dataframe 'entire' rather than from 'enitre_test/train/valid'.  So every model was running the exact same data for train and test.  Error was corrected before submission of task for review.  
# 
# Final model used was GradientBoosting.  This consistnelty had th highest ROC-AUC of 0.8514. Project contained some surprising insights.  It was concerning to me tha the longer a user was a customer the more they ended up paying. That is poor customer servie and understandable why customers would churn.  It was strange to see that most users did not claim dependents but that they had multiple lines.  Is the company mainly providing business lines? Seems unlikely since most business phones in practice are paid directly by employer.  Why would people need multiple lines if not business? Is there another negative tax consequence that users wouold not want to report having dependents? This is a contiued unknown from given data.  Overall I am appreciative that this project gave ample opportuity to demonstrate knowledge and learn from my mistakes.  I feel very joyful to have been able to create high scoring results.  
