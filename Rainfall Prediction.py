#importing libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf
import itertools
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.gridspec as gridspec
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions


# Get data from data set
weather_data = pd.read_csv('C:\\Users\\prana\\PycharmProjects\\Data science project\\weatherAUS.csv')
print(weather_data.head(20))
# print(weather_data.shape)
print(weather_data.info())

#Both "RainToday" and "RainTomorrow" are categorical variables indicating whether it will rain or not (Yes/No).
#We'll convert them into binary values (1/0) for ease of analysis.

weather_data['RainToday'] = weather_data['RainToday'].map({'No': 0, 'Yes': 1})
weather_data['RainTomorrow'] = weather_data['RainTomorrow'].map({'No': 0, 'Yes': 1})

print(weather_data.head(20))

#Next, we will check whether the dataset is imbalanced or balanced.
#If the dataset is imbalanced, we need to undersample majority or oversample minority to balance it.

figure = plt.figure(figsize = (12,10))
weather_data.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['#a4c639','#5d8aa8'], alpha = 0.9, rot=0)
plt.title('RainTomorrow: Unbalanced Dataset with No(0) and Yes(1) Indicators')
plt.show()

# Handling Class Imbalance
# Separate data based on RainTomorrow value
no_rain = weather_data[weather_data['RainTomorrow'] == 0]
yes_rain = weather_data[weather_data['RainTomorrow'] == 1]

# Oversample the minority class ('yes') to balance the dataset
yes_oversampled = resample(yes_rain, replace=True, n_samples=len(no_rain), random_state=123)

# Concatenate the oversampled 'yes' data with the 'no' data
over_sampled = pd.concat([no_rain, yes_oversampled])

#Plotting
fig = plt.figure(figsize = (12,10))
over_sampled.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['#a4c639','#5d8aa8'], alpha = 0.9, rot=0)
plt.title('After oversampling, RainTomorrow Indicator No(0) and Yes(1) (Balanced Dataset)')
plt.show()

# Check Missing Data Pattern in Training Data
sns.heatmap(over_sampled.isnull(), cbar=False, cmap='PuBu')

over_sampled.select_dtypes(include=['object']).columns

# Calculate total missing values and percentage of missing values for each column
total_missing = over_sampled.isnull().sum().sort_values(ascending=False)
#Calculate percentage of missing values
percent_missing = (over_sampled.isnull().sum() / len(over_sampled)).sort_values(ascending=False)
#Concatenate total and percentage missing values into a weather_data
missing_data_summary = pd.concat([total_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
# Display the first 10 rows of the missing data summary
missing_data_summary.head(10)


# Impute categorical variable 'Location' with its mode
over_sampled['Location'] = over_sampled['Location'].fillna(over_sampled['Location'].mode()[0])

# Impute categorical variable 'WindGustDir' with its mode
over_sampled['WindGustDir'] = over_sampled['WindGustDir'].fillna(over_sampled['WindGustDir'].mode()[0])

# Impute categorical variable 'WindDir9am' with its mode
over_sampled['WindDir9am'] = over_sampled['WindDir9am'].fillna(over_sampled['WindDir9am'].mode()[0])

# Impute categorical variable 'WindDir3pm' with its mode
over_sampled['WindDir3pm'] = over_sampled['WindDir3pm'].fillna(over_sampled['WindDir3pm'].mode()[0])

#dictionary
label_Encoders = {}
# Iterate over columns with object dtype and encode them
for col in over_sampled.select_dtypes(include=['object']).columns:
    label_Encoders[col] = LabelEncoder()
    # Fit and transform the column
    over_sampled[col] = label_Encoders[col].fit_transform(over_sampled[col])

warnings.filterwarnings("ignore")
MiceImputed = over_sampled.copy(deep=True)
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(over_sampled)

# Detecting outliers with IQR
Mice1 = MiceImputed.quantile(0.25)
Mice2 = MiceImputed.quantile(0.75)
Mice3 = Mice2 - Mice1
print(Mice3)

corr = MiceImputed.corr()
mask = np.triu(np.ones_like(corr))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})

sns.pairplot(data=MiceImputed, vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm'), hue='RainTomorrow')
plt.show()

sns.pairplot( data=MiceImputed, vars=('Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow' )
plt.show()

# Standardizing data
MinMax_scaler = preprocessing.MinMaxScaler()
MinMax_scaler.fit(MiceImputed)
New_data = pd.DataFrame(MinMax_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)
print(New_data.head(10))

# Feature Importance using Filter Method (Chi-Square)
X = New_data.loc[:,New_data.columns!='RainTomorrow']
y = New_data[['RainTomorrow']]
K = SelectKBest(chi2, k=10)
K.fit(X, y)
K_new = K.transform(X)
print(X.columns[K.get_support(indices=True)])


X = MiceImputed.drop('RainTomorrow', axis=1)
y = MiceImputed['RainTomorrow']
model = SelectFromModel(rf(n_estimators=100, random_state=0))
model.fit(X, y)
support_model = model.get_support()
model_features = X.loc[:,support_model].columns.tolist()
print(model_features)
print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)



column_features = MiceImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                       'RainToday']]
target = MiceImputed['RainTomorrow']

# Split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(column_features, target, test_size=0.2, random_state=42,
                                                    shuffle=True, stratify=target)

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def roc_cur(false_p, true_per):
    plt.plot(false_p, true_per, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, classification_report


def plot_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t_t = time.time()
    if verbose == False:
        model.fit(X_train, y_train, verbose=0)
    else:
        model.fit(X_train, y_train)
    n_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, n_pred)
    roc_auc = roc_auc_score(y_test, n_pred)
    coh_kap = cohen_kappa_score(y_test, n_pred)
    tt = time.time() - t_t
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(tt))
    print(classification_report(y_test, n_pred, digits=5))

    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    false_p, true_p, thresholds = roc_curve(y_test, probs)
    roc_cur(false_p, true_p)

    # plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='all')

    return model, accuracy, roc_auc, coh_kap, tt


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

params_dt = {'max_depth': 18,
             'max_features': "sqrt"}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, coh_kap_dt, tt_dt = plot_model(model_dt, X_train, y_train, X_test, y_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 18,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 43}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = plot_model(model_rf, X_train, y_train, X_test, y_test)


# XGBoost
import xgboost as xgb
params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = plot_model(model_xgb, X_train, y_train, X_test, y_test)


value = 1.90
width = 1.00


Alg1 = DecisionTreeClassifier(random_state=42)
Alg2 = RandomForestClassifier(random_state=42)
Alg3 = xgb.XGBClassifier(random_state=42)
EVC = EnsembleVoteClassifier(clfs=[Alg1, Alg2, Alg3], weights=[1, 1, 1], voting='soft')

X_feature = MiceImputed[["Sunshine", "Humidity9am", "Cloud3pm"]]
X = np.asarray(X_feature, dtype=np.float32)
y_feature = MiceImputed["RainTomorrow"]
y = np.asarray(y_feature, dtype=np.int32)

# Plotting Decision Regions
gd = gridspec.GridSpec(3,3)
fig = plt.figure(figsize=(18, 14))

labels = ['Decision Tree',
          'Random Forest',
          'XGBoost',
          'Ensemble']

for Alg, lab, grd in zip([Alg1, Alg2, Alg3, EVC],
                         labels,
                         itertools.product([0, 1, 2],
                         repeat=2)):
    Alg.fit(X, y)
    axis = plt.subplot(gd[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=Alg,
                                filler_feature_values={2: value},
                                filler_feature_ranges={2: width},
                                legend=2)
    plt.title(lab)

plt.show()

accuracy_dt = 0.8782151517211685
roc_auc_dt = 0.878215560452389
coh_kap_dt = 0.7564305024566328
TT_dt = 1.7221224308013916
accuracy_rf = 0.9427787975615836
roc_auc_rf = 0.9427794523212041
coh_kap_rf = 0.8855577449354552
TT_rf = 134.64815497398376
accuracy_xgb = 0.9436626101933057
roc_auc_xgb = 0.9436632603313401
coh_kap_xgb = 0.8873253668433951
TT_xgb = 55.581029176712036

accuracy_scores = [accuracy_dt, accuracy_rf, accuracy_xgb]
roc_scores = [roc_auc_dt, roc_auc_rf, roc_auc_xgb]
coh_scores = [coh_kap_dt, coh_kap_rf, coh_kap_xgb]
time_taken = [TT_dt, TT_rf, TT_xgb]

feature_data = {'Model': ['Decision Tree','Random Forest','XGBoost'],
              'Accuracy': accuracy_scores,
              'ROC_AUC': roc_scores,
              'Cohen_Kappa': coh_scores,
              'Time taken': time_taken}
data = pd.DataFrame(feature_data)

figure, ax1 = plt.subplots(figsize=(14,12))
ax1.set_title('Model Comparison: Execution Time and Accuracy', fontsize=13)
color = 'tab:blue'
ax1.set_xlabel('Model', fontsize=13)
ax1.set_ylabel('Time taken', fontsize=13, color=color)
ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette='summer')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Accuracy', fontsize=13, color=color)
ax2 = sns.lineplot(x='Model', y='Accuracy', data = data, color=color)
ax2.tick_params(axis='y', color=color)

figure, ax3 = plt.subplots(figsize=(14,12))
ax3.set_title("Comparing the Area under ROC and Cohen's Kappa models", fontsize=13)
color = 'tab:blue'
ax3.grid()
ax3.set_xlabel('Model', fontsize=13)
ax3.set_ylabel('ROC_AUC', fontsize=13, color=color)
ax4 = sns.barplot(x='Model', y='ROC_AUC', data = data, palette='winter')
ax3.tick_params(axis='y')
ax4 = ax3.twinx()
color = 'tab:red'
ax4.set_ylabel('Cohen_Kappa', fontsize=13, color=color)
ax4 = sns.lineplot(x='Model', y='Cohen_Kappa', data = data, color=color)
ax4.tick_params(axis='y', color=color)
plt.show()
