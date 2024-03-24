#importing libraries
import pandas as pd
import numpy as np
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
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


# Get data from data set
weather = pd.read_csv('C:\\Users\\prana\\PycharmProjects\\Data science project\\weatherAUS.csv')
print(weather.head(20))
# print(weather_data.shape)
print(weather.info())

#Both "RainToday" and "RainTomorrow" are categorical variables indicating whether it will rain or not (Yes/No).
#We'll convert them into binary values (1/0) for ease of analysis.

weather['RainToday'] = weather['RainToday'].map({'No': 0, 'Yes': 1})
weather['RainTomorrow'] = weather['RainTomorrow'].map({'No': 0, 'Yes': 1})

print(weather.head(20))

#Next, we will check whether the dataset is imbalanced or balanced.
#If the dataset is imbalanced, we need to undersample majority or oversample minority to balance it.

figure = plt.figure(figsize = (12,10))
weather.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['#a4c639','#5d8aa8'], alpha = 0.9, rot=0)
plt.title('RainTomorrow: Unbalanced Dataset with No(0) and Yes(1) Indicators')
plt.show()

# Handling Class Imbalance
# Separate data based on RainTomorrow value
no_r = weather[weather['RainTomorrow'] == 0]
yes_r = weather[weather['RainTomorrow'] == 1]

# Oversample the minority class ('yes') to balance the dataset
Over = resample(yes_r, replace=True, n_samples=len(no_r), random_state=123)

# Concatenate the oversampled 'yes' data with the 'no' data
over_sampled = pd.concat([no_r, Over])

#Plotting
plt_figure = plt.figure(figsize = (12,10))
over_sampled.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['#a4c639','#5d8aa8'], alpha = 1.0, rot=0)
plt.title(' Balanced dataset After oversampling, RainTomorrow shows No(0) and Yes(1)')
plt.show()


# Check Missing Data Pattern in Training Data
sns.heatmap(over_sampled.isnull(), cbar=False, cmap='PuBu')

over_sampled.select_dtypes(include=['object']).columns

# Calculate total missing values and percentage of missing values for each column
total_missing = over_sampled.isnull().sum().sort_values(ascending=False)
#Calculate percentage of missing values
percent_missing = (over_sampled.isnull().sum() / len(over_sampled)).sort_values(ascending=False)
#Concatenate total and percentage missing values into a weather_data
summary = pd.concat([total_missing, percent_missing], axis=1, keys=['Total Missing', 'Percent Missing'])
# Display the first 10 rows of the missing data summary
summary.head(10)


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
for column in over_sampled.select_dtypes(include=['object']).columns:
    label_Encoders[column] = LabelEncoder()
    # Fit and transform the column
    over_sampled[column] = label_Encoders[column].fit_transform(over_sampled[column])

warnings.filterwarnings("ignore")
Mice_Sample = over_sampled.copy(deep=True)
M_Imputer = IterativeImputer()
Mice_Sample.iloc[:, :] = M_Imputer.fit_transform(over_sampled)

# Detecting outliers with IQR
Mice1 = Mice_Sample.quantile(0.25)
Mice2 = Mice_Sample.quantile(0.75)
Mice3 = Mice2 - Mice1
print(Mice3)

# Mice_Sample = Mice_Sample[~((Mice_Sample < (Q1 - 1.5 * IQR)) |(Mice_Sample > (Q3 + 1.5 * IQR))).any(axis=1)]
# Mice_Sample.shape


cr = Mice_Sample.corr()
mk = np.triu(np.ones_like(cr))
z, x = plt.subplots(figsize=(20, 20))
map = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(cr, mask=mk, cmap=map, vmax=None, center=0, square=True, annot=True, linewidths=.6, cbar_kws={"shrink": 0.9})
plt.show()

sns.pairplot(data=Mice_Sample, vars=('MaxTemp', 'MinTemp', 'Pressure9am', 'Pressure3pm'), hue='RainTomorrow')
plt.show()

sns.pairplot(data=Mice_Sample, vars=('Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow')
plt.show()

# Standardizing data
MinMax = preprocessing.MinMaxScaler()
MinMax.fit(Mice_Sample)
New = pd.DataFrame(MinMax.transform(Mice_Sample), index=Mice_Sample.index, columns=Mice_Sample.columns)
print(New.head(10))

# Feature Importance using Filter Method (Chi-Square)
fitx = New.loc[:, New.columns != 'RainTomorrow']
fity = New[['RainTomorrow']]
K = SelectKBest(chi2, k=10)
K.fit(fitx, fity)
K2 = K.transform(fitx)
print(fitx.columns[K.get_support(indices=True)])


fx = Mice_Sample.drop('RainTomorrow', axis=1)
fy = Mice_Sample['RainTomorrow']
model = SelectFromModel(rf(n_estimators=100, random_state=0))
model.fit(fx, fy)
Demo_model = model.get_support()
fm = fx.loc[:, Demo_model].columns.tolist()
print(fm)
print(rf(n_estimators=100, random_state=0).fit(fx, fy).feature_importances_)



column_f = Mice_Sample[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                       'WindGustSpeed', 'WindDir9am', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                       'RainToday', 'Location', 'WindDir3pm', 'Cloud9am']]

ft = Mice_Sample['RainTomorrow']

# Split into test and train
f_train, f_test, g_train, g_test = train_test_split(column_f, ft, test_size=0.2, random_state=42,
                                                    shuffle=True, stratify=ft)

# Normalize Features
scaler = StandardScaler()
f_train = scaler.fit_transform(f_train)
f_test = scaler.fit_transform(f_test)


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


def plot_model(xmodel, n_train, s_train, n_test, s_test, verbose=True):
    t_t = time.time()
    if verbose == False:
        xmodel.fit(n_train, s_train, verbose=0)
    else:
        xmodel.fit(n_train, s_train)
    n_pred = xmodel.predict(n_test)
    accuracy = accuracy_score(s_test, n_pred)
    roc_auc = roc_auc_score(s_test, n_pred)
    coh_kap = cohen_kappa_score(s_test, n_pred)
    tt = time.time() - t_t
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(tt))
    print(classification_report(s_test, n_pred, digits=5))

    probs = xmodel.predict_proba(n_test)
    probs = probs[:, 1]
    false_p, true_p, thresholds = roc_curve(s_test, probs)
    roc_cur(false_p, true_p)

    return xmodel, accuracy, roc_auc, coh_kap, tt


# Decision Tree

params_dec = {'max_depth': 18,
             'max_features': "sqrt"}

dec_model = DecisionTreeClassifier(**params_dec)
dec_model, accuracy_decision_tree, roc_auc_decision_tree, coh_kap_decision_tree, tt_dt = plot_model(dec_model, f_train, g_train, f_test, g_test)


# Random Forest

params_random_forest = {'max_depth': 18,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 43}

Random_model = RandomForestClassifier(**params_random_forest)
Random_model, accuracy_random_forest, roc_auc_random_forest, coh_kap_random_forest, tt_rf = plot_model(Random_model, f_train, g_train, f_test, g_test)


# XGBoost
params_xgboost ={'n_estimators': 500,
            'max_depth': 16}

xgboost_model = xgb.XGBClassifier(**params_xgboost)
xgboost_model, accuracy_xgboost, roc_auc_xgboost, coh_kap_xgboost, tt_xgb = plot_model(xgboost_model, f_train, g_train, f_test, g_test)


value = 1.90
width = 1.00


Alg1 = DecisionTreeClassifier(random_state=42)
Alg2 = RandomForestClassifier(random_state=42)
Alg3 = xgb.XGBClassifier(random_state=42)
EVC = EnsembleVoteClassifier(clfs=[Alg1, Alg2, Alg3], weights=[1, 1, 1], voting='soft')

n_feature = Mice_Sample[["Sunshine", "Humidity9am", "Cloud3pm"]]
fx = np.asarray(n_feature, dtype=np.float32)
s_feature = Mice_Sample["RainTomorrow"]
fy = np.asarray(s_feature, dtype=np.int32)

# Plotting Decision Regions
gd = gridspec.GridSpec(4,4)
fig = plt.figure(figsize=(18, 14))

Model_labels = ['Decision Tree',
          'Random Forest',
          'XGBoost',
          'Ensemble']

for Alg, lb, zx in zip([Alg1, Alg2, Alg3, EVC],
                       Model_labels,
                       itertools.product([0, 1, 2],
                         repeat=2)):
    Alg.fit(fx, fy)
    axis = plt.subplot(gd[zx[0], zx[1]])
    fig = plot_decision_regions(X=fx, y=fy, clf=Alg,
                                filler_feature_values={2: value},
                                filler_feature_ranges={2: width},
                                legend=2)
    plt.title(lb)

plt.show()

accuracy_decision_tree = 0.8782151517211685
roc_auc_decision_tree = 0.878215560452389
coh_kap_decision_tree = 0.7564305024566328
time_taken_decision_tree = 1.7221224308013916
accuracy_random_forest = 0.9427787975615836
roc_auc_random_forest = 0.9427794523212041
coh_kap_random_forest = 0.8855577449354552
time_taken_random_forest = 134.64815497398376
accuracy_xgboost = 0.9436626101933057
roc_auc_xgboost = 0.9436632603313401
coh_kap_xgboost = 0.8873253668433951
time_taken_xgboost = 55.581029176712036

accuracy_scores = [accuracy_decision_tree, accuracy_random_forest, accuracy_xgboost]
roc_scores = [roc_auc_decision_tree, roc_auc_random_forest, roc_auc_xgboost]
coh_scores = [coh_kap_decision_tree, coh_kap_random_forest, coh_kap_xgboost]
time_taken = [time_taken_decision_tree, time_taken_random_forest, time_taken_xgboost]

feature_model = {'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
              'Accuracy': accuracy_scores,
              'ROC_AUC': roc_scores,
              'Cohen_Kappa': coh_scores,
              'Time taken': time_taken}
s_data = pd.DataFrame(feature_model)

figure, axis_x = plt.subplots(figsize=(14, 12))
axis_x.set_title('Model Comparison: Execution Time and Accuracy', fontsize=13)
color = 'tab:blue'
axis_x.set_xlabel('Model', fontsize=13)
axis_x.set_ylabel('Time taken', fontsize=13, color=color)
axis_y = sns.barplot(x='Model', y='Time taken', data = s_data, palette='summer')
axis_x.tick_params(axis='y')
axis_y = axis_x.twinx()
color = 'tab:red'
axis_y.set_ylabel('Accuracy', fontsize=13, color=color)
axis_y = sns.lineplot(x='Model', y='Accuracy', data = s_data, color=color)
axis_y.tick_params(axis='y', color=color)

figure, axis_n = plt.subplots(figsize=(14, 12))
axis_n.set_title("Comparing the Area under ROC and Cohen's Kappa models", fontsize=13)
color = 'tab:blue'
axis_n.grid()
axis_n.set_xlabel('Model', fontsize=13)
axis_n.set_ylabel('ROC_AUC', fontsize=13, color=color)
axis_m = sns.barplot(x='Model', y='ROC_AUC', data = s_data, palette='winter')
axis_n.tick_params(axis='y')
axis_m = axis_n.twinx()
color = 'tab:red'
axis_m.set_ylabel('Cohen_Kappa', fontsize=13, color=color)
axis_m = sns.lineplot(x='Model', y='Cohen_Kappa', data = s_data, color=color)
axis_m.tick_params(axis='y', color=color)
plt.show()
