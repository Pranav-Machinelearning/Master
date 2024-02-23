#importing libraries
import pandas as pd
# import numpy as np
# import sklearn as sk
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pl

weather_data = pd.read_csv('C:\\Users\\prana\\PycharmProjects\\Data science project\\weatherAUS.csv')
# print(weather_data.head(20))
# print(weather_data.shape)
# print(weather_data.info())

#Both "RainToday" and "RainTomorrow" are categorical variables indicating whether it will rain or not (Yes/No).
#We'll convert them into binary values (1/0) for ease of analysis.

weather_data['RainToday'] = weather_data['RainToday'].map({'No': 0, 'Yes': 1})
weather_data['RainTomorrow'] = weather_data['RainToday'].map({'No': 0, 'Yes': 1})

# print(weather_data.head(20))

#Next, we will check whether the dataset is imbalanced or balanced.
#If the dataset is imbalanced, we need to undersample majority or oversample minority to balance it.

