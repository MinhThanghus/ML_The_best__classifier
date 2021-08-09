import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns

data_frame = pd.read_csv('loan_train.csv')
show_data_frame = data_frame.head()
print(show_data_frame)
# convert data to object
data_frame['due_date'] = pd.to_datetime(data_frame['due_date'])
data_frame['effective_date'] = pd.to_datetime(data_frame['effective_date'])
data_frame_object = data_frame.head(500)
print(data_frame_object)

count_data_frame_object = data_frame_object['loan_status'].value_counts()
print(count_data_frame_object)

bins = np.linspace(data_frame_object.Principal.min(), data_frame_object.Principal.max(), 10)
g = sns.FacetGrid(data_frame_object, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

'''
bins = np.linspace(data_frame_object.age.min(), data_frame_object.age.max(), 10)
g = sns.FacetGrid(data_frame_object, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

'''

# Pre-processing: Feature selection/extraction
data_frame['dayofweek'] = data_frame['effective_date'].dt.dayofweek
bins = np.linspace(data_frame.dayofweek.min(), data_frame.dayofweek.max(), 10)
g = sns.FacetGrid(data_frame, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
# plt.show()
data_frame['weekend'] = data_frame['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
data_frame = data_frame.head()
print(data_frame)

# Convert Categorical features to numerical values
data_frame_numerical = data_frame_object.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
print(data_frame_numerical)

data_frame['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
print(data_frame.head())

# One Hot Encoding
data_frame_one_hot = data_frame_object.groupby(['education'])['loan_status'].value_counts(normalize=True)
print(data_frame_one_hot)
#
data_frame_one_hot=data_frame[['Principal','terms','age','Gender','education']].head()
print(data_frame_one_hot)
Feature = data_frame[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(data_frame['education'])], axis=1)
# Feature=Feature.drop(['Master or Above'], axis = 1,inplace=True)
print(Feature.head())