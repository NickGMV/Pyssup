import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns

### Packages I use all the time




### function for data cleaning


def cleaning_checks(datasource,checks = 'all'):
    report = []
    if datasource.isna().sum().sum():
        nulls_report = datasource.isna().sum()
        report.append('Report on nulls')
        report.append(nulls_report)
    else:
        report.append('No nulls')
        
    if datasource.duplicated().sum():
        dup_report = datasource.duplicated().sum()
        dup_loc = datasource[datasource.duplicated()].index
        report.append('Report on duplicates')
        report.append(f"Duplicate count is {dup_report}")
        report.append(f"Duplicate row numbers are {dup_loc.values}")
    else: 
        report.append('No nulls')
    for item in report:
        print(item,'\n')

    return 'successful report run'


### quickly build a multivariate regression

def regression_builder(data):
    data = pd.get_dummies(data,drop_first = True)
    print('any categorialc data will be ecoded with 1 hot encoding')
    display(data.head())
    pred = 0
    while pred not in data.columns:
        pred = input('This is your data which column is going to be predicted?: ')
    y = data[pred]
    pp = [x for x in data.columns if x!= pred]
    predictors = data[pp]
    z = plt.figure()
    sns.heatmap(predictors.corr(), annot = True, vmin= -1, vmax = 1)
    plt.close()
    #z = sns.heatmap(predictors.corr(), annot = True, vmin= -1, vmax = 1);
    display(z)
    time.sleep(3)
    
    included = []
    while len(included)==0:
        for column in predictors.columns:
            con = input(f"should {column} be included in your regression? [y/n] :")
            if con == 'y':
                included.append(column)
        
    
    X = data[included]
    lr = LinearRegression()
    lr.fit(X,y)
    print(f'linear regression was trained with an accuracy of {lr.score(X,y)}')
    return lr , X, y
    
    