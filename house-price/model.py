import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv("train.csv")
# print(data_train['SalePrice'].describe())
# sns.distplot(data_train['SalePrice'])

corrmat = data_train.corr()
# f, ax = plt.subplots(figsize=(20, 9))
# sns.heatmap(corrmat, vmax=0.8, square=True)

k  = 10 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()