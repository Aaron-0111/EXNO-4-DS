# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/5c42c887-ac30-4637-aa06-2244688ae72b)
```
df.dropna()
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/b9dbeff5-4254-4845-9ce0-e71f02baf2d5)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/00129ae5-d953-4420-88ab-c5f5973a7d82)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/2a0f68d0-5c9e-4387-9132-4f51fd4dd6c7)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/3b87f235-481b-4627-b9b4-db86d7dd2d96)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/8c5600b6-0e80-41a6-bd85-c7e50e084000)
```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/6cd59567-61f1-4f4d-86e1-7139fb453d03)
```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/e91d6ad4-31a7-4ae0-afcb-a1dbb3d96032)
```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/113f0495-c594-489c-b9af-07f18e0b1cb2)
```
data.isnull().sum()
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/8e9036e0-ba47-42f1-9267-92b636d48b17)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/db99dde8-fc2e-4b14-8f52-4f92f307528b)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/19a17b83-7b3f-4576-b074-788ab9b4964d)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/294c908d-2246-4e1b-a55e-349c68735098)
```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/610a7672-265c-4d9b-b258-4390136c555a)
```
data2
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/542727f1-c9c7-4092-a63e-700634bc4bdf)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/ba04ac9f-e06f-41c2-9a40-d42f3eedca07)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/77d36b9f-28f6-4e18-9c25-32e745e7c922)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/5bb38e8c-8930-4c51-99d4-a3e54fd3bade)
```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/0d70c5dc-39f8-4a63-ba0e-af2e9e91c56a)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/a67e439b-c4c7-40d0-834b-0a036344110b)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/4236fbe3-1602-4140-926d-cf79140c6d5a)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
0.8392087523483258

```

print('Misclassified samples: %d' % (test_y != prediction).sum())
```
Misclassified samples: 1455

```
data.shape

(31978, 13)
```
## FEATURE SELECTION TECHNIQUES
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/a8a7dcb3-e5b9-4b1e-92af-276154b4c58c)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/bc83e709-dc31-462d-b2c9-194dae80577f)
```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/b7e421f3-3d82-478f-87e7-a07e95c55292)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/Aaron-0111/EXNO-4-DS/assets/149347631/8bf177cc-925a-436a-bae3-6f07c9643b4c)

# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.

