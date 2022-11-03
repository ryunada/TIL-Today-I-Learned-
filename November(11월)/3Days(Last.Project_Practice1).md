
# Home Credit Default Risk

## 라이브러리 세팅


```python
import pandas as pd
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline
```

## 경로 설정


```python
print(os.getcwd())
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project/home-credit-default-risk-Data')
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project/home-credit-default-risk-Data


## 데이터 모델 설명

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-02%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.42.33.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-02%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.42.33.png)

# 데이터 준비


```python
app_train = pd.read_csv('./application_train.csv')
app_test = pd.read_csv('./application_test.csv')
bureau = pd.read_csv('./bureau.csv')
previous_application = pd.read_csv('./previous_application.csv')
bureadu_balance = pd.read_csv('./bureau_balance.csv')
```


```python
app_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>




```python
# app_test는 target이 없으니 하나차이 난다.
app_train.shape, app_test.shape
```




    ((307511, 122), (48744, 121))



## Target값 분포 및 AMT_INCOME_TOTA 값 Histogram
- Target값 별 분포도ㅡ Pandas, Matplotlib, Seaborn으로 histogram표현


```python
app_train['TARGET'].value_counts()
```




    0    282686
    1     24825
    Name: TARGET, dtype: int64




```python
app_train['AMT_INCOME_TOTAL'].hist()
#plt.hist(app_train['AMT_INCOME_TOTAL'])
```




    <AxesSubplot:>


![output_13_1](https://user-images.githubusercontent.com/87309905/199641560-36ba91c3-b896-4229-92c9-da784211fad3.png)





```python
plt.hist(app_train['AMT_INCOME_TOTAL'])
```




    (array([3.07508e+05, 2.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00]),
     array([2.56500000e+04, 1.17230850e+07, 2.34205200e+07, 3.51179550e+07,
            4.68153900e+07, 5.85128250e+07, 7.02102600e+07, 8.19076950e+07,
            9.36051300e+07, 1.05302565e+08, 1.17000000e+08]),
     <BarContainer object of 10 artists>)


![output_14_1](https://user-images.githubusercontent.com/87309905/199641572-dbb34b1b-1c2a-4077-ab72-c030f5f8d7f5.png)
   



```python
sns.distplot(app_train['AMT_INCOME_TOTAL'])
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/526442897.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train['AMT_INCOME_TOTAL'])





    <AxesSubplot:xlabel='AMT_INCOME_TOTAL', ylabel='Density'>




    
![output_15_2](https://user-images.githubusercontent.com/87309905/199641581-ac2b26c6-4d1b-46ee-8d9b-ae77d9c06d01.png)



```python
sns.boxplot(app_train['AMT_INCOME_TOTAL'])
# -> 이상치 존재
```




    <AxesSubplot:>




    
![output_16_1](https://user-images.githubusercontent.com/87309905/199641590-896d5cdf-7227-4954-add7-c1af34959e14.png)
    


## AMT_INCOME_TOTAL이 1000000 이하인 값에 대한 분포도
- boolean indexing으로 filtering후 histogram표현


```python
# boolean indexing 으로 filtering 적용
app_train[app_train['AMT_INCOME_TOTAL'] < 1000000]['AMT_INCOME_TOTAL'].hist()
```




    <AxesSubplot:>




![output_18_1](https://user-images.githubusercontent.com/87309905/199641596-0f309814-15bc-4bec-811b-a23ebe3f7337.png)




```python
# distplot으로 histogram 표현
sns.distplot(app_train[app_train['AMT_INCOME_TOTAL']<1000000]['AMT_INCOME_TOTAL'])
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/2498703021.py:2: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train[app_train['AMT_INCOME_TOTAL']<1000000]['AMT_INCOME_TOTAL'])





    <AxesSubplot:xlabel='AMT_INCOME_TOTAL', ylabel='Density'>



    
![output_19_2](https://user-images.githubusercontent.com/87309905/199641605-8acfc71f-bbd3-45c1-8aab-ac995f2695df.png)


## TARGET 값에 따른 AMT_INCOME_TOTAL 값 분포도 비교
- distplot과 violinplot 시각화
- plt.subplots() 기반으로 seaborn의 distplot과 violinplot으로 분포도 비교 시각화


```python
# TARGET값에 따른 Filtering 조건 각각 설정
cond1 = (app_train['TARGET'] == 1)
cond0 = (app_train['TARGET'] == 0)
# AMT_INCOME_TOTAL은 매우 큰 값이 있으므로 이는 제외
cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
# distplot으로 TARGET = 1이면 빨간색으로, 0이면 푸른색으로 Histogram 표현
sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], label = '0', color = 'blue')
sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], label = '1', color = 'red')
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/2326682751.py:7: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], label = '0', color = 'blue')
    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/2326682751.py:8: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], label = '1', color = 'red')





    <AxesSubplot:xlabel='AMT_INCOME_TOTAL', ylabel='Density'>



![output_21_2](https://user-images.githubusercontent.com/87309905/199641612-6c89dadc-0a20-4a87-8bc7-fc491fde7321.png)
   



```python
# violineplot을 이용하면 Category 값별로 연속형 값의 분포도를 알 수 있음. X는 category컬럼, y는 연속형 컬럼
sns.violinplot(x = 'TARGET', y = 'AMT_INCOME_TOTAL', data = app_train[cond_amt])
```




    <AxesSubplot:xlabel='TARGET', ylabel='AMT_INCOME_TOTAL'>





![output_22_1](https://user-images.githubusercontent.com/87309905/199641621-7ce91489-8148-4d2f-a38b-1ded1fedc546.png)
    



```python
# 2개의 subplot을 생성
fig, axis = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2)
```



![output_23_0](https://user-images.githubusercontent.com/87309905/199641626-ae5845a3-6535-4ae1-8a23-851fe9e754fb.png)
    



```python
# TARGET 값 유형에 따른 Boolean Indexing 조건
cond1 = (app_train['TARGET'] == 1)
cond0 = (app_train['TARGET'] == 0)
cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
# 2개의 subplot을 생성하고 왼쪽에는 violinplot을 오른쪽에는 distplot을 표현
fig, axs = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2, squeeze = False)
# violin plot을 왼쪽 subplot에 그림.
sns.violinplot(x = 'TARGET', y = 'AMT_INCOME_TOTAL', data = app_train[cond_amt], ax = axs[0][0])
# Histogram을 오른쪽 subplot에 그림.
sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], ax = axs[0][1], label = '0', color = 'blue')
sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], ax = axs[0][1], label = '1', color = 'red')
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/1622922756.py:10: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], ax = axs[0][1], label = '0', color = 'blue')
    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/1622922756.py:11: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], ax = axs[0][1], label = '1', color = 'red')





    <AxesSubplot:xlabel='AMT_INCOME_TOTAL', ylabel='Density'>





![output_24_2](https://user-images.githubusercontent.com/87309905/199641636-0d0630e3-38e0-42fa-b7ac-08fc32d9f834.png)
    



```python
def show_column_hist_by_target(df, column, is_amt=False): 
    cond1 = (df['TARGET'] == 1)
    cond0 = (df['TARGET'] == 0)
    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=2, squeeze=False) 
    # is_amt가 True이면 < 500000 조건으로 filtering
    cond_amt = True
    if is_amt:
        cond_amt = df[column] < 500000

    sns.violinplot(x='TARGET', y=column, data=df[cond_amt], ax=axs[0][0] ) 
    sns.distplot(df[cond0 & cond_amt][column], ax=axs[0][1], label='0', color='blue') 
    sns.distplot(df[cond1 & cond_amt][column], ax=axs[0][1], label='1', color='red')
show_column_hist_by_target(app_train, 'AMT_CREDIT', is_amt=True)
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/961505735.py:11: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df[cond0 & cond_amt][column], ax=axs[0][1], label='0', color='blue')
    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5673/961505735.py:12: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df[cond1 & cond_amt][column], ax=axs[0][1], label='1', color='red')



    
![output_25_1](https://user-images.githubusercontent.com/87309905/199641645-9a669067-78a6-424c-ad99-8c6b90779cf6.png)


### app_train과 app_test를 합쳐서 한번에 데이터 preprocessing 수행


```python
app_train.shape, app_test.shape
```




    ((307511, 122), (48744, 121))



->  app_test에는 타깃 값이 없어 app_train과 속성이 1개 차이가 난다.


```python
# pandas의 concat()을 이용하여 app_train과 app_test를 결합
apps = pd.concat([app_train, app_test])
apps.shape
```




    (356255, 122)




```python
# app_train의 TARGET 값을 NULL로 입력됨.
apps['TARGET'].value_counts(dropna = False)
```




    0.0    282686
    NaN     48744
    1.0     24825
    Name: TARGET, dtype: int64



### Object Feature들을 Label Encoding
- pandas의 factorize()를 이용


```python
apps.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 356255 entries, 0 to 48743
    Columns: 122 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
    dtypes: float64(66), int64(40), object(16)
    memory usage: 334.3+ MB



```python
# pd.factorize()는 편리하게 Category 컬럼을 Label 인코딩 수행
# pd.factorize(Category컬럼 Series)는 Label 인코딩된 Series와 uniq한 Category값을 반환함.
# [0]을 이용하여 Label인코딩 Series
apps['CODE_GENDER'] = pd.factorize(apps['CODE_GENDER'])[0]
```


```python
pd.factorize(apps['CODE_GENDER']) # 두개를 반환한다.. 그래서 첫번째만 반환해라 해서, [0]을 넣는다.
```




    (array([0, 1, 0, ..., 1, 0, 1]), Int64Index([0, 1, 2], dtype='int64'))




```python
apps['CODE_GENDER'].value_counts()
```




    1    235126
    0    121125
    2         4
    Name: CODE_GENDER, dtype: int64




```python
apps.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 356255 entries, 0 to 48743
    Columns: 122 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
    dtypes: float64(66), int64(41), object(15)
    memory usage: 334.3+ MB



```python
# Label 인코딩을 위해 object 유형의 컬럼만 추출
object_columns = apps.dtypes[apps.dtypes == 'object'].index.tolist()
```


```python
object_columns
```




    ['NAME_CONTRACT_TYPE',
     'FLAG_OWN_CAR',
     'FLAG_OWN_REALTY',
     'NAME_TYPE_SUITE',
     'NAME_INCOME_TYPE',
     'NAME_EDUCATION_TYPE',
     'NAME_FAMILY_STATUS',
     'NAME_HOUSING_TYPE',
     'OCCUPATION_TYPE',
     'WEEKDAY_APPR_PROCESS_START',
     'ORGANIZATION_TYPE',
     'FONDKAPREMONT_MODE',
     'HOUSETYPE_MODE',
     'WALLSMATERIAL_MODE',
     'EMERGENCYSTATE_MODE']




```python
# pd.factorize()는 한개의 컬럼만 Label인코딩이 가능하므로 object형 컬럼들을 iteration하면서 변환 수행.
for column in object_columns:
    apps[column] = pd.factorize(apps[column])[0]
```


```python
apps.info
```




    <bound method DataFrame.info of        SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  \
    0          100002     1.0                   0            0             0   
    1          100003     0.0                   0            1             0   
    2          100004     0.0                   1            0             1   
    3          100006     0.0                   0            1             0   
    4          100007     0.0                   0            0             0   
    ...           ...     ...                 ...          ...           ...   
    48739      456221     NaN                   0            1             0   
    48740      456222     NaN                   0            1             0   
    48741      456223     NaN                   0            1             1   
    48742      456224     NaN                   0            0             0   
    48743      456250     NaN                   0            1             1   
    
           FLAG_OWN_REALTY  CNT_CHILDREN  AMT_INCOME_TOTAL  AMT_CREDIT  \
    0                    0             0          202500.0    406597.5   
    1                    1             0          270000.0   1293502.5   
    2                    0             0           67500.0    135000.0   
    3                    0             0          135000.0    312682.5   
    4                    0             0          121500.0    513000.0   
    ...                ...           ...               ...         ...   
    48739                0             0          121500.0    412560.0   
    48740                1             2          157500.0    622413.0   
    48741                0             1          202500.0    315000.0   
    48742                1             0          225000.0    450000.0   
    48743                1             0          135000.0    312768.0   
    
           AMT_ANNUITY  ...  FLAG_DOCUMENT_18  FLAG_DOCUMENT_19  FLAG_DOCUMENT_20  \
    0          24700.5  ...                 0                 0                 0   
    1          35698.5  ...                 0                 0                 0   
    2           6750.0  ...                 0                 0                 0   
    3          29686.5  ...                 0                 0                 0   
    4          21865.5  ...                 0                 0                 0   
    ...            ...  ...               ...               ...               ...   
    48739      17473.5  ...                 0                 0                 0   
    48740      31909.5  ...                 0                 0                 0   
    48741      33205.5  ...                 0                 0                 0   
    48742      25128.0  ...                 0                 0                 0   
    48743      24709.5  ...                 0                 0                 0   
    
           FLAG_DOCUMENT_21  AMT_REQ_CREDIT_BUREAU_HOUR  \
    0                     0                         0.0   
    1                     0                         0.0   
    2                     0                         0.0   
    3                     0                         NaN   
    4                     0                         0.0   
    ...                 ...                         ...   
    48739                 0                         0.0   
    48740                 0                         NaN   
    48741                 0                         0.0   
    48742                 0                         0.0   
    48743                 0                         0.0   
    
           AMT_REQ_CREDIT_BUREAU_DAY  AMT_REQ_CREDIT_BUREAU_WEEK  \
    0                            0.0                         0.0   
    1                            0.0                         0.0   
    2                            0.0                         0.0   
    3                            NaN                         NaN   
    4                            0.0                         0.0   
    ...                          ...                         ...   
    48739                        0.0                         0.0   
    48740                        NaN                         NaN   
    48741                        0.0                         0.0   
    48742                        0.0                         0.0   
    48743                        0.0                         0.0   
    
           AMT_REQ_CREDIT_BUREAU_MON  AMT_REQ_CREDIT_BUREAU_QRT  \
    0                            0.0                        0.0   
    1                            0.0                        0.0   
    2                            0.0                        0.0   
    3                            NaN                        NaN   
    4                            0.0                        0.0   
    ...                          ...                        ...   
    48739                        0.0                        0.0   
    48740                        NaN                        NaN   
    48741                        0.0                        3.0   
    48742                        0.0                        0.0   
    48743                        0.0                        1.0   
    
           AMT_REQ_CREDIT_BUREAU_YEAR  
    0                             1.0  
    1                             0.0  
    2                             0.0  
    3                             NaN  
    4                             0.0  
    ...                           ...  
    48739                         1.0  
    48740                         NaN  
    48741                         1.0  
    48742                         2.0  
    48743                         4.0  
    
    [356255 rows x 122 columns]>



### Null값 일괄 변환


```python
apps.isnull().sum().head(100)
```




    SK_ID_CURR                    0
    TARGET                    48744
    NAME_CONTRACT_TYPE            0
    CODE_GENDER                   0
    FLAG_OWN_CAR                  0
                              ...  
    DAYS_LAST_PHONE_CHANGE        1
    FLAG_DOCUMENT_2               0
    FLAG_DOCUMENT_3               0
    FLAG_DOCUMENT_4               0
    FLAG_DOCUMENT_5               0
    Length: 100, dtype: int64




```python
# -999로 모든 컬럼들을 Null값 변환
apps = apps.fillna(-999)
```


```python
apps.isnull().sum().head(100)
```




    SK_ID_CURR                0
    TARGET                    0
    NAME_CONTRACT_TYPE        0
    CODE_GENDER               0
    FLAG_OWN_CAR              0
                             ..
    DAYS_LAST_PHONE_CHANGE    0
    FLAG_DOCUMENT_2           0
    FLAG_DOCUMENT_3           0
    FLAG_DOCUMENT_4           0
    FLAG_DOCUMENT_5           0
    Length: 100, dtype: int64



### 학습 데이터와 테스트 데이터 다시 분리


```python
# app_test의 TARGET컬럼은 원래 null이었는데 앞에서 fillna(-999)로 -999로 변환됨. 이를 추출함
app_train = apps[apps['TARGET'] != -999]
app_test = apps[apps['TARGET'] == -999]
app_train.shape, app_test.shape
```




    ((307511, 122), (48744, 122))




```python
# app_test의 TARGET컬럼을 DROP
app_test = app_test.drop('TARGET', axis = 1)
```


```python
app_test.shape
```




    (48744, 121)




```python
app_test.dtypes
```




    SK_ID_CURR                      int64
    NAME_CONTRACT_TYPE              int64
    CODE_GENDER                     int64
    FLAG_OWN_CAR                    int64
    FLAG_OWN_REALTY                 int64
                                   ...   
    AMT_REQ_CREDIT_BUREAU_DAY     float64
    AMT_REQ_CREDIT_BUREAU_WEEK    float64
    AMT_REQ_CREDIT_BUREAU_MON     float64
    AMT_REQ_CREDIT_BUREAU_QRT     float64
    AMT_REQ_CREDIT_BUREAU_YEAR    float64
    Length: 121, dtype: object



### 학습 데이터를 검증 데이터로 분리하고 LFBM Classifier로 학습 수행.
- 피처용 데이터와 타겟 데이터 분리
- 학습용/검증요 데이터 세트 분리


```python
ftr_app = app_train.drop(['SK_ID_CURR', 'TARGET'], axis = 1) # prime key값도 제외 시키자
target_app = app_train['TARGET']
```


```python
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size = 0.3, random_state = 2020)
train_x.shape, valid_x.shape
```




    ((215257, 120), (92254, 120))




```python
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    n_jobs = -1,
    n_estimators = 1000,
    learning_rate = 0.02,
    num_leaves = 32,
    subsample = 0.8,
    max_depth = 12,
    silent = -1,
    verbose = -1
    )

clf.fit(train_x, train_y, eval_set = [(train_x, train_y), (valid_x, valid_y)],
       eval_metric = 'auc', verbose = 100, early_stopping_rounds = 50)
```

    Training until validation scores don't improve for 50 rounds
    [100]	training's auc: 0.752205	training's binary_logloss: 0.250372	valid_1's auc: 0.744317	valid_1's binary_logloss: 0.251593
    [200]	training's auc: 0.771473	training's binary_logloss: 0.243554	valid_1's auc: 0.754053	valid_1's binary_logloss: 0.247539
    [300]	training's auc: 0.784885	training's binary_logloss: 0.239292	valid_1's auc: 0.757737	valid_1's binary_logloss: 0.246203
    [400]	training's auc: 0.796336	training's binary_logloss: 0.235948	valid_1's auc: 0.758946	valid_1's binary_logloss: 0.245732
    [500]	training's auc: 0.806016	training's binary_logloss: 0.233017	valid_1's auc: 0.759411	valid_1's binary_logloss: 0.24555
    Early stopping, best iteration is:
    [532]	training's auc: 0.808934	training's binary_logloss: 0.232125	valid_1's auc: 0.759548	valid_1's binary_logloss: 0.245494





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMClassifier(learning_rate=0.02, max_depth=12, n_estimators=1000,
               num_leaves=32, silent=-1, subsample=0.8, verbose=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(learning_rate=0.02, max_depth=12, n_estimators=1000,
               num_leaves=32, silent=-1, subsample=0.8, verbose=-1)</pre></div></div></div></div></div>



### Feature importance 시각화


```python
from lightgbm import plot_importance 
plot_importance(clf, figsize=(16, 32))
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='Feature importance', ylabel='Features'>





![output_55_1](https://user-images.githubusercontent.com/87309905/199641667-eae10d86-c4ab-420f-b27d-a039fac963e9.png)
    


### 학습된 Classifier를 이용하여 테스트 데이터를 예측하고 결과를 Kaggle로 Submit수행.


```python
# 학습된 classifier의 predict_proba()를 이용하여 binary classification에서 1이될 확률만 추출
preds = clf.predict_proba(app_test.drop(['SK_ID_CURR'], axis = 1))[:, 1]
```


```python
clf.predict_proba(app_test.drop(['SK_ID_CURR'],axis = 1)) # 0과 1이 될 확률이 두개다 표시가 된다.
```




    array([[0.97246502, 0.02753498],
           [0.87904013, 0.12095987],
           [0.98381082, 0.01618918],
           ...,
           [0.96802646, 0.03197354],
           [0.94283805, 0.05716195],
           [0.82020099, 0.17979901]])




```python
# app_test의 TARGET으로 1이될 확률 Update
app_test['TARGET'] = preds
app_test['TARGET'].head(10)
```




    0    0.027535
    1    0.120960
    2    0.016189
    3    0.037421
    4    0.146077
    5    0.036411
    6    0.017445
    7    0.041752
    8    0.016933
    9    0.087198
    Name: TARGET, dtype: float64




```python
# SK_ID_CURR과 TARGET값만 csv 형태로 생성.
# app_test[['SK_ID_CURR', 'TARGET']].to_csv('app_baseline_01.csv',index=False)
```
