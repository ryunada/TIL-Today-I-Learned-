## 용어

딥러닝의 차원은 피처 갯수  
특이값 분해 (SVD ; Singular Value Decomposition)  
다중공선성  
오토인코딩  
주성분 분석(PCA ; Principal Component Analysis)  
공분산 행렬(Covariance Matrix)  
정방형 행렬(Square Matrix)  
대칭 행렬(Symmetric Matrix)  
Linear transformation  
nonlinear transformation  
A^T => T : Transferate  
< 이미지 분석 >  
frangi filter  

hit the bull’s eyes  


데이터 셋 이해가 안될때  
PCA -> Clustering -> 뭘 할지 결정  
=> 자동으로 분할하였다.  
LDA  


의사코드(Pseudo-code)  



# 차원 축소
피처 선택(Feature Selection) : 불필요한 피처는 아예 제거, 주요 피처만 선택  
피처 추출(Feature Extraction) : 저차원의 중요피처로 압축해서 추출하는것, 새로운 피처로 추출하는 것 


![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-10-26%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.46.25.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-10-26%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.46.25.png)

# PCA(주성분 분석)
< 수행 절차 >
I. 입력 데이터 세트의 공분산 행렬을 생성합니다.  
II. 공분산 행렬의 고유벡터와 고유값을 계산합니다.  
III. 고유값이 가장 큰 순으로 K개(PCA 변환 차수만큼)만큼 고유벡터를 추출합니다.  
IV. 고유값이 가장 큰 수능로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환합니다.  

## 실습 : 붓꽃 데이터 PCA


```python
# 데이터 읽어오기
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 사이킷런 내장 데이터 셋 API 호출
iris = load_iris()

# 넘파이 데이터 셋을 Pandas DataFrame으로 변환
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
irisDF = pd.DataFrame(iris.data, columns = columns)
irisDF['target']  = iris.target

print(irisDF.shape)
irisDF.head(3)
```

    /opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"


    (150, 5)





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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 데이터시각화 확인
#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 각 target 별로 다른 shape으로 scatter plot 
for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
    y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
```


![output_6_0](https://user-images.githubusercontent.com/87309905/197978934-67d97d8f-043e-470e-833a-f1b66eca69e0.png)
    



```python
from sklearn.preprocessing import StandardScaler

iris_scaled = StandardScaler().fit_transform(irisDF)
```


```python
# PCA tngod(n_components = 2)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
# fit()과 transform()을 호출하여 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

# 차원이 2차원으로 변환된 것 확인
print(iris_pca.shape)
```

    (150, 2)



```python
# PCA 변환된 데이터의 컬럼명을 각각 pca_component_1, pca_component_2로 명명
pca_columns = ['pca_component_1','pca_component_2']

irisDF_pca = pd.DataFrame(iris_pca, columns = pca_columns)
irisDF_pca['target'] = iris.target
irisDF_pca.head(3)`
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
      <th>pca_component_1</th>
      <th>pca_component_2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.576120</td>
      <td>0.474499</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.415322</td>
      <td>-0.678092</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.659333</td>
      <td>-0.348282</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# PCA 차원 축소된 피처들로 데이터 산포도 시각화
#setosa를 세모, versicolor를 네모, virginica를 동그라미로 표시
markers=['^', 's', 'o']

#pca_component_1 을 x축, pc_component_2를 y축으로 scatter plot 수행. 
for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1']
    y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()
```


![output_10_0](https://user-images.githubusercontent.com/87309905/197978908-dcd790d3-73ca-4b67-bbab-0ee6151fad66.png)



```python
# 각 PCA Component 별 변동성 비율
print(pca.explained_variance_ratio_)
```

    [0.76740358 0.18282727]



```python
# 원본 데이터와 PCA 변화된 데이터 간 랜덤포레스트 분류기 예측 성능 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier(random_state = 156)

# 원본 데이터
scores = cross_val_score(rcf, iris.data, iris.target,scoring = 'accuracy', cv = 3)
print(scores)
print(np.mean(scores))
```

    [0.98 0.94 0.96]
    0.96



```python
# PCA로 차원축소한 데이터
pca_X = irisDF_pca[['pca_component_1','pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring = 'accuracy', cv = 3)
print(scores_pca)

print(np.mean(scores_pca))
```

    [0.98 0.98 1.  ]
    0.9866666666666667


## 신용카드 연체 예측 데이터
Data Set : https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


```python
import os
print(os.getcwd())
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/credit_card_default_data')
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/credit_card_default_data



```python
import pandas as pd

df = pd.read_excel('./default of credit card clients.xls', header = 1, sheet_name = 'Data').iloc[0:, 1:]

print(df.shape)
df.head(3)
```

    (30000, 24)





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
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
# 컬럼명 변경
df.rename(columns = {'PAY_0':'PAY_1', 'default payment next month':'default'}, inplace = True)

# 속성과 클래스로 데이터 분류
y_target = df['default']
X_features = df.drop('default', axis = 1)

y_target.value_counts()
```




    0    23364
    1     6636
    Name: default, dtype: int64




```python
X_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 23 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   LIMIT_BAL  30000 non-null  int64
     1   SEX        30000 non-null  int64
     2   EDUCATION  30000 non-null  int64
     3   MARRIAGE   30000 non-null  int64
     4   AGE        30000 non-null  int64
     5   PAY_1      30000 non-null  int64
     6   PAY_2      30000 non-null  int64
     7   PAY_3      30000 non-null  int64
     8   PAY_4      30000 non-null  int64
     9   PAY_5      30000 non-null  int64
     10  PAY_6      30000 non-null  int64
     11  BILL_AMT1  30000 non-null  int64
     12  BILL_AMT2  30000 non-null  int64
     13  BILL_AMT3  30000 non-null  int64
     14  BILL_AMT4  30000 non-null  int64
     15  BILL_AMT5  30000 non-null  int64
     16  BILL_AMT6  30000 non-null  int64
     17  PAY_AMT1   30000 non-null  int64
     18  PAY_AMT2   30000 non-null  int64
     19  PAY_AMT3   30000 non-null  int64
     20  PAY_AMT4   30000 non-null  int64
     21  PAY_AMT5   30000 non-null  int64
     22  PAY_AMT6   30000 non-null  int64
    dtypes: int64(23)
    memory usage: 5.3 MB



```python
y_target.value_counts()
```




    0    23364
    1     6636
    Name: default, dtype: int64




```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

corr = X_features.corr()
plt.figure(figsize = (14,14))
sns.heatmap(corr, annot = True, fmt = '.1g')
```




    <AxesSubplot:>




![output_20_1](https://user-images.githubusercontent.com/87309905/197978858-35c6cd80-60a2-4919-a0e2-874678165541.png)
   



```python
# 일부 상관도가 높은 피처들(BILL_AMT1 ~ 6)을 PCA(n_components = 2) 변환 후 변동성 확인
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# BIll_AMT1 ~ BILL_AMT6 까지 6개의 속성명 생성
cols_bill = ['BILL_AMT'+str(i) for i in range(1,7)]
print(f'대상 속성명: {cols_bill}')

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_계산 위해 fit()호출
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components = 2)
pca.fit(df_cols_scaled)

print(f'PCA Component별 변동성 : {pca.explained_variance_ratio_}')
```

    대상 속성명: ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PCA Component별 변동성 : [0.90555253 0.0509867 ]



```python
# 전체 원본 데이터와 PCA변환된 데이터 간 랜덤 포레스트 예측 성능 비교
# 1. 원본 데이터
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators = 300, random_state = 156)

# 원본 데이터일 때 랜덤 포레스트 예측 성능
scores = cross_val_score(rcf, X_features, y_target, scoring = 'accuracy', cv = 3)

print(f'CV=3 인 경우의 개별 Fold세트별 정확도 : {scores}')
print(f'평균 정확도 :{np.mean(scores):.4f}')
```

    CV=3 인 경우의 개별 Fold세트별 정확도 : [0.8083 0.8196 0.8232]
    평균 정확도 :0.8170



```python
# 2. PCA 변환된 데이터
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 원본 데이터셋에 먼저 StandardScaler적용
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X_features)

# PCA 변환을 수행하고 랜덤 포레스트 예측 성능
for n in range(1,7):
    pca = PCA(n_components = n)
    df_pca = pca.fit_transform(df_scaled)
    scores_pca = cross_val_score(rcf, df_pca, y_target, scoring = 'accuracy', cv = 3)

    print(f'CV={n}인 경우의 PCA 변환된 개별 Fold세트별 정확도: {scores_pca}')
    print(f'PCA 변환 데이터 셋 평균 정확도 : {np.mean(scores_pca)}')
```

    CV=1인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.661  0.663  0.6617]
    PCA 변환 데이터 셋 평균 정확도 : 0.6619
    CV=2인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.77   0.7832 0.7877]
    PCA 변환 데이터 셋 평균 정확도 : 0.7803
    CV=3인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.7858 0.7925 0.797 ]
    PCA 변환 데이터 셋 평균 정확도 : 0.7917666666666667
    CV=4인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.7844 0.7962 0.7961]
    PCA 변환 데이터 셋 평균 정확도 : 0.7922333333333333
    CV=5인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.7867 0.7987 0.8041]
    PCA 변환 데이터 셋 평균 정확도 : 0.7965
    CV=6인 경우의 PCA 변환된 개별 Fold세트별 정확도: [0.7922 0.7967 0.8015]
    PCA 변환 데이터 셋 평균 정확도 : 0.7968000000000001


## LDA(Linear Discriminant Analysis)


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data)
```


```python
lda = LinearDiscriminantAnalysis(n_components = 2)
# fit 호출시 target값 입력
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)
print(iris_lda.shape)
```

    (150, 2)



```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

lda_columns=['lda_component_1','lda_component_2']
irisDF_lda = pd.DataFrame(iris_lda,columns=lda_columns)
irisDF_lda['target']=iris.target

#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 각 target 별로 다른 shape으로 scatter plot
for i, marker in enumerate(markers):
    x_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_1']
    y_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_2']

    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend(loc='upper right')
plt.xlabel('lda_component_1')
plt.ylabel('lda_component_2')
plt.show()
```


![output_27_0](https://user-images.githubusercontent.com/87309905/197978831-b8c95ada-2121-441c-b6c5-3254010d0909.png)
    




```python

```

## 특이값 분해 (SVD ; Singular Value Decomposition)


```python
# numpy의 scd 모듈 import 
import numpy as np
from numpy.linalg import svd

# 4*4 Random 행렬 a 생성
np.random.seed(121)
a = np.random.randn(4,4)
print(np.round(a,3))
```

    [[-0.212 -0.285 -0.574 -0.44 ]
     [-0.33   1.184  1.615  0.367]
     [-0.014  0.63   1.71  -1.327]
     [ 0.402 -0.191  1.404 -1.969]]



```python
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print('U matrix: \n',np.round(U,3))
print('Sigma  matrix: \n',np.round(Sigma,3))
print('U matrix: \n',np.round(Vt,3))
```

    (4, 4) (4,) (4, 4)
    U matrix: 
     [[-0.079 -0.318  0.867  0.376]
     [ 0.383  0.787  0.12   0.469]
     [ 0.656  0.022  0.357 -0.664]
     [ 0.645 -0.529 -0.328  0.444]]
    Sigma  matrix: 
     [3.423 2.023 0.463 0.079]
    U matrix: 
     [[ 0.041  0.224  0.786 -0.574]
     [-0.2    0.562  0.37   0.712]
     [-0.778  0.395 -0.333 -0.357]
     [-0.593 -0.692  0.366  0.189]]



```python
# Sigma를 다시 0을 포함한 대칭행렬로 변환
Sigma_mat = np.diag(Sigma)
a_ = np.dot(np.dot(U, Sigma_mat), Vt)
print(np.round(a_,3))
```

    [[-0.212 -0.285 -0.574 -0.44 ]
     [-0.33   1.184  1.615  0.367]
     [-0.014  0.63   1.71  -1.327]
     [ 0.402 -0.191  1.404 -1.969]]



```python
a[2] = a[0] + a[1]
a[3] = a[0]
print(np.round(a,3))
```

    [[-0.212 -0.285 -0.574 -0.44 ]
     [-0.33   1.184  1.615  0.367]
     [-0.542  0.899  1.041 -0.073]
     [-0.212 -0.285 -0.574 -0.44 ]]



```python
# 다시 SVD를 수행하여 Sigma값 확인
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print('Sigma Value:\n', np.round(Sigma,3))

```

    (4, 4) (4,) (4, 4)
    Sigma Value:
     [2.663 0.807 0.    0.   ]



```python
# U 행렬의 경우는 Sigma와 내적을 수행하므로 Sigma의 앞 2행에 대응되는 앞 2열만 추출
U_ = U[:, :2]
Sigma_ = np.diag(Sigma[:2])
# V 전치 행렬의 경우는 앞 2행만 추출
Vt_ = Vt[:2]
print(U_.shape, Sigma_.shape, Vt_.shape)
# U, Sigma, Vt의 내적을 수행하며, 다시 원본 행렬 복원
a_ = np.dot(np.dot(U_,Sigma_), Vt_)
print(np.round(a_,3))
```

    (4, 2) (2, 2) (2, 4)
    [[-0.212 -0.285 -0.574 -0.44 ]
     [-0.33   1.184  1.615  0.367]
     [-0.542  0.899  1.041 -0.073]
     [-0.212 -0.285 -0.574 -0.44 ]]


## Truncated SVD를 이용한 행렬 분해


```python
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

# 원본 행렬을 출력하고, SVD를 적용할 경우 U, Sigma, Vt 의 차원 확인 
np.random.seed(121)
matrix = np.random.random((6, 6))
print('원본 행렬:\n',matrix)
U, Sigma, Vt = svd(matrix, full_matrices=False)
print('\n분해 행렬 차원:',U.shape, Sigma.shape, Vt.shape)
print('\nSigma값 행렬:', Sigma)

# Truncated SVD로 Sigma 행렬의 특이값을 4개로 하여 Truncated SVD 수행. 
num_components = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix, k=num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
print('\nTruncated SVD Sigma값 행렬:', Sigma_tr)
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)), Vt_tr)  # output of TruncatedSVD

print('\nTruncated SVD로 분해 후 복원 행렬:\n', matrix_tr)
```

    원본 행렬:
     [[0.11133083 0.21076757 0.23296249 0.15194456 0.83017814 0.40791941]
     [0.5557906  0.74552394 0.24849976 0.9686594  0.95268418 0.48984885]
     [0.01829731 0.85760612 0.40493829 0.62247394 0.29537149 0.92958852]
     [0.4056155  0.56730065 0.24575605 0.22573721 0.03827786 0.58098021]
     [0.82925331 0.77326256 0.94693849 0.73632338 0.67328275 0.74517176]
     [0.51161442 0.46920965 0.6439515  0.82081228 0.14548493 0.01806415]]
    
    분해 행렬 차원: (6, 6) (6,) (6, 6)
    
    Sigma값 행렬: [3.2535007  0.88116505 0.83865238 0.55463089 0.35834824 0.0349925 ]
    
    Truncated SVD 분해 행렬 차원: (6, 4) (4,) (4, 6)
    
    Truncated SVD Sigma값 행렬: [0.55463089 0.83865238 0.88116505 3.2535007 ]
    
    Truncated SVD로 분해 후 복원 행렬:
     [[0.19222941 0.21792946 0.15951023 0.14084013 0.81641405 0.42533093]
     [0.44874275 0.72204422 0.34594106 0.99148577 0.96866325 0.4754868 ]
     [0.12656662 0.88860729 0.30625735 0.59517439 0.28036734 0.93961948]
     [0.23989012 0.51026588 0.39697353 0.27308905 0.05971563 0.57156395]
     [0.83806144 0.78847467 0.93868685 0.72673231 0.6740867  0.73812389]
     [0.59726589 0.47953891 0.56613544 0.80746028 0.13135039 0.03479656]]


## 사이킷런 TruncatedSVD 클래스를 이용한 변환


```python
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline

iris = load_iris()
iris_ftrs = iris.data
# 2개의 주요 component로 TruncatedSVD변환
tsvd = TruncatedSVD(n_components = 2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)

# Scatter plot 2차원으로 TruncatedSVD 변환된 데이터 표현, 품종은 색깔로 구분
plt.scatter(x = iris_tsvd[:, 0], y = iris_tsvd[:,1], c = iris.target)
plt.xlabel('TruncatedSVD Component 1')
plt.ylabel('TruncatedSVD Component 2')
```




    Text(0, 0.5, 'TruncatedSVD Component 2')



![output_39_1](https://user-images.githubusercontent.com/87309905/197978786-60d8829f-4e8c-48ff-a8e6-580fb324b0e8.png)
   



```python
from sklearn.preprocessing import StandardScaler

# iris 데이터를 StandardScaler로 변환
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_ftrs)

# 스케일링된 데이터를 기반으로 TrunxatedSVD 변환 수행
tsvd = TruncatedSVD(n_components = 2)
tsvd.fit(iris_scaled)
iris_tsvd = tsvd.transform(iris_scaled)

# 스케일링된 데이터를 기반으로 PCA 변환 수행
pca = PCA(n_components = 2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

# TruncatedSVD변환 데이터를 왼쪽에, PCA변환 데이터를 오른쪽에 표현
fig, (ax1, ax2) = plt.subplots(figsize=(9,4),ncols = 2)
ax1.scatter(x = iris_tsvd[:,0], y = iris_tsvd[:,1], c = iris.target)
ax2.scatter(x = iris_pca[:,0], y = iris_pca[:,1], c = iris.target)
ax1.set_title('Truncated SVD Transformed')
ax2.set_title('PCA transformed')

```




    Text(0.5, 1.0, 'PCA transformed')




![output_40_1](https://user-images.githubusercontent.com/87309905/197978760-a90211b7-c639-488b-b378-55f1345ce7f2.png)
    


## NMF(Non Negative Matric Factorization)

### 행렬분해 (Matrix Factorization)


```python
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline

iris = load_iris()
iris_ftrs = iris.data
nmf = NMF(n_components = 2)
nmf.fit(iris_ftrs)
iris_nmf = nmf.transform(iris_ftrs)
plt.scatter(x=iris_nmf[:,0], y=iris_nmf[:,1], c=iris.target)
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')
```

    /opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/sklearn/decomposition/_nmf.py:1692: ConvergenceWarning: Maximum number of iterations 200 reached. Increase it to improve convergence.
      warnings.warn(





    Text(0, 0.5, 'NMF Component 2')




![output_43_2](https://user-images.githubusercontent.com/87309905/197978733-a9ba2fe4-6d36-40e7-b30a-94a9d90e849c.png)
    
