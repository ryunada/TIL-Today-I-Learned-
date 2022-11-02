# 불균형 데이터로 신용카드 사기탐지 모델 구현하기

## 실습 가이드
1. 데이터를 다운로드하여 Colab에 불러옵니다.
2. 필요한 라이브러리는 모두 코드로 작성되어 있습니다.
3. 코드는 위에서부터 아래로 순서대로 실행합니다.
4. 전체 문제 구성은 좌측 첫 번째 아이콘을 통해 확인할 수 있습니다.

## Step 0. 사기탐지 분류 모형 개요

### 금융 데이터의 특성 (Review)
- 1) 이종(heterogeneous) 데이터의 결합
- 2) <b>분포의 편향성(skewness)</b>
- 3) 분류 레이블의 불명확성
- 4) 변수의 다중공선성(multicollinearity)
- 5) 변수의 비선형성
- 그 외 현실적인 규제·수집·저장 등의 한계 때문에 데이터가 불완전(missing, truncated, censored)할 수 있음

사기탐지(Fraud Detection) 분류는 주로 2)와 관련한 금융 데이터의 특성을 가진 문제입니다. 

### 불균형 데이터의 머신러닝

- 데이터 불균형(Imbalanced Data): 머신러닝의 지도학습에서 분류하기 위한 각 클래스(레이블)에 해당하는 데이터의 양에 차이가 큰 경우
- 특정 클래스가 부족할 때 생기는 문제: (1) 과대적합, (2) 알고리즘이 수렴하지 않는 현상 발생


- 1) X (피처)의 불균형
    - 범주변수일 경우 범주에 따라 빈도가 낮을 수 있음
    - 고차원 피처 공간의 축소(Feature Transformation)
    - PCA, t-SNE 등의 알고리즘 사용
    
    
- 2) y (타겟)의 불균형
    - 여신(대출), 수신(적금), 보험(클레임), 카드(사기탐지), 거시경제(불황) 등 대부분의 금융 데이터는 희소 타겟 문제
    - 리샘플링(Resampling)으로 저빈도 데이터를 극복
    - 무선 과대표집(Random Oversampling), 무선 과소표집(Random Undersampling), SMOTE, Tomek Links 등의 알고리즘 사용

### 학습목표

- 1) 불균형 데이터 분류 문제에 대한 이해
- 2) 피처 변환 알고리즘의 이해
- 3) 과대적합 발생시 해결 방법 습득
- 4) 리샘플링 알고리즘에 대한 이해
- 5) 불균형 데이터를 이용한 분류 결과의 올바른 해석 방법 습득

## Step 1. 데이터를 학습에 맞게 변환하기

- 데이터 소개
https://www.kaggle.com/mlg-ulb/creditcardfraud


```python
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
```


```python
import warnings 
warnings.filterwarnings("ignore")
```


```python
import os
print(os.getcwd())
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/DeepLearning/Park_Professor_10.31/data')
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/DeepLearning/Park_Professor_10.31
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/DeepLearning/Park_Professor_10.31/data


### 문제 01. 데이터 확인하기


```python
# filepath = 'https://github.com/mchoimis/financialml/raw/main/fraud/'
```


```python
# 파일 불러오기
df =  pd.read_csv('./creditcard.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB



```python
# 로드한 데이터의 맨 윗 30개 행 확인하기
df.head(30)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>1.449044</td>
      <td>-1.176339</td>
      <td>0.913860</td>
      <td>-1.375667</td>
      <td>-1.971383</td>
      <td>-0.629152</td>
      <td>-1.423236</td>
      <td>0.048456</td>
      <td>-1.720408</td>
      <td>...</td>
      <td>-0.009302</td>
      <td>0.313894</td>
      <td>0.027740</td>
      <td>0.500512</td>
      <td>0.251367</td>
      <td>-0.129478</td>
      <td>0.042850</td>
      <td>0.016253</td>
      <td>7.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0</td>
      <td>0.384978</td>
      <td>0.616109</td>
      <td>-0.874300</td>
      <td>-0.094019</td>
      <td>2.924584</td>
      <td>3.317027</td>
      <td>0.470455</td>
      <td>0.538247</td>
      <td>-0.558895</td>
      <td>...</td>
      <td>0.049924</td>
      <td>0.238422</td>
      <td>0.009130</td>
      <td>0.996710</td>
      <td>-0.767315</td>
      <td>-0.492208</td>
      <td>0.042472</td>
      <td>-0.054337</td>
      <td>9.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10.0</td>
      <td>1.249999</td>
      <td>-1.221637</td>
      <td>0.383930</td>
      <td>-1.234899</td>
      <td>-1.485419</td>
      <td>-0.753230</td>
      <td>-0.689405</td>
      <td>-0.227487</td>
      <td>-2.094011</td>
      <td>...</td>
      <td>-0.231809</td>
      <td>-0.483285</td>
      <td>0.084668</td>
      <td>0.392831</td>
      <td>0.161135</td>
      <td>-0.354990</td>
      <td>0.026416</td>
      <td>0.042422</td>
      <td>121.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11.0</td>
      <td>1.069374</td>
      <td>0.287722</td>
      <td>0.828613</td>
      <td>2.712520</td>
      <td>-0.178398</td>
      <td>0.337544</td>
      <td>-0.096717</td>
      <td>0.115982</td>
      <td>-0.221083</td>
      <td>...</td>
      <td>-0.036876</td>
      <td>0.074412</td>
      <td>-0.071407</td>
      <td>0.104744</td>
      <td>0.548265</td>
      <td>0.104094</td>
      <td>0.021491</td>
      <td>0.021293</td>
      <td>27.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.0</td>
      <td>-2.791855</td>
      <td>-0.327771</td>
      <td>1.641750</td>
      <td>1.767473</td>
      <td>-0.136588</td>
      <td>0.807596</td>
      <td>-0.422911</td>
      <td>-1.907107</td>
      <td>0.755713</td>
      <td>...</td>
      <td>1.151663</td>
      <td>0.222182</td>
      <td>1.020586</td>
      <td>0.028317</td>
      <td>-0.232746</td>
      <td>-0.235557</td>
      <td>-0.164778</td>
      <td>-0.030154</td>
      <td>58.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12.0</td>
      <td>-0.752417</td>
      <td>0.345485</td>
      <td>2.057323</td>
      <td>-1.468643</td>
      <td>-1.158394</td>
      <td>-0.077850</td>
      <td>-0.608581</td>
      <td>0.003603</td>
      <td>-0.436167</td>
      <td>...</td>
      <td>0.499625</td>
      <td>1.353650</td>
      <td>-0.256573</td>
      <td>-0.065084</td>
      <td>-0.039124</td>
      <td>-0.087086</td>
      <td>-0.180998</td>
      <td>0.129394</td>
      <td>15.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12.0</td>
      <td>1.103215</td>
      <td>-0.040296</td>
      <td>1.267332</td>
      <td>1.289091</td>
      <td>-0.735997</td>
      <td>0.288069</td>
      <td>-0.586057</td>
      <td>0.189380</td>
      <td>0.782333</td>
      <td>...</td>
      <td>-0.024612</td>
      <td>0.196002</td>
      <td>0.013802</td>
      <td>0.103758</td>
      <td>0.364298</td>
      <td>-0.382261</td>
      <td>0.092809</td>
      <td>0.037051</td>
      <td>12.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13.0</td>
      <td>-0.436905</td>
      <td>0.918966</td>
      <td>0.924591</td>
      <td>-0.727219</td>
      <td>0.915679</td>
      <td>-0.127867</td>
      <td>0.707642</td>
      <td>0.087962</td>
      <td>-0.665271</td>
      <td>...</td>
      <td>-0.194796</td>
      <td>-0.672638</td>
      <td>-0.156858</td>
      <td>-0.888386</td>
      <td>-0.342413</td>
      <td>-0.049027</td>
      <td>0.079692</td>
      <td>0.131024</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>14.0</td>
      <td>-5.401258</td>
      <td>-5.450148</td>
      <td>1.186305</td>
      <td>1.736239</td>
      <td>3.049106</td>
      <td>-1.763406</td>
      <td>-1.559738</td>
      <td>0.160842</td>
      <td>1.233090</td>
      <td>...</td>
      <td>-0.503600</td>
      <td>0.984460</td>
      <td>2.458589</td>
      <td>0.042119</td>
      <td>-0.481631</td>
      <td>-0.621272</td>
      <td>0.392053</td>
      <td>0.949594</td>
      <td>46.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15.0</td>
      <td>1.492936</td>
      <td>-1.029346</td>
      <td>0.454795</td>
      <td>-1.438026</td>
      <td>-1.555434</td>
      <td>-0.720961</td>
      <td>-1.080664</td>
      <td>-0.053127</td>
      <td>-1.978682</td>
      <td>...</td>
      <td>-0.177650</td>
      <td>-0.175074</td>
      <td>0.040002</td>
      <td>0.295814</td>
      <td>0.332931</td>
      <td>-0.220385</td>
      <td>0.022298</td>
      <td>0.007602</td>
      <td>5.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>16.0</td>
      <td>0.694885</td>
      <td>-1.361819</td>
      <td>1.029221</td>
      <td>0.834159</td>
      <td>-1.191209</td>
      <td>1.309109</td>
      <td>-0.878586</td>
      <td>0.445290</td>
      <td>-0.446196</td>
      <td>...</td>
      <td>-0.295583</td>
      <td>-0.571955</td>
      <td>-0.050881</td>
      <td>-0.304215</td>
      <td>0.072001</td>
      <td>-0.422234</td>
      <td>0.086553</td>
      <td>0.063499</td>
      <td>231.71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>17.0</td>
      <td>0.962496</td>
      <td>0.328461</td>
      <td>-0.171479</td>
      <td>2.109204</td>
      <td>1.129566</td>
      <td>1.696038</td>
      <td>0.107712</td>
      <td>0.521502</td>
      <td>-1.191311</td>
      <td>...</td>
      <td>0.143997</td>
      <td>0.402492</td>
      <td>-0.048508</td>
      <td>-1.371866</td>
      <td>0.390814</td>
      <td>0.199964</td>
      <td>0.016371</td>
      <td>-0.014605</td>
      <td>34.09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>18.0</td>
      <td>1.166616</td>
      <td>0.502120</td>
      <td>-0.067300</td>
      <td>2.261569</td>
      <td>0.428804</td>
      <td>0.089474</td>
      <td>0.241147</td>
      <td>0.138082</td>
      <td>-0.989162</td>
      <td>...</td>
      <td>0.018702</td>
      <td>-0.061972</td>
      <td>-0.103855</td>
      <td>-0.370415</td>
      <td>0.603200</td>
      <td>0.108556</td>
      <td>-0.040521</td>
      <td>-0.011418</td>
      <td>2.28</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>18.0</td>
      <td>0.247491</td>
      <td>0.277666</td>
      <td>1.185471</td>
      <td>-0.092603</td>
      <td>-1.314394</td>
      <td>-0.150116</td>
      <td>-0.946365</td>
      <td>-1.617935</td>
      <td>1.544071</td>
      <td>...</td>
      <td>1.650180</td>
      <td>0.200454</td>
      <td>-0.185353</td>
      <td>0.423073</td>
      <td>0.820591</td>
      <td>-0.227632</td>
      <td>0.336634</td>
      <td>0.250475</td>
      <td>22.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>22.0</td>
      <td>-1.946525</td>
      <td>-0.044901</td>
      <td>-0.405570</td>
      <td>-1.013057</td>
      <td>2.941968</td>
      <td>2.955053</td>
      <td>-0.063063</td>
      <td>0.855546</td>
      <td>0.049967</td>
      <td>...</td>
      <td>-0.579526</td>
      <td>-0.799229</td>
      <td>0.870300</td>
      <td>0.983421</td>
      <td>0.321201</td>
      <td>0.149650</td>
      <td>0.707519</td>
      <td>0.014600</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>22.0</td>
      <td>-2.074295</td>
      <td>-0.121482</td>
      <td>1.322021</td>
      <td>0.410008</td>
      <td>0.295198</td>
      <td>-0.959537</td>
      <td>0.543985</td>
      <td>-0.104627</td>
      <td>0.475664</td>
      <td>...</td>
      <td>-0.403639</td>
      <td>-0.227404</td>
      <td>0.742435</td>
      <td>0.398535</td>
      <td>0.249212</td>
      <td>0.274404</td>
      <td>0.359969</td>
      <td>0.243232</td>
      <td>26.43</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>23.0</td>
      <td>1.173285</td>
      <td>0.353498</td>
      <td>0.283905</td>
      <td>1.133563</td>
      <td>-0.172577</td>
      <td>-0.916054</td>
      <td>0.369025</td>
      <td>-0.327260</td>
      <td>-0.246651</td>
      <td>...</td>
      <td>0.067003</td>
      <td>0.227812</td>
      <td>-0.150487</td>
      <td>0.435045</td>
      <td>0.724825</td>
      <td>-0.337082</td>
      <td>0.016368</td>
      <td>0.030041</td>
      <td>41.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>23.0</td>
      <td>1.322707</td>
      <td>-0.174041</td>
      <td>0.434555</td>
      <td>0.576038</td>
      <td>-0.836758</td>
      <td>-0.831083</td>
      <td>-0.264905</td>
      <td>-0.220982</td>
      <td>-1.071425</td>
      <td>...</td>
      <td>-0.284376</td>
      <td>-0.323357</td>
      <td>-0.037710</td>
      <td>0.347151</td>
      <td>0.559639</td>
      <td>-0.280158</td>
      <td>0.042335</td>
      <td>0.028822</td>
      <td>16.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>23.0</td>
      <td>-0.414289</td>
      <td>0.905437</td>
      <td>1.727453</td>
      <td>1.473471</td>
      <td>0.007443</td>
      <td>-0.200331</td>
      <td>0.740228</td>
      <td>-0.029247</td>
      <td>-0.593392</td>
      <td>...</td>
      <td>0.077237</td>
      <td>0.457331</td>
      <td>-0.038500</td>
      <td>0.642522</td>
      <td>-0.183891</td>
      <td>-0.277464</td>
      <td>0.182687</td>
      <td>0.152665</td>
      <td>33.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>23.0</td>
      <td>1.059387</td>
      <td>-0.175319</td>
      <td>1.266130</td>
      <td>1.186110</td>
      <td>-0.786002</td>
      <td>0.578435</td>
      <td>-0.767084</td>
      <td>0.401046</td>
      <td>0.699500</td>
      <td>...</td>
      <td>0.013676</td>
      <td>0.213734</td>
      <td>0.014462</td>
      <td>0.002951</td>
      <td>0.294638</td>
      <td>-0.395070</td>
      <td>0.081461</td>
      <td>0.024220</td>
      <td>12.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 31 columns</p>
</div>




```python
# Missing 여부 확인하기
df.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
# 불러온 데이터의 클래스 분포 확인하기
plt.hist(df.Class)
plt.show()
df.groupby(by=['Class']).count()
```

![output_16_0](https://user-images.githubusercontent.com/87309905/199407358-63dc78fb-15f2-412f-8d5d-1d93500aafb7.png)
   





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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>...</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
      <td>284315</td>
    </tr>
    <tr>
      <th>1</th>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>...</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>




```python
print('Target class is ', '{0:0.4f}'. format( df.Class.mean()*100 ), '%') # df.Class.mean()*100 == 492/(284315+492)*100
```

    Target class is  0.1727 %


### 문제 02. 변수의 스케일 변환하기

### 참고: scikit-learn이 제공하는 스케일러 (Review)

scikit-learn에서 제공하는 피처 스케일러(scaler)
- `StandardScaler`: 기본 스케일, 각 피처의 평균을 0, 표준편차를 1로 변환
- `RobustScaler`: 위와 유사하지만 평균 대신 중간값(median)과 일분위, 삼분위값(quartile)을 사용하여 이상치 영향을 최소화
- `MinMaxScaler`: 모든 피처의 최대치와 최소치가 각각 1, 0이 되도록 스케일 조정
- `Normalizer`: 피처(컬럼)이 아니라 row마다 정규화되며, 유클리드 거리가 1이 되도록 데이터를 조정하여 빠르게 학습할 수 있게 함

<p> 스케일 조정을 하는 이유는 데이터의 값이 너무 크거나 작을 때 학습이 제대로 되지 않을 수도 있기 때문입니다. <b> 또한 스케일의 영향이 절대적인 분류기(예: knn과 같은 거리기반 알고리즘)의 경우, 스케일 조정을 필수적으로 검토해야 합니다. </b>
    
<p> 반면 어떤 항목은 원본 데이터의 분포를 유지하는 것이 나을 수도 있습니다. 예를 들어, 데이터가 거의 한 곳에 집중되어 있는 feature를 표준화시켜 분포를 같게 만들었을 때, 작은 단위의 변화가 큰 차이를 나타내는 것처럼 학습될 수도 있습니다. 또한 스케일의 영향을 크게 받지 않는 분류기(예: 트리 기반 앙상블 알고리즘)를 사용할 경우에도 성능이 준수하게 나오거나 과대적합(overfitting)의 우려가 적다면 생략할 수도 있습니다.
    
<p> <b>스케일 조정시 유의해야할 점은 원본 데이터의 의미를 잃어버릴 수 있다는 것입니다.</b> 최종적으로 답을 구하는 것이 목적이 아니라 모델의 해석이나 향후 다른 데이터셋으로의 응용이 더 중요할 때 원 피처에 대한 설명력을 잃어버린다면 모델 개선이 어려울 수도 있습니다. 이 점을 함께 고려하시면 좋겠습니다.

# 스케일러 구분

https://mkjjo.github.io/python/2019/01/10/scaler.html


```python
df.head()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# 데이터 스케일 조정하기
from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler() 
rob_scaler = RobustScaler() ##

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1)) 
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

# 원 데이터에서 Time 컬럼과 Amount 컬럼 제외하기 
df.drop(['Time','Amount'], axis=1, inplace=True)
```


```python
# 스케일 조정된 컬럼 추가하기 
scaled_amount = df['scaled_amount'] 
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True) 
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

## 스케일 조정된 데이터 확인하기 
df.head()
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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.783274</td>
      <td>-0.994983</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>...</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.269825</td>
      <td>-0.994983</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>...</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.983721</td>
      <td>-0.994972</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>...</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.418291</td>
      <td>-0.994972</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>...</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.670579</td>
      <td>-0.994960</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>...</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### 문제 03. 샘플 데이터 나누기


```python
# X와 y 데이터 셋 만들기
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import KFold, StratifiedKFold

X = df.drop('Class', axis=1) 
y = df['Class']

# 데이터 나누기
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in sss.split(X, y):
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index] 
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
```

## 샘플 축소 492 : 492


```python
# 클래스의 skew 정도가 매우 높기 때문에 클래스간 분포를 맞추는 것이 필요합니다.
# subsample 구축 전 셔플링을 통해 레이블이 한쪽에 몰려있지 않도록 하겠습니다.

df =  df.sample(frac = 1) # 1 -> 100% 랜덤

# 데이터 준비
# Class가 1(즉, 신용카드 사기)인 데이터 분리
# 1대1 비율로 분석을 진행한다.(즉, 데이터 492,492개)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# 데이터 셔플하기
new_df = normal_distributed_df.sample(frac=1, random_state=0)

# 셔플한 새로운 데이터 셋 확인
new_df.head()

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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152295</th>
      <td>-0.170195</td>
      <td>0.147358</td>
      <td>-17.537592</td>
      <td>12.352519</td>
      <td>-20.134613</td>
      <td>11.122771</td>
      <td>-14.571080</td>
      <td>-0.381622</td>
      <td>-23.928661</td>
      <td>-4.724921</td>
      <td>...</td>
      <td>1.925103</td>
      <td>-4.352213</td>
      <td>2.389041</td>
      <td>2.019128</td>
      <td>0.627192</td>
      <td>-1.085997</td>
      <td>-0.071803</td>
      <td>-3.838198</td>
      <td>-0.802564</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74291</th>
      <td>0.081464</td>
      <td>-0.342885</td>
      <td>1.288396</td>
      <td>0.123746</td>
      <td>-0.202700</td>
      <td>0.214985</td>
      <td>-0.093542</td>
      <td>-0.967644</td>
      <td>0.305824</td>
      <td>-0.289460</td>
      <td>...</td>
      <td>-0.043661</td>
      <td>-0.087110</td>
      <td>-0.271861</td>
      <td>-0.131381</td>
      <td>-0.066842</td>
      <td>0.574602</td>
      <td>0.629710</td>
      <td>-0.075460</td>
      <td>0.000610</td>
      <td>0</td>
    </tr>
    <tr>
      <th>251866</th>
      <td>-0.252917</td>
      <td>0.832282</td>
      <td>0.711155</td>
      <td>2.617105</td>
      <td>-4.722363</td>
      <td>5.842970</td>
      <td>-0.600179</td>
      <td>-1.646313</td>
      <td>-2.785198</td>
      <td>0.540368</td>
      <td>...</td>
      <td>0.461032</td>
      <td>0.360501</td>
      <td>-0.865526</td>
      <td>0.139978</td>
      <td>-0.336238</td>
      <td>0.128449</td>
      <td>-0.155646</td>
      <td>0.799460</td>
      <td>0.392170</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43846</th>
      <td>-0.097813</td>
      <td>-0.505398</td>
      <td>1.524784</td>
      <td>-0.414005</td>
      <td>-0.908836</td>
      <td>-1.174552</td>
      <td>0.233440</td>
      <td>-0.094322</td>
      <td>-0.191415</td>
      <td>-0.131176</td>
      <td>...</td>
      <td>0.110961</td>
      <td>0.124854</td>
      <td>0.246631</td>
      <td>-0.386705</td>
      <td>-1.353961</td>
      <td>0.980555</td>
      <td>0.005462</td>
      <td>-0.037688</td>
      <td>-0.026253</td>
      <td>0</td>
    </tr>
    <tr>
      <th>176108</th>
      <td>0.121149</td>
      <td>0.445694</td>
      <td>1.849442</td>
      <td>-0.668927</td>
      <td>0.141211</td>
      <td>0.960292</td>
      <td>-1.337755</td>
      <td>-0.122763</td>
      <td>-1.292617</td>
      <td>0.280271</td>
      <td>...</td>
      <td>-0.219220</td>
      <td>0.099728</td>
      <td>0.647218</td>
      <td>0.132593</td>
      <td>-0.050519</td>
      <td>-0.302508</td>
      <td>-0.205372</td>
      <td>0.099488</td>
      <td>0.005322</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



## Step 2. PCA와 t-SNE, SVD를 이용하여 차원 축소하기

### 참고: 차원축소 방법
    
- 주성분 분석(Principal Component Analysis)
- t-SNE (Stochastic Neighbor Embedding)
    - SNE는 n 차원에 분포된 이산 데이터를 k(n 이하의 정수) 차원으로 축소하며 거리 정보를 보존하되, 거리가 가까운 데이터의 정보를 우선하여 보존하기 위해 고안되었음
    - 단어 벡터와 같이 고차원 데이터를 시각화하는 데 가장 자주 쓰이는 알고리즘
    - SNE 학습과정에 사용되는 가우시안 분포는 t 분포에 비해 거리에 따른 확률 값 변화의 경사가 가파른 특징을 가지기 때문에 특정 거리 이상부터는 학습과정에 거의 반영이 되지 않는 문제점을 가지고 있음(Crowding Problem)
    - 이러한 문제점을 보완하기 위해 고안된 방법이 t-SNE: 학습과정에서 가우시안 분포 대신 t 분포를 이용
    - t-SNE는 보통 word2vec으로 임베딩한 단어벡터를 시각화하는데 쓰임
    
- 특이값 분해(Singular Value Decomposition)
- 그 외 잠재 의미분석(Latent Semantic Analysis), 행렬 인수분해(Matrix Factorization) 등

실무에서는 스케일이 매우 큰 실제 데이터를 분석하기 위해서, 여러 방법론을 융합하여 사용하는 것이 필요

### 문제 04. 차원 축소하기


```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# 차원 축소할 데이터 준비
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# t-SNE  

X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(X.values)
print('t-SNE done')

# PCA 
X_reduced_pca = PCA(n_components = 2, random_state = 0).fit_transform(X.values)
print('PCA done')

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components =2 , algorithm = 'randomized', random_state = 0).fit_transform(X.values)
print('Truncated SVD done')
```

    t-SNE done
    PCA done
    Truncated SVD done


### 문제 05. 결과 시각화하기


```python
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
f.suptitle('Clusters after Dimensionality Reduction', fontsize=16)

# Label 범례 설정
labels = ['No Fraud','Fraud']
blue_patch = mpatches.Patch(color = 'red', label = 'No Fraud')
red_patch = mpatches.Patch(color = 'blue', label = 'Fraud' )

# t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c = (y == 0), cmap='coolwarm', label='No Fraud', linewidths=2) ###
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c = (y == 1), cmap='coolwarm', label='Fraud', linewidths=2) ###
# ax1.set_title('t-SNE', fontsize=14)
# ax1.grid(True)
# ax1.legend(handles = [blue_patch, red_patch]) ###

# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c = (y == 0), cmap='coolwarm', label='No Fraud', linewidths=2) ###
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c = (y == 1), cmap='coolwarm', label='Fraud', linewidths=2) ###
ax2.set_title('PCA', fontsize=14)
ax2.grid(True)
ax2.legend(handles = [blue_patch, red_patch]) ###

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1],  c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2) ###
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1],  c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2) ###
ax3.set_title('Truncated SVD', fontsize=14)
ax3.grid(True)
ax3.legend(handles = [blue_patch, red_patch]) ###

plt.show()
```

![output_33_0](https://user-images.githubusercontent.com/87309905/199407309-0863e282-ae29-4de2-895e-fcfe0e7f4c76.png)

 


## Step 3. Random Undersampling 으로 샘플 재구축하기

### 문제 06. 재구축 샘플로 분류모델 구현하기


```python
# 재구축한 데이터의 클래스 분포 확인하기

new_df.groupby(by = ['Class']).count()
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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>...</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
    </tr>
    <tr>
      <th>1</th>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>...</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
      <td>492</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>




```python
# X와 y 데이터 셋 만들기
X = new_df.drop('Class', axis=1)
y = new_df['Class']
```


```python
# 언더샘플링을 위한 샘플 데이터 구축
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
# 모델 인풋에 들어가기 위한 데이터의 형태 바꾸기
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
```


```python
# 학습시킬 모델 로드하기
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

classifiers = {
    "Logisitic Regression": LogisticRegression(),
    "K Nearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "LightGBM Classifier": LGBMClassifier()
}
```


```python
# 모델별 cross validation 한 결과의 평균 정확도 점수 출력하기
from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print(classifier.__class__.__name__, ':', round(training_score.mean(), 2) * 100, '% accuracy')
```

    LogisticRegression : 94.0 % accuracy
    KNeighborsClassifier : 93.0 % accuracy
    SVC : 93.0 % accuracy
    DecisionTreeClassifier : 90.0 % accuracy
    RandomForestClassifier : 94.0 % accuracy
    GradientBoostingClassifier : 94.0 % accuracy
    LGBMClassifier : 94.0 % accuracy


### 문제 07. 분류 결과 확인하기


```python
### 올바른 예
```


```python
original_Xtest
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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>212516</th>
      <td>-0.307273</td>
      <td>0.636779</td>
      <td>-1.298443</td>
      <td>1.948100</td>
      <td>-4.509947</td>
      <td>1.305805</td>
      <td>-0.019486</td>
      <td>-0.509238</td>
      <td>-2.643398</td>
      <td>1.283545</td>
      <td>...</td>
      <td>0.152892</td>
      <td>0.250415</td>
      <td>1.178032</td>
      <td>1.360989</td>
      <td>-0.272013</td>
      <td>-0.325948</td>
      <td>0.290703</td>
      <td>0.841295</td>
      <td>0.643094</td>
      <td>0.201156</td>
    </tr>
    <tr>
      <th>212644</th>
      <td>9.863900</td>
      <td>0.637343</td>
      <td>-2.356348</td>
      <td>1.746360</td>
      <td>-6.374624</td>
      <td>1.772205</td>
      <td>-3.439294</td>
      <td>1.457811</td>
      <td>-0.362577</td>
      <td>1.443791</td>
      <td>...</td>
      <td>-0.663371</td>
      <td>0.194810</td>
      <td>0.857942</td>
      <td>0.621203</td>
      <td>0.964817</td>
      <td>-0.619437</td>
      <td>-1.732613</td>
      <td>0.108361</td>
      <td>1.130828</td>
      <td>0.415703</td>
    </tr>
    <tr>
      <th>213092</th>
      <td>0.006567</td>
      <td>0.639281</td>
      <td>-4.666500</td>
      <td>-3.952320</td>
      <td>0.206094</td>
      <td>5.153525</td>
      <td>5.229469</td>
      <td>0.939040</td>
      <td>-0.635033</td>
      <td>-0.704506</td>
      <td>...</td>
      <td>1.060154</td>
      <td>-2.286137</td>
      <td>-0.664263</td>
      <td>1.821422</td>
      <td>0.113563</td>
      <td>-0.759673</td>
      <td>-0.502304</td>
      <td>0.630639</td>
      <td>-0.513880</td>
      <td>0.729526</td>
    </tr>
    <tr>
      <th>213116</th>
      <td>-0.191434</td>
      <td>0.639399</td>
      <td>-3.975939</td>
      <td>-1.244939</td>
      <td>-3.707414</td>
      <td>4.544772</td>
      <td>4.050676</td>
      <td>-3.407679</td>
      <td>-5.063118</td>
      <td>1.007042</td>
      <td>...</td>
      <td>3.569733</td>
      <td>2.109403</td>
      <td>1.059737</td>
      <td>-0.037395</td>
      <td>0.348707</td>
      <td>-0.162929</td>
      <td>0.410531</td>
      <td>-0.123612</td>
      <td>0.877424</td>
      <td>0.667568</td>
    </tr>
    <tr>
      <th>214662</th>
      <td>1.376930</td>
      <td>0.647035</td>
      <td>0.467992</td>
      <td>1.100118</td>
      <td>-5.607145</td>
      <td>2.204714</td>
      <td>-0.578539</td>
      <td>-0.174200</td>
      <td>-3.454201</td>
      <td>1.102823</td>
      <td>...</td>
      <td>-0.173814</td>
      <td>0.589575</td>
      <td>0.983481</td>
      <td>0.899876</td>
      <td>-0.285103</td>
      <td>-1.929717</td>
      <td>0.319869</td>
      <td>0.170636</td>
      <td>0.851798</td>
      <td>0.372098</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>284802</th>
      <td>-0.296653</td>
      <td>1.034951</td>
      <td>-11.881118</td>
      <td>10.071785</td>
      <td>-9.834783</td>
      <td>-2.066656</td>
      <td>-5.364473</td>
      <td>-2.606837</td>
      <td>-4.918215</td>
      <td>7.305334</td>
      <td>...</td>
      <td>-0.682920</td>
      <td>1.475829</td>
      <td>0.213454</td>
      <td>0.111864</td>
      <td>1.014480</td>
      <td>-0.509348</td>
      <td>1.436807</td>
      <td>0.250034</td>
      <td>0.943651</td>
      <td>0.823731</td>
    </tr>
    <tr>
      <th>284803</th>
      <td>0.038986</td>
      <td>1.034963</td>
      <td>-0.732789</td>
      <td>-0.055080</td>
      <td>2.035030</td>
      <td>-0.738589</td>
      <td>0.868229</td>
      <td>1.058415</td>
      <td>0.024330</td>
      <td>0.294869</td>
      <td>...</td>
      <td>-1.545556</td>
      <td>0.059616</td>
      <td>0.214205</td>
      <td>0.924384</td>
      <td>0.012463</td>
      <td>-1.016226</td>
      <td>-0.606624</td>
      <td>-0.395255</td>
      <td>0.068472</td>
      <td>-0.053527</td>
    </tr>
    <tr>
      <th>284804</th>
      <td>0.641096</td>
      <td>1.034975</td>
      <td>1.919565</td>
      <td>-0.301254</td>
      <td>-3.249640</td>
      <td>-0.557828</td>
      <td>2.630515</td>
      <td>3.031260</td>
      <td>-0.296827</td>
      <td>0.708417</td>
      <td>...</td>
      <td>-0.577252</td>
      <td>0.001396</td>
      <td>0.232045</td>
      <td>0.578229</td>
      <td>-0.037501</td>
      <td>0.640134</td>
      <td>0.265745</td>
      <td>-0.087371</td>
      <td>0.004455</td>
      <td>-0.026561</td>
    </tr>
    <tr>
      <th>284805</th>
      <td>-0.167680</td>
      <td>1.034975</td>
      <td>-0.240440</td>
      <td>0.530483</td>
      <td>0.702510</td>
      <td>0.689799</td>
      <td>-0.377961</td>
      <td>0.623708</td>
      <td>-0.686180</td>
      <td>0.679145</td>
      <td>...</td>
      <td>2.897849</td>
      <td>0.127434</td>
      <td>0.265245</td>
      <td>0.800049</td>
      <td>-0.163298</td>
      <td>0.123205</td>
      <td>-0.569159</td>
      <td>0.546668</td>
      <td>0.108821</td>
      <td>0.104533</td>
    </tr>
    <tr>
      <th>284806</th>
      <td>2.724796</td>
      <td>1.035022</td>
      <td>-0.533413</td>
      <td>-0.189733</td>
      <td>0.703337</td>
      <td>-0.506271</td>
      <td>-0.012546</td>
      <td>-0.649617</td>
      <td>1.577006</td>
      <td>-0.414650</td>
      <td>...</td>
      <td>-0.256117</td>
      <td>0.382948</td>
      <td>0.261057</td>
      <td>0.643078</td>
      <td>0.376777</td>
      <td>0.008797</td>
      <td>-0.473649</td>
      <td>-0.818267</td>
      <td>-0.002415</td>
      <td>0.013649</td>
    </tr>
  </tbody>
</table>
<p>56961 rows × 30 columns</p>
</div>




```python
# 모델별 분류결과 확인하기 (올바른 예)
from sklearn.metrics import classification_report

for key, classifier in classifiers.items():
    y_pred = classifier.predict(original_Xtest)  ####
    results = classification_report(original_ytest, y_pred)  ####
    print(classifier.__class__.__name__, '-------','\n', results)
```

    LogisticRegression ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.97      0.98     56863
               1       0.05      0.92      0.09        98
    
        accuracy                           0.97     56961
       macro avg       0.52      0.94      0.54     56961
    weighted avg       1.00      0.97      0.98     56961
    
    KNeighborsClassifier ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.98      0.99     56863
               1       0.06      0.92      0.12        98
    
        accuracy                           0.98     56961
       macro avg       0.53      0.95      0.55     56961
    weighted avg       1.00      0.98      0.99     56961
    
    SVC ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.99      0.99     56863
               1       0.09      0.88      0.17        98
    
        accuracy                           0.98     56961
       macro avg       0.55      0.93      0.58     56961
    weighted avg       1.00      0.98      0.99     56961
    
    DecisionTreeClassifier ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.91      0.95     56863
               1       0.02      0.95      0.04        98
    
        accuracy                           0.91     56961
       macro avg       0.51      0.93      0.49     56961
    weighted avg       1.00      0.91      0.95     56961
    
    RandomForestClassifier ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.97      0.98     56863
               1       0.05      0.97      0.09        98
    
        accuracy                           0.97     56961
       macro avg       0.52      0.97      0.53     56961
    weighted avg       1.00      0.97      0.98     56961
    
    GradientBoostingClassifier ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.96      0.98     56863
               1       0.04      0.97      0.08        98
    
        accuracy                           0.96     56961
       macro avg       0.52      0.96      0.53     56961
    weighted avg       1.00      0.96      0.98     56961
    
    LGBMClassifier ------- 
                   precision    recall  f1-score   support
    
               0       1.00      0.96      0.98     56863
               1       0.04      0.97      0.08        98
    
        accuracy                           0.96     56961
       macro avg       0.52      0.97      0.53     56961
    weighted avg       1.00      0.96      0.98     56961
    



```python
# 모델별 Confusion Matrix 확인하기 (올바른 예)
from sklearn.metrics import confusion_matrix

for key, classifier in classifiers.items():
    y_pred = classifier.predict(original_Xtest) ####
    cm = confusion_matrix(original_ytest, y_pred) ####
    print(classifier.__class__.__name__, '\n', cm, '\n')
```

    LogisticRegression 
     [[55127  1736]
     [    8    90]] 
    
    KNeighborsClassifier 
     [[55522  1341]
     [    8    90]] 
    
    SVC 
     [[56011   852]
     [   12    86]] 
    
    DecisionTreeClassifier 
     [[51820  5043]
     [    5    93]] 
    
    RandomForestClassifier 
     [[54888  1975]
     [    3    95]] 
    
    GradientBoostingClassifier 
     [[54555  2308]
     [    3    95]] 
    
    LGBMClassifier 
     [[54804  2059]
     [    3    95]] 
    



```python
### 잘못된 예
```


```python
# 모델별 분류결과 확인하기 (잘못된 예)
from sklearn.metrics import classification_report

for key, classifier in classifiers.items():
    y_pred = classifier.predict(X_test)  ####
    results_wrong = classification_report(y_test, y_pred)   ####
    print(classifier.__class__.__name__, '-------','\n', results_wrong)
```

    LogisticRegression ------- 
                   precision    recall  f1-score   support
    
               0       0.94      0.94      0.94       100
               1       0.94      0.94      0.94        97
    
        accuracy                           0.94       197
       macro avg       0.94      0.94      0.94       197
    weighted avg       0.94      0.94      0.94       197
    
    KNeighborsClassifier ------- 
                   precision    recall  f1-score   support
    
               0       0.92      0.97      0.94       100
               1       0.97      0.91      0.94        97
    
        accuracy                           0.94       197
       macro avg       0.94      0.94      0.94       197
    weighted avg       0.94      0.94      0.94       197
    
    SVC ------- 
                   precision    recall  f1-score   support
    
               0       0.92      0.98      0.95       100
               1       0.98      0.91      0.94        97
    
        accuracy                           0.94       197
       macro avg       0.95      0.94      0.94       197
    weighted avg       0.95      0.94      0.94       197
    
    DecisionTreeClassifier ------- 
                   precision    recall  f1-score   support
    
               0       0.90      0.90      0.90       100
               1       0.90      0.90      0.90        97
    
        accuracy                           0.90       197
       macro avg       0.90      0.90      0.90       197
    weighted avg       0.90      0.90      0.90       197
    
    RandomForestClassifier ------- 
                   precision    recall  f1-score   support
    
               0       0.93      0.95      0.94       100
               1       0.95      0.93      0.94        97
    
        accuracy                           0.94       197
       macro avg       0.94      0.94      0.94       197
    weighted avg       0.94      0.94      0.94       197
    
    GradientBoostingClassifier ------- 
                   precision    recall  f1-score   support
    
               0       0.92      0.93      0.93       100
               1       0.93      0.92      0.92        97
    
        accuracy                           0.92       197
       macro avg       0.92      0.92      0.92       197
    weighted avg       0.92      0.92      0.92       197
    
    LGBMClassifier ------- 
                   precision    recall  f1-score   support
    
               0       0.94      0.93      0.93       100
               1       0.93      0.94      0.93        97
    
        accuracy                           0.93       197
       macro avg       0.93      0.93      0.93       197
    weighted avg       0.93      0.93      0.93       197
    



```python
# 모델별 Confusion Matrix 확인하기 (잘못된 예)
from sklearn.metrics import confusion_matrix

for key, classifier in classifiers.items():
    y_pred = classifier.predict(X_test)  ####
    cm_wrong = confusion_matrix(y_test, y_pred)   ####
    print(classifier.__class__.__name__, '\n', cm_wrong, '\n')
```

    LogisticRegression 
     [[94  6]
     [ 6 91]] 
    
    KNeighborsClassifier 
     [[97  3]
     [ 9 88]] 
    
    SVC 
     [[98  2]
     [ 9 88]] 
    
    DecisionTreeClassifier 
     [[90 10]
     [10 87]] 
    
    RandomForestClassifier 
     [[95  5]
     [ 7 90]] 
    
    GradientBoostingClassifier 
     [[93  7]
     [ 8 89]] 
    
    LGBMClassifier 
     [[93  7]
     [ 6 91]] 
    


## Step 4. SMOTE 로 Oversampling 하기 

### 참고:

- 리샘플링(Synthetic Minority Oversampling Technique)

- 모델 파라미터 조정
    - `scale_pos_weight`
    - `is_unbalance`
    - `{class_label: weight}`

### 문제 08. SMOTE로 Oversampling하기


```python
from imblearn.over_sampling import SMOTE


sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(original_Xtrain,list(original_ytrain))  ####


print('Before SMOTE, original X_train: {}'.format(original_Xtrain.shape)) 
print('Before SMOTE, original y_train: {}'.format(np.array(original_ytrain).shape))
print('After  SMOTE, resampled original X_train: {}'.format(X_resampled.shape)) 
print('After  SMOTE, resampled original y_train: {} \n'.format(np.array(y_resampled).shape))

print("Before SMOTE,     fraud counts: {}".format(sum(np.array(original_ytrain)==1)))
print("Before SMOTE, non-fraud counts: {}".format(sum(np.array(original_ytrain)==0)))
print("After  SMOTE,     fraud counts: {}".format(sum(np.array(y_resampled)==1)))
print("After  SMOTE, non-fraud counts: {}".format(sum(np.array(y_resampled)==0)))
```

    Before SMOTE, original X_train: (227846, 30)
    Before SMOTE, original y_train: (227846,)
    After  SMOTE, resampled original X_train: (454904, 30)
    After  SMOTE, resampled original y_train: (454904,) 
    
    Before SMOTE,     fraud counts: 394
    Before SMOTE, non-fraud counts: 227452
    After  SMOTE,     fraud counts: 227452
    After  SMOTE, non-fraud counts: 227452


### 문제 09. 재구축한 샘플로 분류 모형 구현하기(2가지 방법)


```python
# 방법 1: 모델의 파라미터를 조정하는 방법 
```


```python
from sklearn.metrics import accuracy_score, recall_score
# f1_score, roc_auc_score, precision_score
```


```python
# Logistic Regression 모델의 weight 파라미터 지정하기

w = {1:0, 1:99} ## 불균형 클래스 weight 파라미터 지정

# 모델 피팅
logreg_weighted = LogisticRegression(random_state=0, class_weight=w) ### 
logreg_weighted.fit(original_Xtrain,original_ytrain) ###

# 예측값 
y_pred = logreg_weighted.predict(original_Xtest) ###

# 예측결과 확인하기
print('Logistic Regression ------ Weighted')
print(f'Accuracy: {accuracy_score(original_ytest,y_pred)}') ###

print('\n')
print(f'Confusion Matrix: \n{confusion_matrix(original_ytest, y_pred)}')###

print('\n')
print(f'Recall: {recall_score(original_ytest,y_pred)}') ###

```

    Logistic Regression ------ Weighted
    Accuracy: 0.9980337423851408
    
    
    Confusion Matrix: 
    [[56768    95]
     [   17    81]]
    
    
    Recall: 0.826530612244898



```python
# imblearn 패키지를 이용하여 예측 결과 확인하기
from imblearn.metrics import classification_report_imbalanced

label = ['non-fraud', 'fraud'] 
print(classification_report_imbalanced(original_ytest, y_pred, target_names=label))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      non-fraud       1.00      1.00      0.83      1.00      0.91      0.84     56863
          fraud       0.46      0.83      1.00      0.59      0.91      0.81        98
    
    avg / total       1.00      1.00      0.83      1.00      0.91      0.84     56961
    



```python
# 방법 2: Resampling 으로 재구축한 샘플을 이용하는 방법
```


```python
# 학습시킬 모델 로드하기
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from lightgbm import LGBMClassifier

classifiers = {
    "Logisitic Regression": LogisticRegression(),
    "K Nearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "LightGBM Classifier": LGBMClassifier()
}
```


```python
# 재구축한 샘플 데이터로 모델 피팅하기
logreg_resampled = LogisticRegression(random_state=0) ### 
logreg_resampled.fit(X_resampled, y_resampled) ###

# 예측값 구하기
y_pred = logreg_resampled.predict(original_Xtest)

print('Logistic Regression ------ Resampled Data')
print(f'Accuracy: {accuracy_score(original_ytest,y_pred)}') ###
print('\n')
print(f'Confusion Matrix: \n{confusion_matrix(original_ytest, y_pred)}') ### 
print('\n')
print(f'Recall: {recall_score(original_ytest,y_pred)}') ###
print('\n')
print(' ---- ')

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train,y_train, cv=5)
    print(classifier.__class__.__name__, ':', round(training_score.mean(), 2) * 100, '% accuracy')
```

    Logistic Regression ------ Resampled Data
    Accuracy: 0.9887291304576816
    
    
    Confusion Matrix: 
    [[56234   629]
     [   13    85]]
    
    
    Recall: 0.8673469387755102
    
    
     ---- 
    LogisticRegression : 94.0 % accuracy
    KNeighborsClassifier : 93.0 % accuracy
    SVC : 93.0 % accuracy
    DecisionTreeClassifier : 89.0 % accuracy
    RandomForestClassifier : 94.0 % accuracy
    GradientBoostingClassifier : 94.0 % accuracy
    LGBMClassifier : 94.0 % accuracy



```python
# imblearn 패키지를 이용하여 예측 결과 확인하기
from imblearn.metrics import classification_report_imbalanced

label = ['non-fraud', 'fraud'] 
print(classification_report_imbalanced(original_ytest, y_pred, target_names=label))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      non-fraud       1.00      0.99      0.87      0.99      0.93      0.87     56863
          fraud       0.12      0.87      0.99      0.21      0.93      0.85        98
    
    avg / total       1.00      0.99      0.87      0.99      0.93      0.87     56961
    


## Step 5. 요약

- 1) 불균형 분류 문제에 대한 이해: 사기탐지 데이터
- 2) 피처 변환 알고리즘의 이해: <b>PCA</b>, <b>t-SNE</b>, <b>SVD</b>
- 3) 과대적합 발생시 해결 방법 습득: 모델 파라미터 조정, 샘플 재구축
- 4) 리샘플링 알고리즘에 대한 이해: <b>Random Undersampling</b>, <b>Random Oversampling</b>, <b>SMOTE Oversampling</b> 등
- 5) 불균형 데이터를 이용한 분류 결과의 올바른 해석 방법 습득: `classification_report_imbalanced` 이용하기
