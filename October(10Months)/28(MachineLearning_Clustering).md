# 데이터 셋 로딩과 데이터 클린징

## 경로 변경


```python
import os
print(os.getcwd())
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/Clustering_Data')
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/Clustering_Data


## 데이터 불러오기


```python
import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

retail_df = pd.read_excel(io = './Online Retail.xlsx')
retail_df.head(3)
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 확인


```python
retail_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    541909 non-null  object        
     1   StockCode    541909 non-null  object        
     2   Description  540455 non-null  object        
     3   Quantity     541909 non-null  int64         
     4   InvoiceDate  541909 non-null  datetime64[ns]
     5   UnitPrice    541909 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      541909 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 33.1+ MB


## 변수 확인
- InvoiceNo : 송장번호
- StockCode : 상품(품목)코드
- Description : 상품(품목)명
- Quantitiy : 수량
- InvoiceDate : 송장 날짜
- UnitPrice : 단가
- CustomerID : 고객.ID
- Country : 국가

## 전처리


```python
# 필요없는 항목 제외
# 수량(Qunatity), 단가(UnitPrice)이 0인 경우는 필요 없으니 제외
retail_df = retail_df[retail_df['Quantity'] > 0]
retail_df = retail_df[retail_df['UnitPrice'] > 0]
retail_df = retail_df[retail_df['CustomerID'].notnull()] # 고객(CustomerID)가 없는 경우도 필요 없으니 제외
print(retail_df.shape)
retail_df.isnull().sum()
```

    (397884, 8)





    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
# 나라별 데이터 갯수
retail_df['Country'].value_counts()[:5]
```




    United Kingdom    354321
    Germany             9040
    France              8341
    EIRE                7236
    Spain               2484
    Name: Country, dtype: int64




```python
# 영국(UK; United Kindgom) 데이터만 선택
retail_df = retail_df[retail_df['Country'] == 'United Kingdom']
print(retail_df.shape)
```

    (354321, 8)



```python
# 변경된 데이터 확인
retail_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 354321 entries, 0 to 541893
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    354321 non-null  object        
     1   StockCode    354321 non-null  object        
     2   Description  354321 non-null  object        
     3   Quantity     354321 non-null  int64         
     4   InvoiceDate  354321 non-null  datetime64[ns]
     5   UnitPrice    354321 non-null  float64       
     6   CustomerID   354321 non-null  float64       
     7   Country      354321 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 24.3+ MB


## RFM 기반 데이터 가공


```python
# 고객이 구매한 금액 데이터(Scale_amount)생성
retail_df['sale_amount'] = retail_df['Quantity'] * retail_df['UnitPrice']
# 고개ID(CustomerID) 형태 변환
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)
```


```python
print(retail_df['CustomerID'].value_counts().head(5))

print(retail_df.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])
```

    17841    7847
    14096    5111
    12748    4595
    14606    2700
    15311    2379
    Name: CustomerID, dtype: int64
    CustomerID
    18102    259657.30
    17450    194550.79
    16446    168472.50
    17511     91062.38
    16029     81024.84
    Name: sale_amount, dtype: float64



```python
retail_df.head(3)
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>sale_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>15.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>22.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Invoice안에 다양한 SockCode가 존재할 수 있기 때문에 평균을 추출
retail_df.groupby(['InvoiceNo','StockCode'])['InvoiceNo'].count().mean()
```




    1.028702077315023




```python
# DataFrame의 groupby()의 multiple연산을 위해 agg()이용
# Recency는 IncvoiceDate 컬럼의 max()에서 데이터 가공
# Frequency는 InvoiceNo컬럼의 count(), Monetary value는 sale_amount 컬럼의 sum()
aggregations = {
    'InvoiceDate' : 'max',
    'InvoiceNo' : 'count',
    'sale_amount' : 'sum'
}

cust_df = retail_df.groupby('CustomerID').agg(aggregations)
# groupby된 결과 컬럼값을 Recency, Frequency, Monetary로 변경
cust_df = cust_df.rename(columns = {'InvoiceDate' : 'Recency',
                                    'InvoiceNo' : 'Frequency',
                                    'sale_amount' : 'Monetary'
                                   }
                        )
cust_df = cust_df.reset_index()
cust_df.head(3)
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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346</td>
      <td>2011-01-18 10:01:00</td>
      <td>1</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12747</td>
      <td>2011-12-07 14:34:00</td>
      <td>103</td>
      <td>4196.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12748</td>
      <td>2011-12-09 12:20:00</td>
      <td>4595</td>
      <td>33719.73</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 가장 최근 날짜 추출
cust_df['Recency'].max()
```




    Timestamp('2011-12-09 12:49:00')




```python
import datetime as dt

# 날짜 컬럼 숫자(D-Day)로 변경
cust_df['Recency'] = dt.datetime(2011, 12, 10) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days + 1)
print('cust_df 로우와 컬럼 건수는 ', cust_df.shape)
cust_df.head()
```

    cust_df 로우와 컬럼 건수는  (3920, 4)





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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346</td>
      <td>326</td>
      <td>1</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12747</td>
      <td>3</td>
      <td>103</td>
      <td>4196.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12748</td>
      <td>1</td>
      <td>4595</td>
      <td>33719.73</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12749</td>
      <td>4</td>
      <td>199</td>
      <td>4090.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12820</td>
      <td>4</td>
      <td>59</td>
      <td>942.34</td>
    </tr>
  </tbody>
</table>
</div>



## RFM 기반 고객 세그먼테이션


```python
fig, (ax1, ax2, ax3) = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 3)
ax1.set_title('Recency Histogram')
ax1.hist(cust_df['Recency'])

ax2.set_title('Frequency Histogram')
ax2.hist(cust_df['Frequency'])

ax3.set_title('Monetary Histogram')
ax3.hist(cust_df['Monetary'])

# VVIP독점으로 인하여 그래프가 극단적이다.
```




    (array([3.887e+03, 1.900e+01, 9.000e+00, 2.000e+00, 0.000e+00, 0.000e+00,
            1.000e+00, 1.000e+00, 0.000e+00, 1.000e+00]),
     array([3.75000000e+00, 2.59691050e+04, 5.19344600e+04, 7.78998150e+04,
            1.03865170e+05, 1.29830525e+05, 1.55795880e+05, 1.81761235e+05,
            2.07726590e+05, 2.33691945e+05, 2.59657300e+05]),
     <BarContainer object of 10 artists>)




    
![output_22_1](https://user-images.githubusercontent.com/87309905/198540848-054a2a00-0cc9-4883-a779-ce44866fe9a8.png)



```python
cust_df[['Recency','Frequency','Monetary']].describe()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3920.000000</td>
      <td>3920.000000</td>
      <td>3920.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>92.742092</td>
      <td>90.388010</td>
      <td>1864.385601</td>
    </tr>
    <tr>
      <th>std</th>
      <td>99.533485</td>
      <td>217.808385</td>
      <td>7482.817477</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>17.000000</td>
      <td>300.280000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51.000000</td>
      <td>41.000000</td>
      <td>652.280000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>143.000000</td>
      <td>99.250000</td>
      <td>1576.585000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>374.000000</td>
      <td>7847.000000</td>
      <td>259657.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X_features = cust_df[['Recency', 'Frequency', 'Monetary']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

kmeans = KMeans(n_clusters = 3, random_state = 0)
labels = kmeans.fit_predict(X_features_scaled)
cust_df['cluster_label'] = labels

print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled, labels)))
```

    실루엣 스코어는 : 0.592


=> 실루엣 스코어 : 얼마나 모여 있는가..


```python
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성 
def visualize_silhouette(cluster_lists, X_features):
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt 
    import matplotlib.cm as cm 
    import math

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함 
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화 
    for ind, n_cluster in enumerate(cluster_lists):
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0) 
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels) 
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                           'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values") 
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10]) 
        axs[ind].set_yticks([])

        # Clear the yaxis labels / ticks 
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i] 
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0] 
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster) 
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                    facecolor=color, edgecolor=color, alpha=0.7) 

            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
```


```python
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 클러스터링 결과를 시각화 
def visualize_kmeans_plot_multi(cluster_lists, X_features):
    from sklearn.cluster import KMeans 
    from sklearn.decomposition import PCA 
    import pandas as pd
    import numpy as np
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화 
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1','PCA2'])
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장.
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0) 
        cluster_labels = clusterer.fit_predict(pca_transformed) 
        dataframe['cluster']=cluster_labels
        
        unique_labels = np.unique(clusterer.labels_) 
        markers=['o', 's', '^', 'x', '*']
        
        # 클러스터링 결과값 별로 scatter plot 으로 시각화 
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster']==label] 
            if label == -1:
                cluster_legend = 'Noise' 
            else :
                cluster_legend = 'Cluster '+str(label) 
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\
                                edgecolor='k', marker=markers[label], label=cluster_legend) 
            
            axs[ind].set_title('Number of Cluster : '+ str(n_cluster))
            axs[ind].legend(loc='upper right')
    plt.show() 
```


```python
visualize_silhouette([2,3,4,5],X_features_scaled)

visualize_kmeans_plot_multi([2,3,4,5],X_features_scaled)

```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5104/1624528060.py:34: UserWarning: You passed a edgecolor/edgecolors ('k') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\
    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5104/1624528060.py:34: UserWarning: You passed a edgecolor/edgecolors ('k') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\



    
![output_28_1](https://user-images.githubusercontent.com/87309905/198540825-5f69e76d-048b-4937-9836-a0bdde073116.png)
  



![output_28_2](https://user-images.githubusercontent.com/87309905/198540799-597354eb-8fc9-4ec1-98b2-29971214a189.png)
   



```python
### Log 변환을 통해 데이터 변환
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Recency, Frequency, Monetary 컬럼에 np.log1p() 로 Log Transformation
cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])

# Log Transformation 데이터 StandardScaler 적용
X_features = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

kmeans = KMeans(n_clusters = 3, random_state = 0)
labels = kmeans.fit_predict(X_features_scaled)
cust_df['cluster_labels'] = labels

print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled, labels)))
```

    실루엣 스코어는 : 0.303



```python
visualize_silhouette([2,3,4,5], X_features_scaled)
visualize_kmeans_plot_multi([2,3,4,5], X_features_scaled)
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5104/1624528060.py:34: UserWarning: You passed a edgecolor/edgecolors ('k') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\
    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_5104/1624528060.py:34: UserWarning: You passed a edgecolor/edgecolors ('k') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\


![output_30_1](https://user-images.githubusercontent.com/87309905/198540761-5aede9d1-e1f2-4f18-9703-2b8922ffbb65.png)
    



![output_30_2](https://user-images.githubusercontent.com/87309905/198540747-ae1adcd8-b22d-4357-a94e-d01d8b29ce73.png)
  
