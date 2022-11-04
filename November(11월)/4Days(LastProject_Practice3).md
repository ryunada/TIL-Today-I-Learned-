# 작업 경로 설정


```python
import os
print(os.getcwd())
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project/home-credit-default-risk-Data')
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Last_Project/home-credit-default-risk-Data


### prev_application 데이터 세트 기반의 EDA와 Feature Engineering 수행 후 학습 모델 생성/평가

#### 라이브러리 및 데이터 세트 로딩. 이전 application 데이터의 FE 함수 복사


```python
import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
```


```python
app_train = pd.read_csv('./application_train.csv')
app_test = pd.read_csv('./application_test.csv')
```


```python
def get_apps_dataset():
    app_train = pd.read_csv('./application_train.csv')
    app_test = pd.read_csv('./application_test.csv')
    apps = pd.concat([app_train, app_test])
    
    return apps

apps = get_apps_dataset()
```

### 이전 application 데이터의 Feature Engineering 함수 복사


```python
def get_apps_processed(apps):
    
    # EXT_SOURCE_X FEATURE 가공
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps['APPS_EXT_SOURCE_STD'].fillna(apps['APPS_EXT_SOURCE_STD'].mean())
    
    # AMT_CREDIT 비율로 Feature 가공
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_CREDIT']
    apps['APPS_GOODS_CREDIT_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_CREDIT']
    
    # AMT_INCOME_TOTAL 비율로 Feature 가공
    apps['APPS_ANNUITY_INCOME_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT']/apps['AMT_INCOME_TOTAL']
    apps['APPS_GOODS_INCOME_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CNT_FAM_INCOME_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['CNT_FAM_MEMBERS']
    
    # DAYS_BIRTH, DAYS_EMPLOYED 비율로 Feature 가공
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED']/apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_EMPLOYED']
    apps['APPS_INCOME_BIRTH_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_BIRTH']
    apps['APPS_CAR_BIRTH_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_EMPLOYED_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_EMPLOYED']
    
    return apps
```

### previous 데이터 로딩


```python
prev = pd.read_csv('./previous_application.csv')
print(prev.shape, apps.shape)
```

    (1670214, 37) (356255, 122)


### application와 previous outer 조인하고 누락된 집합들 확인.


```python
# previous와 applications를 양쪽 OUTER 조인하여 안되는 대상 조사
# pandas merge()시 인자로 indicator = True 부여하면 어느 집합이 조인에서 누락되는지 알 수 있음.
prev_app_outer = prev.merge(apps['SK_ID_CURR'], on = 'SK_ID_CURR', how = 'outer', indicator = True) 
# indicator = True -> 행의 소스에 대한 정보와 함께 _merge 출력 ( both :관찰의 병합키가 존재, left_only , right_only )
prev_app_outer.head(10)
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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>FLAG_LAST_APPL_PER_CONTRACT</th>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <th>RATE_DOWN_PAYMENT</th>
      <th>RATE_INTEREST_PRIMARY</th>
      <th>RATE_INTEREST_PRIVILEGED</th>
      <th>NAME_CASH_LOAN_PURPOSE</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>DAYS_DECISION</th>
      <th>NAME_PAYMENT_TYPE</th>
      <th>CODE_REJECT_REASON</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_CLIENT_TYPE</th>
      <th>NAME_GOODS_CATEGORY</th>
      <th>NAME_PORTFOLIO</th>
      <th>NAME_PRODUCT_TYPE</th>
      <th>CHANNEL_TYPE</th>
      <th>SELLERPLACE_AREA</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495.0</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.182832</td>
      <td>0.867336</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-73.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>Mobile</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Country-wide</td>
      <td>35.0</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1696966.0</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>68258.655</td>
      <td>1800000.0</td>
      <td>1754721.0</td>
      <td>180000.0</td>
      <td>1800000.0</td>
      <td>SATURDAY</td>
      <td>18.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.101325</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Refused</td>
      <td>-472.0</td>
      <td>Cash through the bank</td>
      <td>SCO</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>Clothing and Accessories</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Regional / Local</td>
      <td>55.0</td>
      <td>Furniture</td>
      <td>36.0</td>
      <td>low_normal</td>
      <td>POS industry with interest</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2154916.0</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>12417.390</td>
      <td>108400.5</td>
      <td>119848.5</td>
      <td>0.0</td>
      <td>108400.5</td>
      <td>SUNDAY</td>
      <td>14.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-548.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>New</td>
      <td>Furniture</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Stone</td>
      <td>196.0</td>
      <td>Furniture</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS industry with interest</td>
      <td>365243.0</td>
      <td>-512.0</td>
      <td>-182.0</td>
      <td>-392.0</td>
      <td>-387.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2802425.0</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-164.0</td>
      <td>XNA</td>
      <td>XAP</td>
      <td>Unaccompanied</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Contact center</td>
      <td>-1.0</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1536272.0</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>21709.125</td>
      <td>450000.0</td>
      <td>512370.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>WEDNESDAY</td>
      <td>9.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-515.0</td>
      <td>XNA</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>AP+ (Cash loan)</td>
      <td>6.0</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_normal</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-485.0</td>
      <td>565.0</td>
      <td>-155.0</td>
      <td>-147.0</td>
      <td>1.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2068863.0</td>
      <td>108129</td>
      <td>Consumer loans</td>
      <td>4830.930</td>
      <td>47250.0</td>
      <td>23688.0</td>
      <td>24750.0</td>
      <td>47250.0</td>
      <td>THURSDAY</td>
      <td>11.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.556485</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-619.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Family</td>
      <td>Repeater</td>
      <td>Audio/Video</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Stone</td>
      <td>110.0</td>
      <td>Consumer electronics</td>
      <td>6.0</td>
      <td>high</td>
      <td>POS household with interest</td>
      <td>365243.0</td>
      <td>-588.0</td>
      <td>-438.0</td>
      <td>-588.0</td>
      <td>-580.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2551979.0</td>
      <td>108129</td>
      <td>Consumer loans</td>
      <td>6664.275</td>
      <td>71352.0</td>
      <td>71352.0</td>
      <td>0.0</td>
      <td>71352.0</td>
      <td>WEDNESDAY</td>
      <td>9.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-1208.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Unaccompanied</td>
      <td>New</td>
      <td>Consumer Electronics</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Stone</td>
      <td>108.0</td>
      <td>Furniture</td>
      <td>12.0</td>
      <td>low_normal</td>
      <td>POS industry with interest</td>
      <td>365243.0</td>
      <td>-1176.0</td>
      <td>-846.0</td>
      <td>-846.0</td>
      <td>-840.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2517198.0</td>
      <td>108129</td>
      <td>Revolving loans</td>
      <td>11250.000</td>
      <td>0.0</td>
      <td>225000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TUESDAY</td>
      <td>13.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-957.0</td>
      <td>XNA</td>
      <td>XAP</td>
      <td>NaN</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cards</td>
      <td>x-sell</td>
      <td>Contact center</td>
      <td>-1.0</td>
      <td>XNA</td>
      <td>0.0</td>
      <td>XNA</td>
      <td>Card X-Sell</td>
      <td>-713.0</td>
      <td>-673.0</td>
      <td>365243.0</td>
      <td>-461.0</td>
      <td>-61.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1760610.0</td>
      <td>108129</td>
      <td>Consumer loans</td>
      <td>8593.965</td>
      <td>33052.5</td>
      <td>33052.5</td>
      <td>0.0</td>
      <td>33052.5</td>
      <td>SUNDAY</td>
      <td>10.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XAP</td>
      <td>Approved</td>
      <td>-819.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Unaccompanied</td>
      <td>Repeater</td>
      <td>Computers</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Stone</td>
      <td>108.0</td>
      <td>Furniture</td>
      <td>4.0</td>
      <td>low_action</td>
      <td>POS industry with interest</td>
      <td>365243.0</td>
      <td>-783.0</td>
      <td>-693.0</td>
      <td>-753.0</td>
      <td>-748.0</td>
      <td>0.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2523466.0</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XNA</td>
      <td>Approved</td>
      <td>-301.0</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Spouse, partner</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Credit and cash offices</td>
      <td>-1.0</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
      <td>both</td>
    </tr>
  </tbody>
</table>
</div>




```python
prev_app_outer['_merge'].value_counts()
```




    both          1670214
    right_only      17398
    left_only           0
    Name: _merge, dtype: int64



### previous 컬럼 설명

|Table|컬럼명|컬럼 대분류|컬럼 중분류|컬럼 설명|
|------|--------|----|--------|-----------------|
|previous_application.csv|SK_ID_PREV |대출|고유ID|과거 대출 고유 ID|
|previous_application.csv|SK_ID_CURR|대출|고유ID|현재 대출 고유 ID|
|previous_application.csv|NAME_CONTRACT_TYPE|대출|대출 유형|대출 유형|
|previous_application.csv|AMT_ANNUITY|대출|대출 금액|월 대출 지급액|
|previous_application.csv|AMT_APPLICATION|대출|대출 금액|대출 신청 금액|
|previous_application.csv|AMT_CREDIT|대출|대출 금액|대출금액(허가)|
|previous_application.csv|AMT_DOWN_PAYMENT|대출|대출 금액|대출 시 납부한 선금액|
|previous_application.csv|AMT_GOODS_PRICE|대출|대출 금액|소비자 대출상품액|
|previous_application.csv|WEEKDAY_APPR_PROCESS_START|고객|행동|대출 신청 시작 요일|
|previous_application.csv|HOUR_APPR_PROCESS_START|고객|행동|대출 신청 시작 시간대|
|previous_application.csv|FLAG_LAST_APPL_PER_CONTRACT|고객|행동|이전 계약의 마지막 대출 신청 여부|
|previous_application.csv|NFLAG_LAST_APPL_IN_DAY|고객|행동|하루중 마지막 대출 신청 여부(하루에 여러 번 대출 신청했을 경우)|
|previous_application.csv|NFLAG_MICRO_CASH|대출|대출 유형|소액 대출 여부|
|previous_application.csv|RATE_DOWN_PAYMENT|대출|대출 금액|선금 비율(정규화됨)|
|previous_application.csv|RATE_INTEREST_PRIMARY|대출|대출 금액|이자율|
|previous_application.csv|RATE_INTEREST_PRIVILEGED|대출|대출 금액|이자율|
|previous_application.csv|NAME_CASH_LOAN_PURPOSE|대출|대출 유형|현금 대출 목적|
|previous_application.csv|NAME_CONTRACT_STATUS|대출|대출 상태|대출 상태(허가, 취소)|
|previous_application.csv|DAYS_DECISION|대출|대출 상태|과거 신청 대비 현재 신청 결정 기간|
|previous_application.csv|NAME_PAYMENT_TYPE|대출|대출 유형|과거 대출 신청의 납부 방법|
|previous_application.csv|CODE_REJECT_REASON|대출|대출 상태|과거 신청 거절 사유|
|previous_application.csv|NAME_TYPE_SUITE|고객|행동(추천)|동행 고객|
|previous_application.csv|NAME_CLIENT_TYPE|고객|행동|신규 고객 또는 기존 대출 고객 여부|
|previous_application.csv|NAME_GOODS_CATEGORY|대출|대출 유형|대출 상품 중분류 유형|
|previous_application.csv|NAME_PORTFOLIO|대출|대출 유형|현금대출/POS/CAR 대출 유형|
|previous_application.csv|NAME_PRODUCT_TYPE|채널|판매 유형|고객이 찾아온 대출인가, 영업 대출인가|
|previous_application.csv|CHANNEL_TYPE|채널|채널 유형|채널 유형|
|previous_application.csv|SELLERPLACE_AREA|채널|채널 유형|판매자 판매 지역|
|previous_application.csv|NAME_SELLER_INDUSTRY|채널|채널 유형|판매자 Industry|
|previous_application.csv|CNT_PAYMENT|대출|대출 금액|이전 대출 신청의 대출금액 관련 Term|
|previous_application.csv|NAME_YIELD_GROUP|대출|대출 금액|집단 금리 적용 유형|
|previous_application.csv|PRODUCT_COMBINATION|대출|대출 유형|이전 대출 결합 상품|
|previous_application.csv|DAYS_FIRST_DRAWING|대출|상태|신청날짜부터 최초 대출 지급까지의 일자|
|previous_application.csv|DAYS_FIRST_DUE|대출|상태|신청날짜부터 마감일까지의 일자|
|previous_application.csv|DAYS_LAST_DUE_1ST_VERSION|대출|상태|신청날짜부터 첫 만기일까지의 일자|
|previous_application.csv|DAYS_LAST_DUE|대출|상태|신청날짜부터 마지막 만기일까지의 일자|
|previous_application.csv|DAYS_TERMINATION|대출|상태|현 대출 신청일자 대비 대출 예상 종료 일자|
|previous_application.csv|NFLAG_INSURED_ON_APPROVAL|대출|상태|대출 신청 시 보험가입 요청여부|


### previous 컬럼과 Null 값 조사


```python
prev.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1670214 entries, 0 to 1670213
    Data columns (total 37 columns):
     #   Column                       Non-Null Count    Dtype  
    ---  ------                       --------------    -----  
     0   SK_ID_PREV                   1670214 non-null  int64  
     1   SK_ID_CURR                   1670214 non-null  int64  
     2   NAME_CONTRACT_TYPE           1670214 non-null  object 
     3   AMT_ANNUITY                  1297979 non-null  float64
     4   AMT_APPLICATION              1670214 non-null  float64
     5   AMT_CREDIT                   1670213 non-null  float64
     6   AMT_DOWN_PAYMENT             774370 non-null   float64
     7   AMT_GOODS_PRICE              1284699 non-null  float64
     8   WEEKDAY_APPR_PROCESS_START   1670214 non-null  object 
     9   HOUR_APPR_PROCESS_START      1670214 non-null  int64  
     10  FLAG_LAST_APPL_PER_CONTRACT  1670214 non-null  object 
     11  NFLAG_LAST_APPL_IN_DAY       1670214 non-null  int64  
     12  RATE_DOWN_PAYMENT            774370 non-null   float64
     13  RATE_INTEREST_PRIMARY        5951 non-null     float64
     14  RATE_INTEREST_PRIVILEGED     5951 non-null     float64
     15  NAME_CASH_LOAN_PURPOSE       1670214 non-null  object 
     16  NAME_CONTRACT_STATUS         1670214 non-null  object 
     17  DAYS_DECISION                1670214 non-null  int64  
     18  NAME_PAYMENT_TYPE            1670214 non-null  object 
     19  CODE_REJECT_REASON           1670214 non-null  object 
     20  NAME_TYPE_SUITE              849809 non-null   object 
     21  NAME_CLIENT_TYPE             1670214 non-null  object 
     22  NAME_GOODS_CATEGORY          1670214 non-null  object 
     23  NAME_PORTFOLIO               1670214 non-null  object 
     24  NAME_PRODUCT_TYPE            1670214 non-null  object 
     25  CHANNEL_TYPE                 1670214 non-null  object 
     26  SELLERPLACE_AREA             1670214 non-null  int64  
     27  NAME_SELLER_INDUSTRY         1670214 non-null  object 
     28  CNT_PAYMENT                  1297984 non-null  float64
     29  NAME_YIELD_GROUP             1670214 non-null  object 
     30  PRODUCT_COMBINATION          1669868 non-null  object 
     31  DAYS_FIRST_DRAWING           997149 non-null   float64
     32  DAYS_FIRST_DUE               997149 non-null   float64
     33  DAYS_LAST_DUE_1ST_VERSION    997149 non-null   float64
     34  DAYS_LAST_DUE                997149 non-null   float64
     35  DAYS_TERMINATION             997149 non-null   float64
     36  NFLAG_INSURED_ON_APPROVAL    997149 non-null   float64
    dtypes: float64(15), int64(6), object(16)
    memory usage: 471.5+ MB


-> 연속형 : 21개, object : 16개 

## 주요 컬럼 EDA 수행
### SK_ID_CURR평균 SK_ID_PREV 건수 구하기
- groupby로 편균 건수 구함.

- boxplot으로 시각화



```python
# 한 ID당 대출 횟수
# SK_ID_CURR로 groupby하여 평균 건수 구함.
prev.groupby('SK_ID_CURR')['SK_ID_CURR'].count().mean()
```




    4.928964135313716



-> 총 한사람 당 5번 정도 대출을 함.


```python
# box plot으로 시각화. 일부 데이터는 특정 SK_ID_CURR로 몇십개의 데이터가 잇음.
sns.boxplot(prev.groupby('SK_ID_CURR')['SK_ID_CURR'].count())
```

    /opt/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(





    <AxesSubplot:xlabel='SK_ID_CURR'>



![output_21_2](https://user-images.githubusercontent.com/87309905/199922191-f35d563a-2414-46ee-b57d-dfc0d175857e.png)

    
 


### 숫자형 피처들의 Histogram을 TARGET유형에 따라 비교
- application_train의 TARGET값을 가져오기 위해 prev와 app_train을 inner join후 TARGET 유형에 따라 비교
- 숫자형 컬럼명 필터링


```python
# TARGET값을 application에서 가져오기 위해 조인.
app_prev = prev.merge(app_train[['SK_ID_CURR', 'TARGET']], on = 'SK_ID_CURR', how = 'left')
app_prev.shape
```




    (1670214, 38)



기준 target 0, 1  
연속 -> violin plot  
object catplot  


```python
def show_hist_by_target(df, columns):
    cond_1 = (df['TARGET'] == 1)
    cond_0 = (df['TARGET'] == 0)
    
    for column in columns:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), squeeze=False)
        sns.violinplot(x='TARGET', y=column, data=df, ax=axs[0][0] )
        sns.distplot(df[cond_0][column], ax=axs[0][1], label='0', color='blue')
        sns.distplot(df[cond_1][column], ax=axs[0][1], label='1', color='red')   
```


```python
# 연속형인(즉, 형태가 object가 아닌 컬럼을 추출)
num_columns = app_prev.dtypes[app_prev.dtypes != 'object'].index.tolist()
num_columns = [column for column in num_columns if column not in['SK_ID_CURR', 'SK_ID_CURR','TARGET']]
print(num_columns)
```

    ['SK_ID_PREV', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL']



```python
show_hist_by_target(app_prev, num_columns)
```

    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)


![output_27_1](https://user-images.githubusercontent.com/87309905/199922039-134df589-13b0-4354-9e44-93c89c4a2c55.png)

![output_27_2](https://user-images.githubusercontent.com/87309905/199922042-1c42afcc-a344-4822-95f2-3608e894a70e.png)

![output_27_3](https://user-images.githubusercontent.com/87309905/199922045-17210cd3-0bab-4014-b800-3d1272cc48ed.png)

![output_27_4](https://user-images.githubusercontent.com/87309905/199922048-54e86545-27b0-4259-af8f-c2bc602bc5a5.png)

![output_27_5](https://user-images.githubusercontent.com/87309905/199922051-6fe8dbc7-462c-44e4-befb-da130ec8c874.png)

![output_27_6](https://user-images.githubusercontent.com/87309905/199922052-9a84ba00-17d4-4f81-8b4d-0d48c94406ac.png)

![output_27_7](https://user-images.githubusercontent.com/87309905/199922054-4b2248d8-0b21-4e8e-a4df-e5b6e301aff8.png)

![output_27_8](https://user-images.githubusercontent.com/87309905/199922056-8156310b-5b28-4a0b-9071-b3d318b11f70.png)

![output_27_9](https://user-images.githubusercontent.com/87309905/199922058-c04c9a02-f4cd-456c-ac1b-d79599537804.png)

![output_27_10](https://user-images.githubusercontent.com/87309905/199922061-4a727030-771f-4934-8ea0-4642e2fa09cf.png)

![output_27_11](https://user-images.githubusercontent.com/87309905/199922064-4e7537c9-faf2-42ae-9348-7619e7ef7a5d.png)

![output_27_12](https://user-images.githubusercontent.com/87309905/199922066-39d2be3a-9161-4779-b59f-6b811a32613c.png)

![output_27_13](https://user-images.githubusercontent.com/87309905/199922069-81c3f737-580e-4bad-98ba-49d1716bb94e.png)

![output_27_14](https://user-images.githubusercontent.com/87309905/199922074-4da16efd-b23f-4da8-8ff3-ba3498efe77c.png)

![output_27_15](https://user-images.githubusercontent.com/87309905/199922080-473d1df9-7516-4ed4-86c0-1865e9d7b1b5.png)

![output_27_16](https://user-images.githubusercontent.com/87309905/199922082-f283f9c0-4cf3-4abe-819e-d939b7589f0f.png)

![output_27_17](https://user-images.githubusercontent.com/87309905/199922086-004a5f52-3f4b-4c1b-9549-9d0b3e9e1187.png)

![output_27_18](https://user-images.githubusercontent.com/87309905/199922088-b0cf660a-0f0a-439d-8d1f-02c30c258ea7.png)

![output_27_19](https://user-images.githubusercontent.com/87309905/199922091-3cd4e63a-895b-4171-a539-e8bf90645cda.png)

![output_27_20](https://user-images.githubusercontent.com/87309905/199922094-ff50ea06-5e04-4392-96a6-f7902250a49c.png)



* AMT_ANNUITY, AMT_CREDIT, AMT_APPLICATION, AMT_GOODS_CREDIT는 TARGET=1일 경우에 소액 비율이 약간 높음(큰 차이는 아님)
* RATE_DOWN_PAYMENT는 큰 차이 없음. 
* RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED 는 NULL값이 매우 많아서 판단 어려움
* DAYS_DECISION은 TARGET=1일 때 0에 가까운(최근일)값이 약간 더 많음. 
* DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE, DAYS_TERMINATION은 365243 값이 매우 많음. 

### Category 피처들의 Histogram을 TARGET유형에 따라 비교


```python
object_columns = app_prev.dtypes[app_prev.dtypes == 'object'].index.tolist()
object_columns
```




    ['NAME_CONTRACT_TYPE',
     'WEEKDAY_APPR_PROCESS_START',
     'FLAG_LAST_APPL_PER_CONTRACT',
     'NAME_CASH_LOAN_PURPOSE',
     'NAME_CONTRACT_STATUS',
     'NAME_PAYMENT_TYPE',
     'CODE_REJECT_REASON',
     'NAME_TYPE_SUITE',
     'NAME_CLIENT_TYPE',
     'NAME_GOODS_CATEGORY',
     'NAME_PORTFOLIO',
     'NAME_PRODUCT_TYPE',
     'CHANNEL_TYPE',
     'NAME_SELLER_INDUSTRY',
     'NAME_YIELD_GROUP',
     'PRODUCT_COMBINATION']



### Categorical Plots
-> 카테고리 플랏은 색이 고정이됨
1) barplot  
2) countplot  
3) boxplot  
4) violineplot  
5) stripplot  
6) swarmplot  
7) boxenplot  
8) pointplot  
9) catplot  


```python
def show_category_by_target(df, columns):
    for column in columns:
        chart = sns.catplot(x=column, col="TARGET", data=df, kind="count")
        chart.set_xticklabels(rotation=65)
        
show_category_by_target(app_prev, object_columns)
```


![output_32_0](https://user-images.githubusercontent.com/87309905/199921884-89ec8ccc-b8cd-4a65-8506-423fde128608.png)

![output_32_1](https://user-images.githubusercontent.com/87309905/199921891-e647eaf7-0a25-4b70-b8c7-a84023d96371.png)

![output_32_2](https://user-images.githubusercontent.com/87309905/199921895-626f844a-68ab-4a54-86b8-069b36475d21.png)

![output_32_3](https://user-images.githubusercontent.com/87309905/199921897-2cae5042-59d9-45d1-b0a4-c26027613a6b.png)

![output_32_4](https://user-images.githubusercontent.com/87309905/199921902-e4660687-0f4e-4495-8441-3789458be95a.png)

![output_32_5](https://user-images.githubusercontent.com/87309905/199921905-571eec9b-ac0f-4762-8af3-b625448ac01a.png)

![output_32_6](https://user-images.githubusercontent.com/87309905/199921907-0706f3e5-a6b6-4bb4-9532-dc8f47565f0f.png)

![output_32_7](https://user-images.githubusercontent.com/87309905/199921908-e498471b-049d-4ee4-a322-97d3f7341676.png)

![output_32_8](https://user-images.githubusercontent.com/87309905/199921910-fa063131-8771-4bed-bdcc-dd1778a5fe58.png)

![output_32_9](https://user-images.githubusercontent.com/87309905/199921912-42212aba-fc44-4637-9903-5207f6eaa1b0.png)

![output_32_10](https://user-images.githubusercontent.com/87309905/199921915-03db72fe-640d-45e5-b382-9b99e9eea827.png)

![output_32_11](https://user-images.githubusercontent.com/87309905/199921918-09fafbec-b238-4708-b0fc-6c50631f4069.png)

![output_32_12](https://user-images.githubusercontent.com/87309905/199921920-f31980cb-108f-4947-8c84-500f73770a17.png)

![output_32_13](https://user-images.githubusercontent.com/87309905/199921923-961be35c-d10b-4a75-b56b-8e7ebe009e15.png)

![output_32_14](https://user-images.githubusercontent.com/87309905/199921824-1c6cac67-8405-4dee-8306-b8f6fc09bbde.png)

![output_32_15](https://user-images.githubusercontent.com/87309905/199921826-237ddfb2-bf1f-4b51-b301-f49b7ce69e24.png)




* NAME_CONTRACT_TYPE은 TARGET=1일때 CASH_LOAN의 비중이 약간 높음
* NAME_CONTRACT_STATUS(대출허가상태)는 TARGET=1일때 상대적으로 TARGET=0 대비 (당연히) Refused의 비율이 높음. 
* NAME_PAYMENT_TYPE(대출납부방법)는 TARGET=1일때 상대적으로 TARGET=0 대비 XNA의 비율이 약간 높음.

### prev 데이터 세트 feature engineering 수행. 
#### SQL 대비  Pandas groupby 사용 로직 비교

##### SQL로 SK_ID_CURR별 건수, 평균 AMT_CREDIT, 최대 AMT_CREDIT, 최소 AMT_CREDIT 구하기


```python
# select sk_id_curr, count(*), avg(amt_credit) , max(amt_credit), min(amt_credit) from previous group by sk_id_curr
```

##### pandas groupby 단일 aggregation 함수 사용
* groupby SK_ID_CURR
* SK_ID_CURR별 건수, AMT_CREDIT에 대한 평균, 최대 값


```python
prev.groupby('SK_ID_CURR')
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fedb73995b0>




```python
# DataFrameGrouopby 생성.
prev_group = prev.groupby('SK_ID_CURR')

# DataFrameGroupby 객체에 aggregation 함수 수행 결과를 저장한 DataFrame 생성 및 aggregation값 저장
prev_agg = pd.DataFrame()
prev_agg['CNT'] = prev_group['SK_ID_CURR'].count()
prev_agg['AVG_CREDIT'] = prev_group['AMT_CREDIT'].mean()
prev_agg['MAX_CREDIT'] = prev_group['AMT_CREDIT'].max()

# groupby 컬럼값이 DataFrame의 Index가 됨. 컬럼으로 변환하려면 reset_index()로 변환 필요.
prev_agg.head(10)
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
      <th>CNT</th>
      <th>AVG_CREDIT</th>
      <th>MAX_CREDIT</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>1</td>
      <td>23787.000000</td>
      <td>23787.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1</td>
      <td>179055.000000</td>
      <td>179055.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>3</td>
      <td>484191.000000</td>
      <td>1035882.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>1</td>
      <td>20106.000000</td>
      <td>20106.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20076.750000</td>
      <td>40153.5</td>
    </tr>
    <tr>
      <th>100006</th>
      <td>9</td>
      <td>291695.500000</td>
      <td>906615.0</td>
    </tr>
    <tr>
      <th>100007</th>
      <td>6</td>
      <td>166638.750000</td>
      <td>284400.0</td>
    </tr>
    <tr>
      <th>100008</th>
      <td>5</td>
      <td>162767.700000</td>
      <td>501975.0</td>
    </tr>
    <tr>
      <th>100009</th>
      <td>7</td>
      <td>70137.642857</td>
      <td>98239.5</td>
    </tr>
    <tr>
      <th>100010</th>
      <td>1</td>
      <td>260811.000000</td>
      <td>260811.0</td>
    </tr>
  </tbody>
</table>
</div>



### groupby agg()함수를 이용하여 여러개의 aggregation함수 적용


```python
prev_group = prev.groupby('SK_ID_CURR')

# DataFrameGroupby의 agg() 함수를 이용하여 여러개의 aggregation 함수 적용
prev_agg1 = prev_group['AMT_CREDIT'].agg(['mean','max','sum'])
prev_agg2 = prev_group['AMT_ANNUITY'].agg(['mean','max','sum'])
# merge를 이용하여 두개의 DataFrame결합.
prev_agg = prev_agg1.merge(prev_agg2, on = 'SK_ID_CURR', how = 'inner')
prev_agg.head(10)
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
      <th>mean_x</th>
      <th>max_x</th>
      <th>sum_x</th>
      <th>mean_y</th>
      <th>max_y</th>
      <th>sum_y</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
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
      <th>100001</th>
      <td>23787.000000</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000000</td>
      <td>3951.000</td>
      <td>3951.000</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>179055.000000</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775000</td>
      <td>9251.775</td>
      <td>9251.775</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>484191.000000</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990000</td>
      <td>98356.995</td>
      <td>169661.970</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>20106.000000</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250000</td>
      <td>5357.250</td>
      <td>5357.250</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>20076.750000</td>
      <td>40153.5</td>
      <td>40153.5</td>
      <td>4813.200000</td>
      <td>4813.200</td>
      <td>4813.200</td>
    </tr>
    <tr>
      <th>100006</th>
      <td>291695.500000</td>
      <td>906615.0</td>
      <td>2625259.5</td>
      <td>23651.175000</td>
      <td>39954.510</td>
      <td>141907.050</td>
    </tr>
    <tr>
      <th>100007</th>
      <td>166638.750000</td>
      <td>284400.0</td>
      <td>999832.5</td>
      <td>12278.805000</td>
      <td>22678.785</td>
      <td>73672.830</td>
    </tr>
    <tr>
      <th>100008</th>
      <td>162767.700000</td>
      <td>501975.0</td>
      <td>813838.5</td>
      <td>15839.696250</td>
      <td>25309.575</td>
      <td>63358.785</td>
    </tr>
    <tr>
      <th>100009</th>
      <td>70137.642857</td>
      <td>98239.5</td>
      <td>490963.5</td>
      <td>10051.412143</td>
      <td>17341.605</td>
      <td>70359.885</td>
    </tr>
    <tr>
      <th>100010</th>
      <td>260811.000000</td>
      <td>260811.0</td>
      <td>260811.0</td>
      <td>27463.410000</td>
      <td>27463.410</td>
      <td>27463.410</td>
    </tr>
  </tbody>
</table>
</div>



agg()에 dictionary를 이용하여 groupby 적용


```python
agg_dict = {
    'SK_ID_CURR' : ['count'],
    'AMT_CREDIT' : ['mean','max','sum'],
    'AMT_ANNUITY' : ['mean','max','sum'],
    'AMT_APPLICATION' : ['mean','max','sum'],
    'AMT_DOWN_PAYMENT' : ['mean','max','sum'],
    'AMT_GOODS_PRICE' : ['mean','max','sum']
}

prev_group = prev.groupby('SK_ID_CURR')
prev_amt_agg = prev_group.agg(agg_dict)
prev_amt_agg.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>SK_ID_CURR</th>
      <th colspan="3" halign="left">AMT_CREDIT</th>
      <th colspan="3" halign="left">AMT_ANNUITY</th>
      <th colspan="3" halign="left">AMT_APPLICATION</th>
      <th colspan="3" halign="left">AMT_DOWN_PAYMENT</th>
      <th colspan="3" halign="left">AMT_GOODS_PRICE</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
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
      <th>100001</th>
      <td>1</td>
      <td>23787.00</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>24835.5</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>3</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>169661.970</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>3442.5</td>
      <td>6885.0</td>
      <td>6885.0</td>
      <td>435436.5</td>
      <td>900000.0</td>
      <td>1306309.5</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>1</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20076.75</td>
      <td>40153.5</td>
      <td>40153.5</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>44617.5</td>
    </tr>
  </tbody>
</table>
</div>



agg()에 dictionary를 이용하여 groupby 적용

##### grouby agg 로 만들어진  Multi index 컬럼 변경. 
* MultiIndex로 되어 있는 컬럼명 확인
* MultiIndex 컬럼명을 _로 연결하여 컬럼명 변경. 


```python
prev_amt_agg.columns
```




    MultiIndex([(      'SK_ID_CURR', 'count'),
                (      'AMT_CREDIT',  'mean'),
                (      'AMT_CREDIT',   'max'),
                (      'AMT_CREDIT',   'sum'),
                (     'AMT_ANNUITY',  'mean'),
                (     'AMT_ANNUITY',   'max'),
                (     'AMT_ANNUITY',   'sum'),
                ( 'AMT_APPLICATION',  'mean'),
                ( 'AMT_APPLICATION',   'max'),
                ( 'AMT_APPLICATION',   'sum'),
                ('AMT_DOWN_PAYMENT',  'mean'),
                ('AMT_DOWN_PAYMENT',   'max'),
                ('AMT_DOWN_PAYMENT',   'sum'),
                ( 'AMT_GOODS_PRICE',  'mean'),
                ( 'AMT_GOODS_PRICE',   'max'),
                ( 'AMT_GOODS_PRICE',   'sum')],
               )




```python
[column[0] + column[1] for column in prev_amt_agg.columns]
```




    ['SK_ID_CURRcount',
     'AMT_CREDITmean',
     'AMT_CREDITmax',
     'AMT_CREDITsum',
     'AMT_ANNUITYmean',
     'AMT_ANNUITYmax',
     'AMT_ANNUITYsum',
     'AMT_APPLICATIONmean',
     'AMT_APPLICATIONmax',
     'AMT_APPLICATIONsum',
     'AMT_DOWN_PAYMENTmean',
     'AMT_DOWN_PAYMENTmax',
     'AMT_DOWN_PAYMENTsum',
     'AMT_GOODS_PRICEmean',
     'AMT_GOODS_PRICEmax',
     'AMT_GOODS_PRICEsum']




```python
('_').join(['test01','test_02'])
```




    'test01_test_02'




```python
# mulit index 컬럼을 '_'로 연결하여 컬럼명 변경
prev_amt_agg.columns = ['PREV_' +'_'.join(x).upper() for x in prev_amt_agg.columns.ravel()] # ravel : 여러 행렬의 함수를 하나의 행 or 열로 변환해주는 함수
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_25104/1520059396.py:2: FutureWarning: Index.ravel returning ndarray is deprecated; in a future version this will return a view on self.
      prev_amt_agg.columns = ['PREV_' +'_'.join(x).upper() for x in prev_amt_agg.columns.ravel()] # ravel : 여러 행렬의 함수를 하나의 행 or 열로 변환해주는 함수



```python
prev_amt_agg.head()
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
      <th>PREV_SK_ID_CURR_COUNT</th>
      <th>PREV_AMT_CREDIT_MEAN</th>
      <th>PREV_AMT_CREDIT_MAX</th>
      <th>PREV_AMT_CREDIT_SUM</th>
      <th>PREV_AMT_ANNUITY_MEAN</th>
      <th>PREV_AMT_ANNUITY_MAX</th>
      <th>PREV_AMT_ANNUITY_SUM</th>
      <th>PREV_AMT_APPLICATION_MEAN</th>
      <th>PREV_AMT_APPLICATION_MAX</th>
      <th>PREV_AMT_APPLICATION_SUM</th>
      <th>PREV_AMT_DOWN_PAYMENT_MEAN</th>
      <th>PREV_AMT_DOWN_PAYMENT_MAX</th>
      <th>PREV_AMT_DOWN_PAYMENT_SUM</th>
      <th>PREV_AMT_GOODS_PRICE_MEAN</th>
      <th>PREV_AMT_GOODS_PRICE_MAX</th>
      <th>PREV_AMT_GOODS_PRICE_SUM</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
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
      <th>100001</th>
      <td>1</td>
      <td>23787.00</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>24835.5</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>3</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>169661.970</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>3442.5</td>
      <td>6885.0</td>
      <td>6885.0</td>
      <td>435436.5</td>
      <td>900000.0</td>
      <td>1306309.5</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>1</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20076.75</td>
      <td>40153.5</td>
      <td>40153.5</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>44617.5</td>
    </tr>
  </tbody>
</table>
</div>



##### prev 피처 가공. 대출 신청액 대비 다른 금액 차이 및 비율 생성. 


```python
# 대출 신청 금액과 실제 대출액 / 대출 상품금액 차이 및 비율
prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']
prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY'] / prev['AMT_APPLICATION']
prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_APPLICATION']
```

### DAYS_XXX 피처의 365243을 NULL로 변환하고, 첫번째 만기일과 마지막 만기일까지의 기간 가공


```python
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace = True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace = True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace = True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace = True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace = True)

# 첫 번째 만기일과 마지막 만기일까지의 기간
prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
```

### 기존 이자율 관련 컬럼이 null이 많아서 새롭게 간단한 이자율을 대출 금액과 대출 금액 납부 횟수를 기반으로 계산


```python
# 매월 납부 금액과 납부 횟수 곱해서 전체 납부 금액을 구함
all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']

# 전체 납부 금액 대비 AMT_CREDIT 비율을 구하기 여기에 다시 납부횟수로 나누어서 이자율 계산.
prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']
```


```python
prev.iloc[:,-7:].head(10)
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
      <th>PREV_CREDIT_DIFF</th>
      <th>PREV_GOODS_DIFF</th>
      <th>PREV_CREDIT_APPL_RATIO</th>
      <th>PREV_ANNUITY_APPL_RATIO</th>
      <th>PREV_GOODS_APPL_RATIO</th>
      <th>PREV_DAYS_LAST_DUE_DIFF</th>
      <th>PREV_INTERESTS_RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.100929</td>
      <td>1.0</td>
      <td>342.0</td>
      <td>0.017596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-72171.0</td>
      <td>0.0</td>
      <td>1.118800</td>
      <td>0.041463</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.009282</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-23944.5</td>
      <td>0.0</td>
      <td>1.212840</td>
      <td>0.133873</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.027047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-20790.0</td>
      <td>0.0</td>
      <td>1.046200</td>
      <td>0.104536</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0.016587</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-66555.0</td>
      <td>0.0</td>
      <td>1.197200</td>
      <td>0.094591</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.037343</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-25573.5</td>
      <td>0.0</td>
      <td>1.081186</td>
      <td>0.075251</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.014044</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



##### 기존 대출 금액, 대출 상태 관련 피처들과 이들을 가공하여 만들어진 새로운 컬럼들로 aggregation 수행.


```python

# 새롭게 생성된 대출 신청액 대비 다른 금액 차이 및 비율로 aggregation 수행. 
agg_dict = {
    # 기존 컬럼.
    'SK_ID_CURR':['count'],
    'AMT_CREDIT':['mean', 'max', 'sum'], 
    'AMT_ANNUITY':['mean', 'max', 'sum'], 
    'AMT_APPLICATION':['mean', 'max', 'sum'], 
    'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
    'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
    # 가공 컬럼
    'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
    'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
    'PREV_GOODS_DIFF':['mean', 'max', 'sum'], 
    'PREV_GOODS_APPL_RATIO':['mean', 'max'],
    'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'], 
    'PREV_INTERESTS_RATE':['mean', 'max']

}
prev_group = prev.groupby('SK_ID_CURR')
prev_amt_agg = prev_group.agg(agg_dict)

# multi index 컬럼을 '_'로 연결하여 컬럼명 변경
prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_25104/3738128054.py:26: FutureWarning: Index.ravel returning ndarray is deprecated; in a future version this will return a view on self.
      prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]



```python
prev_amt_agg.shape
```




    (338857, 39)




```python
prev_amt_agg.head()
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
      <th>PREV_SK_ID_CURR_COUNT</th>
      <th>PREV_AMT_CREDIT_MEAN</th>
      <th>PREV_AMT_CREDIT_MAX</th>
      <th>PREV_AMT_CREDIT_SUM</th>
      <th>PREV_AMT_ANNUITY_MEAN</th>
      <th>PREV_AMT_ANNUITY_MAX</th>
      <th>PREV_AMT_ANNUITY_SUM</th>
      <th>PREV_AMT_APPLICATION_MEAN</th>
      <th>PREV_AMT_APPLICATION_MAX</th>
      <th>PREV_AMT_APPLICATION_SUM</th>
      <th>PREV_AMT_DOWN_PAYMENT_MEAN</th>
      <th>PREV_AMT_DOWN_PAYMENT_MAX</th>
      <th>PREV_AMT_DOWN_PAYMENT_SUM</th>
      <th>PREV_AMT_GOODS_PRICE_MEAN</th>
      <th>PREV_AMT_GOODS_PRICE_MAX</th>
      <th>PREV_AMT_GOODS_PRICE_SUM</th>
      <th>PREV_RATE_DOWN_PAYMENT_MIN</th>
      <th>PREV_RATE_DOWN_PAYMENT_MAX</th>
      <th>PREV_RATE_DOWN_PAYMENT_MEAN</th>
      <th>PREV_DAYS_DECISION_MIN</th>
      <th>PREV_DAYS_DECISION_MAX</th>
      <th>PREV_DAYS_DECISION_MEAN</th>
      <th>PREV_CNT_PAYMENT_MEAN</th>
      <th>PREV_CNT_PAYMENT_SUM</th>
      <th>PREV_PREV_CREDIT_DIFF_MEAN</th>
      <th>PREV_PREV_CREDIT_DIFF_MAX</th>
      <th>PREV_PREV_CREDIT_DIFF_SUM</th>
      <th>PREV_PREV_CREDIT_APPL_RATIO_MEAN</th>
      <th>PREV_PREV_CREDIT_APPL_RATIO_MAX</th>
      <th>PREV_PREV_GOODS_DIFF_MEAN</th>
      <th>PREV_PREV_GOODS_DIFF_MAX</th>
      <th>PREV_PREV_GOODS_DIFF_SUM</th>
      <th>PREV_PREV_GOODS_APPL_RATIO_MEAN</th>
      <th>PREV_PREV_GOODS_APPL_RATIO_MAX</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_MEAN</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_MAX</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_SUM</th>
      <th>PREV_PREV_INTERESTS_RATE_MEAN</th>
      <th>PREV_PREV_INTERESTS_RATE_MAX</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
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
      <th>100001</th>
      <td>1</td>
      <td>23787.00</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>0.104326</td>
      <td>0.104326</td>
      <td>0.104326</td>
      <td>-1740</td>
      <td>-1740</td>
      <td>-1740.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>1048.5</td>
      <td>1048.5</td>
      <td>1048.5</td>
      <td>0.957782</td>
      <td>0.957782</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>0.041099</td>
      <td>0.041099</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-606</td>
      <td>-606</td>
      <td>-606.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>0.010003</td>
      <td>0.010003</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>3</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>169661.970</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>3442.5</td>
      <td>6885.0</td>
      <td>6885.0</td>
      <td>435436.5</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>0.000000</td>
      <td>0.100061</td>
      <td>0.050030</td>
      <td>-2341</td>
      <td>-746</td>
      <td>-1305.0</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>-48754.5</td>
      <td>756.0</td>
      <td>-146263.5</td>
      <td>1.057664</td>
      <td>1.150980</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>0.015272</td>
      <td>0.018533</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>1</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>0.212008</td>
      <td>0.212008</td>
      <td>0.212008</td>
      <td>-815</td>
      <td>-815</td>
      <td>-815.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4176.0</td>
      <td>4176.0</td>
      <td>4176.0</td>
      <td>0.828021</td>
      <td>0.828021</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>0.016450</td>
      <td>0.016450</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20076.75</td>
      <td>40153.5</td>
      <td>40153.5</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>0.108964</td>
      <td>0.108964</td>
      <td>0.108964</td>
      <td>-757</td>
      <td>-315</td>
      <td>-536.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>2232.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>0.899950</td>
      <td>0.899950</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>0.036537</td>
      <td>0.036537</td>
    </tr>
  </tbody>
</table>
</div>



#### SK_ID_CURR별로 NAME_CONTRACT_STATUS가 Refused 일 경우의 건수 및 과거 대출건 대비 비율

#### Group by 기준  컬럼 기반에서 다른 컬럼들의 기준에 따라 세분화된 aggregation 수행. 


```python
prev['NAME_CONTRACT_STATUS'].value_counts()
```




    Approved        1036781
    Canceled         316319
    Refused          290678
    Unused offer      26436
    Name: NAME_CONTRACT_STATUS, dtype: int64



##### SQL Group by Case when 과 pandas의 차이


```python
# SK_ID_CURR레벨로 groupby 된 count와  name_contract_status가 Refused일 때의 count 
'''select sk_id_curr, cnt_refused/cnt
from
(
    select sk_id_curr, count(*) cnt, count(case when name_contract_status == 'Refused' end) cnt_refused
    from previous group by sk_id_curr
) 
'''
```




    "select sk_id_curr, cnt_refused/cnt\nfrom\n(\n    select sk_id_curr, count(*) cnt, count(case when name_contract_status == 'Refused' end) cnt_refused\n    from previous group by sk_id_curr\n) \n"



##### Pandas는 원 DataFrame 에 groupby 적용된 DataFrame 과 세부기준으로 filtering 된 DataFrame에 groupby 적용된 DataFrame 을 조인하여 생성. 
* NAME_CONTRACT_STATUS == 'Refused' 세부 기준으로 filtering 및 filtering 된 DataFrame에 groupby 적용 
* groupby 완료 후 기존 prev_amt_agg와 조인
* 효율적인 오류 방지를 위해서 groupby 시 적용후 groupby key값을 DataFrame의 Index가 아닌 일반 컬럼으로 변경.


```python
# NAME_CONRACT_STATUE == 'Refused' 세부 기준으로 filtering
cond_refused = (prev['NAME_CONTRACT_STATUS'] == 'Refused')
prev_refused = prev[cond_refused]
prev_refused.shape, prev.shape
```




    ((290678, 44), (1670214, 44))




```python
# NAME_CONTRACT_STATUS == 'Refused' 세부 기준으로 filtering 된 DataFrame에 groupby 적용
prev_refused_agg = prev_refused.groupby(['SK_ID_CURR'])['SK_ID_CURR'].count()
print(prev_amt_agg.shape, prev_refused_agg.shape)
prev_refused_agg.head(10)
```

    (338857, 39) (118277,)





    SK_ID_CURR
    100006     1
    100011     1
    100027     1
    100030    10
    100035     8
    100036     3
    100037     2
    100043     5
    100046     1
    100047     2
    Name: SK_ID_CURR, dtype: int64




```python
type(prev_refused.groupby(['SK_ID_CURR'])['SK_ID_CURR'].count())
```




    pandas.core.series.Series




```python
# prev_refused_agg은 Series객체이고 Index는 SK_ID_CURR, prev_amt_agg은 DataFrame, Index는 SK_ID_CURR. 하지만 JOIN되지 
# Series와 DataFrame 조인 시 Series를 DataFrame으로 내부 변환하는데 조인 시 Index명과 컬럼명이 서로 충돌하여 오류. 
prev_amt_refused_agg = prev_amt_agg.merge(prev_refused_agg, on='SK_ID_CURR', how='left')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [40], in <cell line: 3>()
          1 # prev_refused_agg은 Series객체이고 Index는 SK_ID_CURR, prev_amt_agg은 DataFrame, Index는 SK_ID_CURR. 하지만 JOIN되지 
          2 # Series와 DataFrame 조인 시 Series를 DataFrame으로 내부 변환하는데 조인 시 Index명과 컬럼명이 서로 충돌하여 오류. 
    ----> 3 prev_amt_refused_agg = prev_amt_agg.merge(prev_refused_agg, on='SK_ID_CURR', how='left')


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py:9345, in DataFrame.merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
       9326 @Substitution("")
       9327 @Appender(_merge_doc, indents=2)
       9328 def merge(
       (...)
       9341     validate: str | None = None,
       9342 ) -> DataFrame:
       9343     from pandas.core.reshape.merge import merge
    -> 9345     return merge(
       9346         self,
       9347         right,
       9348         how=how,
       9349         on=on,
       9350         left_on=left_on,
       9351         right_on=right_on,
       9352         left_index=left_index,
       9353         right_index=right_index,
       9354         sort=sort,
       9355         suffixes=suffixes,
       9356         copy=copy,
       9357         indicator=indicator,
       9358         validate=validate,
       9359     )


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/reshape/merge.py:107, in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
         90 @Substitution("\nleft : DataFrame or named Series")
         91 @Appender(_merge_doc, indents=0)
         92 def merge(
       (...)
        105     validate: str | None = None,
        106 ) -> DataFrame:
    --> 107     op = _MergeOperation(
        108         left,
        109         right,
        110         how=how,
        111         on=on,
        112         left_on=left_on,
        113         right_on=right_on,
        114         left_index=left_index,
        115         right_index=right_index,
        116         sort=sort,
        117         suffixes=suffixes,
        118         copy=copy,
        119         indicator=indicator,
        120         validate=validate,
        121     )
        122     return op.get_result()


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/reshape/merge.py:700, in _MergeOperation.__init__(self, left, right, how, on, left_on, right_on, axis, left_index, right_index, sort, suffixes, copy, indicator, validate)
        693 self._cross = cross_col
        695 # note this function has side effects
        696 (
        697     self.left_join_keys,
        698     self.right_join_keys,
        699     self.join_names,
    --> 700 ) = self._get_merge_keys()
        702 # validate the merge keys dtypes. We may need to coerce
        703 # to avoid incompatible dtypes
        704 self._maybe_coerce_merge_keys()


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/reshape/merge.py:1097, in _MergeOperation._get_merge_keys(self)
       1095 if not is_rkey(rk):
       1096     if rk is not None:
    -> 1097         right_keys.append(right._get_label_or_level_values(rk))
       1098     else:
       1099         # work-around for merge_asof(right_index=True)
       1100         right_keys.append(right.index)


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py:1835, in NDFrame._get_label_or_level_values(self, key, axis)
       1832 other_axes = [ax for ax in range(self._AXIS_LEN) if ax != axis]
       1834 if self._is_label_reference(key, axis=axis):
    -> 1835     self._check_label_or_level_ambiguity(key, axis=axis)
       1836     values = self.xs(key, axis=other_axes[0])._values
       1837 elif self._is_level_reference(key, axis=axis):


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py:1794, in NDFrame._check_label_or_level_ambiguity(self, key, axis)
       1786 label_article, label_type = (
       1787     ("a", "column") if axis == 0 else ("an", "index")
       1788 )
       1790 msg = (
       1791     f"'{key}' is both {level_article} {level_type} level and "
       1792     f"{label_article} {label_type} label, which is ambiguous."
       1793 )
    -> 1794 raise ValueError(msg)


    ValueError: 'SK_ID_CURR' is both an index level and a column label, which is ambiguous.


Error 이유 -> 인덱스 명이 동일하기 때문


```python
pd.DataFrame(prev_refused_agg)
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
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100006</th>
      <td>1</td>
    </tr>
    <tr>
      <th>100011</th>
      <td>1</td>
    </tr>
    <tr>
      <th>100027</th>
      <td>1</td>
    </tr>
    <tr>
      <th>100030</th>
      <td>10</td>
    </tr>
    <tr>
      <th>100035</th>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>456244</th>
      <td>1</td>
    </tr>
    <tr>
      <th>456247</th>
      <td>1</td>
    </tr>
    <tr>
      <th>456249</th>
      <td>1</td>
    </tr>
    <tr>
      <th>456250</th>
      <td>1</td>
    </tr>
    <tr>
      <th>456255</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>118277 rows × 1 columns</p>
</div>




```python
prev_refused_agg.reset_index()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [42], in <cell line: 1>()
    ----> 1 prev_refused_agg.reset_index()


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/util/_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/series.py:1494, in Series.reset_index(self, level, drop, name, inplace)
       1491         name = self.name
       1493 df = self.to_frame(name)
    -> 1494 return df.reset_index(level=level, drop=drop)


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/util/_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py:5839, in DataFrame.reset_index(self, level, drop, inplace, col_level, col_fill)
       5833         if lab is not None:
       5834             # if we have the codes, extract the values with a mask
       5835             level_values = algorithms.take(
       5836                 level_values, lab, allow_fill=True, fill_value=lev._na_value
       5837             )
    -> 5839         new_obj.insert(0, name, level_values)
       5841 new_obj.index = new_index
       5842 if not inplace:


    File /opt/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py:4440, in DataFrame.insert(self, loc, column, value, allow_duplicates)
       4434     raise ValueError(
       4435         "Cannot specify 'allow_duplicates=True' when "
       4436         "'self.flags.allows_duplicate_labels' is False."
       4437     )
       4438 if not allow_duplicates and column in self.columns:
       4439     # Should this be a different kind of error??
    -> 4440     raise ValueError(f"cannot insert {column}, already exists")
       4441 if not isinstance(loc, int):
       4442     raise TypeError("loc must be int")


    ValueError: cannot insert SK_ID_CURR, already exists



```python
# 일반적으로 groupby key를 INDEX로 하는 것보다 일반 컬럼으로 하는 것이 여러가지 오류 예방에 효율적.
prev_refused_agg = prev_refused_agg.reset_index(name = 'PREV_REFUSED_COUNT')
```


```python
prev_refused_agg
```


```python
prev_amt_agg = prev_amt_agg.reset_index()

prev_amt_refused_agg = prev_amt_agg.merge(prev_refused_agg, on = 'SK_ID_CURR', how = 'left')
prev_amt_refused_agg.head()
```

#### 계산된 PREV_REFUSED_COUNT중 NULL값은 0으로 변경하고 SK_ID_CURR개별 건수 대비 PREV_REFUSED_COUNT 비율 계산


```python
prev_amt_refused_agg['PREV_REFUSED_COUNT'].value_counts(dropna = False)
```


```python
prev_amt_refused_agg.head(30)
```

##### 세부 레벨 groupby 와 unstack()을 이용하여 SQL Group by Case when 구현. 세부 조건이 2개 이상일때


###### SQL 일 경우
SELECT COUNT(CASE WHEN == 'Approved' END) , COUNT(CASE WHEN == 'Refused' END) FROM PREV GROUP BY SK_ID_CURR

##### Pandas로 수행


```python
# 원래 groupby 컬럼 + 세부 기준 컬럼으로 groupby 수행. 세분화된 레벨로 aggregation 수행 한 뒤에 unstack()으로 컬럼레벨로 변환
prev_refused_appr_group = prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])].groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
```


```python
prev_refused_appr_group['SK_ID_CURR'].count()
```


```python
prev_refused_appr_group['SK_ID_CURR'].count().unstack() # unstack : 쌓지 않고 새로운 속성으로
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [43], in <cell line: 1>()
    ----> 1 prev_refused_appr_group['SK_ID_CURR'].count().unstack()


    NameError: name 'prev_refused_appr_group' is not defined



```python
prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()
prev_refused_appr_agg.head(30)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [44], in <cell line: 1>()
    ----> 1 prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()
          2 prev_refused_appr_agg.head(30)


    NameError: name 'prev_refused_appr_group' is not defined



```python
# prev_refu_appr_group['NAME_CONTRACT_STATUS'].count()
# prev_refu_appr_group['NAME_CONTRACT_STATUS'].count().unstack()
```

### 컬럼명 변경, Null 처리, 그리고 기존의 prev_amt_agg와 조인 후 데이터 가공


```python
# 컬럼명 변경.
prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT']
# NaN값은 모두 0으로 변경
prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [46], in <cell line: 2>()
          1 # 컬럼명 변경.
    ----> 2 prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT']
          3 # NaN값은 모두 0으로 변경
          4 prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)


    NameError: name 'prev_refused_appr_agg' is not defined



```python
prev_refused_appr_agg.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [47], in <cell line: 1>()
    ----> 1 prev_refused_appr_agg.head()


    NameError: name 'prev_refused_appr_agg' is not defined



```python
prev_amt_agg.head()
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
      <th>PREV_SK_ID_CURR_COUNT</th>
      <th>PREV_AMT_CREDIT_MEAN</th>
      <th>PREV_AMT_CREDIT_MAX</th>
      <th>PREV_AMT_CREDIT_SUM</th>
      <th>PREV_AMT_ANNUITY_MEAN</th>
      <th>PREV_AMT_ANNUITY_MAX</th>
      <th>PREV_AMT_ANNUITY_SUM</th>
      <th>PREV_AMT_APPLICATION_MEAN</th>
      <th>PREV_AMT_APPLICATION_MAX</th>
      <th>PREV_AMT_APPLICATION_SUM</th>
      <th>PREV_AMT_DOWN_PAYMENT_MEAN</th>
      <th>PREV_AMT_DOWN_PAYMENT_MAX</th>
      <th>PREV_AMT_DOWN_PAYMENT_SUM</th>
      <th>PREV_AMT_GOODS_PRICE_MEAN</th>
      <th>PREV_AMT_GOODS_PRICE_MAX</th>
      <th>PREV_AMT_GOODS_PRICE_SUM</th>
      <th>PREV_RATE_DOWN_PAYMENT_MIN</th>
      <th>PREV_RATE_DOWN_PAYMENT_MAX</th>
      <th>PREV_RATE_DOWN_PAYMENT_MEAN</th>
      <th>PREV_DAYS_DECISION_MIN</th>
      <th>PREV_DAYS_DECISION_MAX</th>
      <th>PREV_DAYS_DECISION_MEAN</th>
      <th>PREV_CNT_PAYMENT_MEAN</th>
      <th>PREV_CNT_PAYMENT_SUM</th>
      <th>PREV_PREV_CREDIT_DIFF_MEAN</th>
      <th>PREV_PREV_CREDIT_DIFF_MAX</th>
      <th>PREV_PREV_CREDIT_DIFF_SUM</th>
      <th>PREV_PREV_CREDIT_APPL_RATIO_MEAN</th>
      <th>PREV_PREV_CREDIT_APPL_RATIO_MAX</th>
      <th>PREV_PREV_GOODS_DIFF_MEAN</th>
      <th>PREV_PREV_GOODS_DIFF_MAX</th>
      <th>PREV_PREV_GOODS_DIFF_SUM</th>
      <th>PREV_PREV_GOODS_APPL_RATIO_MEAN</th>
      <th>PREV_PREV_GOODS_APPL_RATIO_MAX</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_MEAN</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_MAX</th>
      <th>PREV_PREV_DAYS_LAST_DUE_DIFF_SUM</th>
      <th>PREV_PREV_INTERESTS_RATE_MEAN</th>
      <th>PREV_PREV_INTERESTS_RATE_MAX</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
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
      <th>100001</th>
      <td>1</td>
      <td>23787.00</td>
      <td>23787.0</td>
      <td>23787.0</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>2520.0</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>0.104326</td>
      <td>0.104326</td>
      <td>0.104326</td>
      <td>-1740</td>
      <td>-1740</td>
      <td>-1740.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>1048.5</td>
      <td>1048.5</td>
      <td>1048.5</td>
      <td>0.957782</td>
      <td>0.957782</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>0.041099</td>
      <td>0.041099</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-606</td>
      <td>-606</td>
      <td>-606.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>0.010003</td>
      <td>0.010003</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>3</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>169661.970</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>3442.5</td>
      <td>6885.0</td>
      <td>6885.0</td>
      <td>435436.5</td>
      <td>900000.0</td>
      <td>1306309.5</td>
      <td>0.000000</td>
      <td>0.100061</td>
      <td>0.050030</td>
      <td>-2341</td>
      <td>-746</td>
      <td>-1305.0</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>-48754.5</td>
      <td>756.0</td>
      <td>-146263.5</td>
      <td>1.057664</td>
      <td>1.150980</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>150.0</td>
      <td>150.0</td>
      <td>0.015272</td>
      <td>0.018533</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>1</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>4860.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>0.212008</td>
      <td>0.212008</td>
      <td>0.212008</td>
      <td>-815</td>
      <td>-815</td>
      <td>-815.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4176.0</td>
      <td>4176.0</td>
      <td>4176.0</td>
      <td>0.828021</td>
      <td>0.828021</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>0.016450</td>
      <td>0.016450</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20076.75</td>
      <td>40153.5</td>
      <td>40153.5</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>44617.5</td>
      <td>0.108964</td>
      <td>0.108964</td>
      <td>0.108964</td>
      <td>-757</td>
      <td>-315</td>
      <td>-536.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>2232.0</td>
      <td>4464.0</td>
      <td>4464.0</td>
      <td>0.899950</td>
      <td>0.899950</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>0.036537</td>
      <td>0.036537</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prev_amt_agg와 조인. prev_amt_agg와 prev_refused_appr_agg 모두 SK_ID_CURR을 INDEX로 가지고 있음. 
prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
# SK_ID_CURR별 과거 대출건수 대비 APPROVED_COUNT 및 REFUSED_COUNT 비율 생성. 
prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT'] 
prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT'] 
# 'PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT' 컬럼 drop
prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'], axis=1)
# prev_amt_agg와 prev_refused_appr_agg INDEX인 SK_ID_CURR이 조인 후 정식 컬럼으로 생성됨. 
prev_agg.head(30)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [49], in <cell line: 2>()
          1 # prev_amt_agg와 조인. prev_amt_agg와 prev_refused_appr_agg 모두 SK_ID_CURR을 INDEX로 가지고 있음. 
    ----> 2 prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
          3 # SK_ID_CURR별 과거 대출건수 대비 APPROVED_COUNT 및 REFUSED_COUNT 비율 생성. 
          4 prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT'] 


    NameError: name 'prev_refused_appr_agg' is not defined


#### 가공된 최종 데이터 세트 생성
##### 이전에 application 데이터 세트의 feature engineering 수행 후 새롭게 previous 데이터 세트로 가공된 데이터를 조인. 


```python
apps_all =  get_apps_processed(apps)
```


```python
print(apps_all.shape, prev_agg.shape)
apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
print(apps_all.shape)
```

    (356255, 135) (338857, 6)
    (356255, 141)



```python
apps_all.head()
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
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>LANDAREA_AVG</th>
      <th>LIVINGAPARTMENTS_AVG</th>
      <th>LIVINGAREA_AVG</th>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_MODE</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_MODE</th>
      <th>ENTRANCES_MODE</th>
      <th>FLOORSMAX_MODE</th>
      <th>FLOORSMIN_MODE</th>
      <th>LANDAREA_MODE</th>
      <th>LIVINGAPARTMENTS_MODE</th>
      <th>LIVINGAREA_MODE</th>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>APARTMENTS_MEDI</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>COMMONAREA_MEDI</th>
      <th>ELEVATORS_MEDI</th>
      <th>ENTRANCES_MEDI</th>
      <th>FLOORSMAX_MEDI</th>
      <th>FLOORSMIN_MEDI</th>
      <th>LANDAREA_MEDI</th>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
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
      <th>APPS_EXT_SOURCE_MEAN</th>
      <th>APPS_EXT_SOURCE_STD</th>
      <th>APPS_ANNUITY_CREDIT_RATIO</th>
      <th>APPS_GOODS_CREDIT_RATIO</th>
      <th>APPS_ANNUITY_INCOME_RATIO</th>
      <th>APPS_CREDIT_INCOME_RATIO</th>
      <th>APPS_GOODS_INCOME_RATIO</th>
      <th>APPS_CNT_FAM_INCOME_RATIO</th>
      <th>APPS_EMPLOYED_BIRTH_RATIO</th>
      <th>APPS_INCOME_EMPLOYED_RATIO</th>
      <th>APPS_INCOME_BIRTH_RATIO</th>
      <th>APPS_CAR_BIRTH_RATIO</th>
      <th>APPS_CAR_EMPLOYED_RATIO</th>
      <th>mean_x</th>
      <th>max_x</th>
      <th>sum_x</th>
      <th>mean_y</th>
      <th>max_y</th>
      <th>sum_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.018801</td>
      <td>-9461</td>
      <td>-637</td>
      <td>-3648.0</td>
      <td>-2120</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
      <td>0.0247</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6192</td>
      <td>0.0143</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0369</td>
      <td>0.0202</td>
      <td>0.0190</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0252</td>
      <td>0.0383</td>
      <td>0.9722</td>
      <td>0.6341</td>
      <td>0.0144</td>
      <td>0.0000</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0377</td>
      <td>0.022</td>
      <td>0.0198</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0250</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6243</td>
      <td>0.0144</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0375</td>
      <td>0.0205</td>
      <td>0.0193</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.161787</td>
      <td>0.092026</td>
      <td>0.060749</td>
      <td>0.863262</td>
      <td>0.121978</td>
      <td>2.007889</td>
      <td>1.733333</td>
      <td>202500.0</td>
      <td>0.067329</td>
      <td>-317.896389</td>
      <td>-21.403657</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003541</td>
      <td>-16765</td>
      <td>-1188</td>
      <td>-1186.0</td>
      <td>-291</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>School</td>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>NaN</td>
      <td>0.0959</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>0.0605</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0130</td>
      <td>0.0773</td>
      <td>0.0549</td>
      <td>0.0039</td>
      <td>0.0098</td>
      <td>0.0924</td>
      <td>0.0538</td>
      <td>0.9851</td>
      <td>0.8040</td>
      <td>0.0497</td>
      <td>0.0806</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0128</td>
      <td>0.079</td>
      <td>0.0554</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0968</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7987</td>
      <td>0.0608</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0132</td>
      <td>0.0787</td>
      <td>0.0558</td>
      <td>0.0039</td>
      <td>0.01</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.466757</td>
      <td>0.219895</td>
      <td>0.027598</td>
      <td>0.873211</td>
      <td>0.132217</td>
      <td>4.790750</td>
      <td>4.183333</td>
      <td>135000.0</td>
      <td>0.070862</td>
      <td>-227.272727</td>
      <td>-16.104981</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>484191.00</td>
      <td>1035882.0</td>
      <td>1452573.0</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>169661.970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-19046</td>
      <td>-225</td>
      <td>-4260.0</td>
      <td>-2531</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.555912</td>
      <td>0.729567</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.642739</td>
      <td>0.122792</td>
      <td>0.050000</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>67500.0</td>
      <td>0.011814</td>
      <td>-300.000000</td>
      <td>-3.544051</td>
      <td>-0.001365</td>
      <td>-0.115556</td>
      <td>20106.00</td>
      <td>20106.0</td>
      <td>20106.0</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008019</td>
      <td>-19005</td>
      <td>-3039</td>
      <td>-9833.0</td>
      <td>-2437</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.650442</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.650442</td>
      <td>0.151008</td>
      <td>0.094941</td>
      <td>0.949845</td>
      <td>0.219900</td>
      <td>2.316167</td>
      <td>2.200000</td>
      <td>67500.0</td>
      <td>0.159905</td>
      <td>-44.422507</td>
      <td>-7.103394</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>291695.50</td>
      <td>906615.0</td>
      <td>2625259.5</td>
      <td>23651.175</td>
      <td>39954.510</td>
      <td>141907.050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19932</td>
      <td>-3038</td>
      <td>-4311.0</td>
      <td>-3458</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Religion</td>
      <td>NaN</td>
      <td>0.322738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.322738</td>
      <td>0.151008</td>
      <td>0.042623</td>
      <td>1.000000</td>
      <td>0.179963</td>
      <td>4.222222</td>
      <td>4.222222</td>
      <td>121500.0</td>
      <td>0.152418</td>
      <td>-39.993417</td>
      <td>-6.095725</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>166638.75</td>
      <td>284400.0</td>
      <td>999832.5</td>
      <td>12278.805</td>
      <td>22678.785</td>
      <td>73672.830</td>
    </tr>
  </tbody>
</table>
</div>




```python
apps_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 356255 entries, 0 to 356254
    Columns: 141 entries, SK_ID_CURR to sum_y
    dtypes: float64(85), int64(40), object(16)
    memory usage: 386.0+ MB


#### 데이터 레이블 인코딩, NULL값은 LightGBM 내부에서 처리하도록 특별한 변경하지 않음. 


```python
object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
for column in object_columns:
    apps_all[column] = pd.factorize(apps_all[column])[0]
```


```python
apps_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 356255 entries, 0 to 356254
    Columns: 141 entries, SK_ID_CURR to sum_y
    dtypes: float64(85), int64(56)
    memory usage: 386.0 MB


### 학습 데이터와 테스트 데이터 다시 분리


```python
apps_all_train = apps_all[~apps_all['TARGET'].isnull()]
apps_all_test = apps_all[apps_all['TARGET'].isnull()]

apps_all_test = apps_all_test.drop('TARGET', axis=1)
```


```python
apps_all_train.columns.tolist()
```




    ['SK_ID_CURR',
     'TARGET',
     'NAME_CONTRACT_TYPE',
     'CODE_GENDER',
     'FLAG_OWN_CAR',
     'FLAG_OWN_REALTY',
     'CNT_CHILDREN',
     'AMT_INCOME_TOTAL',
     'AMT_CREDIT',
     'AMT_ANNUITY',
     'AMT_GOODS_PRICE',
     'NAME_TYPE_SUITE',
     'NAME_INCOME_TYPE',
     'NAME_EDUCATION_TYPE',
     'NAME_FAMILY_STATUS',
     'NAME_HOUSING_TYPE',
     'REGION_POPULATION_RELATIVE',
     'DAYS_BIRTH',
     'DAYS_EMPLOYED',
     'DAYS_REGISTRATION',
     'DAYS_ID_PUBLISH',
     'OWN_CAR_AGE',
     'FLAG_MOBIL',
     'FLAG_EMP_PHONE',
     'FLAG_WORK_PHONE',
     'FLAG_CONT_MOBILE',
     'FLAG_PHONE',
     'FLAG_EMAIL',
     'OCCUPATION_TYPE',
     'CNT_FAM_MEMBERS',
     'REGION_RATING_CLIENT',
     'REGION_RATING_CLIENT_W_CITY',
     'WEEKDAY_APPR_PROCESS_START',
     'HOUR_APPR_PROCESS_START',
     'REG_REGION_NOT_LIVE_REGION',
     'REG_REGION_NOT_WORK_REGION',
     'LIVE_REGION_NOT_WORK_REGION',
     'REG_CITY_NOT_LIVE_CITY',
     'REG_CITY_NOT_WORK_CITY',
     'LIVE_CITY_NOT_WORK_CITY',
     'ORGANIZATION_TYPE',
     'EXT_SOURCE_1',
     'EXT_SOURCE_2',
     'EXT_SOURCE_3',
     'APARTMENTS_AVG',
     'BASEMENTAREA_AVG',
     'YEARS_BEGINEXPLUATATION_AVG',
     'YEARS_BUILD_AVG',
     'COMMONAREA_AVG',
     'ELEVATORS_AVG',
     'ENTRANCES_AVG',
     'FLOORSMAX_AVG',
     'FLOORSMIN_AVG',
     'LANDAREA_AVG',
     'LIVINGAPARTMENTS_AVG',
     'LIVINGAREA_AVG',
     'NONLIVINGAPARTMENTS_AVG',
     'NONLIVINGAREA_AVG',
     'APARTMENTS_MODE',
     'BASEMENTAREA_MODE',
     'YEARS_BEGINEXPLUATATION_MODE',
     'YEARS_BUILD_MODE',
     'COMMONAREA_MODE',
     'ELEVATORS_MODE',
     'ENTRANCES_MODE',
     'FLOORSMAX_MODE',
     'FLOORSMIN_MODE',
     'LANDAREA_MODE',
     'LIVINGAPARTMENTS_MODE',
     'LIVINGAREA_MODE',
     'NONLIVINGAPARTMENTS_MODE',
     'NONLIVINGAREA_MODE',
     'APARTMENTS_MEDI',
     'BASEMENTAREA_MEDI',
     'YEARS_BEGINEXPLUATATION_MEDI',
     'YEARS_BUILD_MEDI',
     'COMMONAREA_MEDI',
     'ELEVATORS_MEDI',
     'ENTRANCES_MEDI',
     'FLOORSMAX_MEDI',
     'FLOORSMIN_MEDI',
     'LANDAREA_MEDI',
     'LIVINGAPARTMENTS_MEDI',
     'LIVINGAREA_MEDI',
     'NONLIVINGAPARTMENTS_MEDI',
     'NONLIVINGAREA_MEDI',
     'FONDKAPREMONT_MODE',
     'HOUSETYPE_MODE',
     'TOTALAREA_MODE',
     'WALLSMATERIAL_MODE',
     'EMERGENCYSTATE_MODE',
     'OBS_30_CNT_SOCIAL_CIRCLE',
     'DEF_30_CNT_SOCIAL_CIRCLE',
     'OBS_60_CNT_SOCIAL_CIRCLE',
     'DEF_60_CNT_SOCIAL_CIRCLE',
     'DAYS_LAST_PHONE_CHANGE',
     'FLAG_DOCUMENT_2',
     'FLAG_DOCUMENT_3',
     'FLAG_DOCUMENT_4',
     'FLAG_DOCUMENT_5',
     'FLAG_DOCUMENT_6',
     'FLAG_DOCUMENT_7',
     'FLAG_DOCUMENT_8',
     'FLAG_DOCUMENT_9',
     'FLAG_DOCUMENT_10',
     'FLAG_DOCUMENT_11',
     'FLAG_DOCUMENT_12',
     'FLAG_DOCUMENT_13',
     'FLAG_DOCUMENT_14',
     'FLAG_DOCUMENT_15',
     'FLAG_DOCUMENT_16',
     'FLAG_DOCUMENT_17',
     'FLAG_DOCUMENT_18',
     'FLAG_DOCUMENT_19',
     'FLAG_DOCUMENT_20',
     'FLAG_DOCUMENT_21',
     'AMT_REQ_CREDIT_BUREAU_HOUR',
     'AMT_REQ_CREDIT_BUREAU_DAY',
     'AMT_REQ_CREDIT_BUREAU_WEEK',
     'AMT_REQ_CREDIT_BUREAU_MON',
     'AMT_REQ_CREDIT_BUREAU_QRT',
     'AMT_REQ_CREDIT_BUREAU_YEAR',
     'APPS_EXT_SOURCE_MEAN',
     'APPS_EXT_SOURCE_STD',
     'APPS_ANNUITY_CREDIT_RATIO',
     'APPS_GOODS_CREDIT_RATIO',
     'APPS_ANNUITY_INCOME_RATIO',
     'APPS_CREDIT_INCOME_RATIO',
     'APPS_GOODS_INCOME_RATIO',
     'APPS_CNT_FAM_INCOME_RATIO',
     'APPS_EMPLOYED_BIRTH_RATIO',
     'APPS_INCOME_EMPLOYED_RATIO',
     'APPS_INCOME_BIRTH_RATIO',
     'APPS_CAR_BIRTH_RATIO',
     'APPS_CAR_EMPLOYED_RATIO',
     'mean_x',
     'max_x',
     'sum_x',
     'mean_y',
     'max_y',
     'sum_y']



#### 학습 데이터를 검증 데이터로 분리하고 LGBM Classifier로 학습 수행. 


```python
from sklearn.model_selection import train_test_split

ftr_app = apps_all_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
target_app = apps_all_train['TARGET']

train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size=0.3, random_state=2020)
train_x.shape, valid_x.shape
```




    ((215257, 139), (92254, 139))




```python
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
        n_jobs=-1,
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=32,
        subsample=0.8,
        max_depth=12,
        silent=-1,
        verbose=-1
        )

clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'auc', verbose= 100, 
        early_stopping_rounds= 50)
```

    Training until validation scores don't improve for 50 rounds
    [100]	training's auc: 0.760602	training's binary_logloss: 0.247427	valid_1's auc: 0.74972	valid_1's binary_logloss: 0.249567
    [200]	training's auc: 0.782476	training's binary_logloss: 0.240014	valid_1's auc: 0.7607	valid_1's binary_logloss: 0.245421
    [300]	training's auc: 0.797854	training's binary_logloss: 0.235014	valid_1's auc: 0.766221	valid_1's binary_logloss: 0.243641
    [400]	training's auc: 0.810029	training's binary_logloss: 0.231029	valid_1's auc: 0.76822	valid_1's binary_logloss: 0.242951
    [500]	training's auc: 0.820909	training's binary_logloss: 0.227523	valid_1's auc: 0.768796	valid_1's binary_logloss: 0.242733
    [600]	training's auc: 0.831141	training's binary_logloss: 0.224214	valid_1's auc: 0.769067	valid_1's binary_logloss: 0.242615
    [700]	training's auc: 0.839926	training's binary_logloss: 0.221131	valid_1's auc: 0.769403	valid_1's binary_logloss: 0.242487
    [800]	training's auc: 0.848738	training's binary_logloss: 0.218084	valid_1's auc: 0.769791	valid_1's binary_logloss: 0.242372
    [900]	training's auc: 0.856637	training's binary_logloss: 0.215239	valid_1's auc: 0.769962	valid_1's binary_logloss: 0.242315
    Early stopping, best iteration is:
    [874]	training's auc: 0.854704	training's binary_logloss: 0.215952	valid_1's auc: 0.769995	valid_1's binary_logloss: 0.242305





    LGBMClassifier(learning_rate=0.02, max_depth=12, n_estimators=1000,
                   num_leaves=32, silent=-1, subsample=0.8, verbose=-1)




```python
from lightgbm import plot_importance

plot_importance(clf, figsize=(16, 32))
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='Feature importance', ylabel='Features'>


![output_108_1](https://user-images.githubusercontent.com/87309905/199921666-137aa41e-c6c3-4a2b-a1b1-21229f70600c.png)
  

    


#### 학습된 Classifier를 이용하여 테스트 데이터 예측하고 결과를 Kaggle로 Submit 수행. 


```python
preds = clf.predict_proba(apps_all_test.drop('SK_ID_CURR', axis=1))[:, 1 ]
apps_all_test['TARGET'] = preds
```

##### 코랩 버전은 Google Drive로 예측 결과 CSV를 생성.


```python
# import os, sys 
# from google.colab import drive 

# drive.mount('/content/gdrive')
```


```python
# # SK_ID_CURR과 TARGET 값만 csv 형태로 생성. 코랩 버전은 구글 드라이브 절대 경로로 입력  
# default_dir = "/content/gdrive/My Drive"
# app_test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(default_dir,'prev_baseline_01.csv'), index=False)
```

#### 지금까지 만든 로직을 별도의 함수로 생성. 


```python
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# 신청금액과 실제 대출액, 상품금액과의 차이 비율, 만기일 차이 비교, 이자율 계산등의 주요 컬럼 가공 생산. 
def get_prev_processed(prev):
    # 대출 신청 금액과 실제 대출액/대출 상품금액 차이 및 비율
    prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT']/prev['AMT_APPLICATION']
    # prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE']/prev['AMT_APPLICATION']
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # 첫번째 만기일과 마지막 만기일까지의 기간
    prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    # 매월 납부 금액과 납부 횟수 곱해서 전체 납부 금액 구함. 
    all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    # 전체 납부 금액 대비 AMT_CREDIT 비율을 구하고 여기에 다시 납부횟수로 나누어서 이자율 계산. 
    prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']
        
    return prev

# 기존 컬럼및 위에서 가공된 신규 컬럼들에 대해서 SK_ID_CURR 레벨로 Aggregation 수행.  
def get_prev_amt_agg(prev):

    agg_dict = {
         # 기존 주요 컬럼들을 SK_ID_CURR 레벨로 Aggregation 수행. . 
        'SK_ID_CURR':['count'],
        'AMT_CREDIT':['mean', 'max', 'sum'],
        'AMT_ANNUITY':['mean', 'max', 'sum'], 
        'AMT_APPLICATION':['mean', 'max', 'sum'],
        'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
        'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        # 신규 가공 컬럼들을 SK_ID_CURR 레벨로 Aggregation 수행. .
        'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
        'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
        'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
        'PREV_GOODS_APPL_RATIO':['mean', 'max'],
        'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
        'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_group = prev.groupby('SK_ID_CURR')
    prev_amt_agg = prev_group.agg(agg_dict)

    # multi index 컬럼을 '_'로 연결하여 컬럼명 변경
    prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]
    
    # 'SK_ID_CURR'로 조인하기 위해 SK_ID_CURR을 컬럼으로 변환  
    prev_amt_agg = prev_amt_agg.reset_index()
    
    return prev_amt_agg

# NAME_CONTRACT_STATUS의 SK_ID_CURR별 Approved, Refused의 건수 계산.  
def get_prev_refused_appr_agg(prev):
    # 원래 groupby 컬럼 + 세부 기준 컬럼으로 groupby 수행. 세분화된 레벨로 aggregation 수행 한 뒤에 unstack()으로 컬럼레벨로 변형. 
    prev_refused_appr_group = prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])].groupby([ 'SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()
    # 컬럼명 변경. 
    prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT' ]
    # NaN값은 모두 0으로 변경. 
    prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)
    
    # 'SK_ID_CURR'로 조인하기 위해 SK_ID_CURR을 컬럼으로 변환  
    prev_refused_appr_agg = prev_refused_appr_agg.reset_index()
    
    return prev_refused_appr_agg

    
# 앞에서 구한 prev_amt_agg와 prev_refused_appr_agg를 조인하고 SK_ID_CURR별 APPROVED_COUNT 및 REFUSED_COUNT 비율 생성
def get_prev_agg(prev):
    prev = get_prev_processed(prev)
    prev_amt_agg = get_prev_amt_agg(prev)
    prev_refused_appr_agg = get_prev_refused_appr_agg(prev)
    
    # prev_amt_agg와 조인. 
    prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
    # SK_ID_CURR별 과거 대출건수 대비 APPROVED_COUNT 및 REFUSED_COUNT 비율 생성. 
    prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    # 'PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT' 컬럼 drop 
    prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'], axis=1)
    
    return prev_agg

# apps와 previous 데이터 세트를 SK_ID_CURR레벨로 다양한 컬럼이 aggregation되어 있는 prev_agg 조인
def get_apps_all_with_prev_agg(apps, prev):
    apps_all =  get_apps_processed(apps)
    prev_agg = get_prev_agg(prev)
    print('prev_agg shape:', prev_agg.shape)
    print('apps_all before merge shape:', apps_all.shape)
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    print('apps_all after merge with prev_agg shape:', apps_all.shape)
    
    return apps_all

# Label 인코딩 수행. 
def get_apps_all_encoded(apps_all):
    object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
    for column in object_columns:
        apps_all[column] = pd.factorize(apps_all[column])[0]
    
    return apps_all

# 학습 데이터와 테스트 데이터 세트 분리. 
def get_apps_all_train_test(apps_all):
    apps_all_train = apps_all[~apps_all['TARGET'].isnull()]
    apps_all_test = apps_all[apps_all['TARGET'].isnull()]

    apps_all_test = apps_all_test.drop('TARGET', axis=1)
    
    return apps_all_train, apps_all_test

# 학습 수행. 
def train_apps_all(apps_all_train):
    ftr_app = apps_all_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    target_app = apps_all_train['TARGET']

    train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size=0.3, random_state=2020)
    print('train shape:', train_x.shape, 'valid shape:', valid_x.shape)
    clf = LGBMClassifier(
            n_jobs=-1,
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=32,
            subsample=0.8,
            max_depth=12,
            silent=-1,
            verbose=-1
                )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'auc', verbose= 100, 
                early_stopping_rounds= 100)
    
    return clf
```

##### 함수를 호출하여 재학습 및 평가 


```python
apps_all = get_apps_all_with_prev_agg(apps, prev)
apps_all = get_apps_all_encoded(apps_all)
apps_all_train, apps_all_test = get_apps_all_train_test(apps_all)
clf = train_apps_all(apps_all_train)

```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_25104/1711856574.py:54: FutureWarning: Index.ravel returning ndarray is deprecated; in a future version this will return a view on self.
      prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]


    prev_agg shape: (338857, 42)
    apps_all before merge shape: (356255, 135)
    apps_all after merge with prev_agg shape: (356255, 176)
    train shape: (215257, 174) valid shape: (92254, 174)
    Training until validation scores don't improve for 100 rounds
    [100]	training's auc: 0.766589	training's binary_logloss: 0.245898	valid_1's auc: 0.753688	valid_1's binary_logloss: 0.248717
    [200]	training's auc: 0.78916	training's binary_logloss: 0.237745	valid_1's auc: 0.765764	valid_1's binary_logloss: 0.24412
    [300]	training's auc: 0.804615	training's binary_logloss: 0.23241	valid_1's auc: 0.770737	valid_1's binary_logloss: 0.242391
    [400]	training's auc: 0.817089	training's binary_logloss: 0.228151	valid_1's auc: 0.772765	valid_1's binary_logloss: 0.241665
    [500]	training's auc: 0.82915	training's binary_logloss: 0.224266	valid_1's auc: 0.773429	valid_1's binary_logloss: 0.241421
    [600]	training's auc: 0.839882	training's binary_logloss: 0.220713	valid_1's auc: 0.773558	valid_1's binary_logloss: 0.241337
    [700]	training's auc: 0.849389	training's binary_logloss: 0.217359	valid_1's auc: 0.773654	valid_1's binary_logloss: 0.241273
    [800]	training's auc: 0.858076	training's binary_logloss: 0.214147	valid_1's auc: 0.773849	valid_1's binary_logloss: 0.241202
    [900]	training's auc: 0.866338	training's binary_logloss: 0.21105	valid_1's auc: 0.77398	valid_1's binary_logloss: 0.241154
    [1000]	training's auc: 0.874113	training's binary_logloss: 0.207983	valid_1's auc: 0.774232	valid_1's binary_logloss: 0.241072
    Did not meet early stopping. Best iteration is:
    [1000]	training's auc: 0.874113	training's binary_logloss: 0.207983	valid_1's auc: 0.774232	valid_1's binary_logloss: 0.241072



```python
preds = clf.predict_proba(apps_all_test.drop(['SK_ID_CURR'], axis=1))[:, 1 ]
apps_all_test['TARGET'] = preds
apps_all_test[['SK_ID_CURR', 'TARGET']].to_csv('prev_baseline_03.csv', index=False)
```


```python
# preds = clf.predict_proba(apps_all_test.drop(['SK_ID_CURR','TARGET'], axis=1))[:, 1 ]
# apps_all_test['TARGET'] = preds
# apps_all_test[['SK_ID_CURR', 'TARGET']].to_csv('prev_baseline_03.csv', index=False)
```
