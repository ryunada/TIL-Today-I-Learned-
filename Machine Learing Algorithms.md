# Machine Learning Algorithms

## 1-1. 나이브 베이즈(Naive Bayes)

- 베이즈 법칙을 이용하여 각 클래스에 속할 확률을 추정함으로써 결과를 예측하는 기법

< 베이즈 법칙 >  
- 가정
    - 변수들이 서로 독립, 모드 동등 → 변수들 마다 중요도가 다른 경우에도 잘 작동
    - 각 클래스 내 데이터 셋은 정규분포를 따른다.

| 장 * 단점 | <center>설명</center>                                        |
| :-------: | :----------------------------------------------------------- |
|   장점    | • 튜닝(Tuning)해야 하는 초모수(Hyperparameter)가 없음.<br /> • Topic 분류 문제에 잘 작동. |
|   단점    | • 연속형 예측 변수는 정규 분포를 가정. <br />• 예측 변수들은 서로 독립이어야 함. |

<img src = "https://p.ipic.vip/5a7auk.png" width = "100%">

- 종류
    - 가우시안 나이브 베이즈(Gaussian Naive Bayes)
        - 연속형 데이터를 처리하는데 사용.
    - 베르누이 나이브 베이즈(Bernoulli Naive Bayes)
        - 이진 데이터를 처리하는데 사용.
    - 다항 나이브 베이즈(Multinomial Naive Bayes) 
        - 다중 클래스 분류와 텍스트 분류에 주로 사용.

```Python
from sklearn.naive_bayes import GaussianNB

# 나이브 베이즈(Naive Bayes) 모형 정의
GaussianNB()
```

## 1-2. 로지스틱 회귀(Logistic Regression)

- 반응 변수가 특정 클래스에 속할 확률을 모델링하는 방법

I. 이항 로지스틱 회귀(Binomial Logistic Regression)  
< 클래스가 2개 일때, 분류과정 >

1. 학습된 모형을 이용하여 각 Case에 대한 로그 오즈를 계산
2. 함수 'logistic'을 이용하여 로그 오즈를 확률 p로 변환
3. 확률 p가 Cutoff Value보다 크면 관심 클래스로 분류

<img src = "https://p.ipic.vip/txny5g.png">

II. 다항 로지스틱 회귀(Multinomial Rogistic Regression)
< 클래스가 3개 이상일 때, 분류과정>

1. 학습된 모형을 이용하여 각 case에 대해 클래스 당 하나의 로그 오즈 계산
2. 함수 'Soft Max'를 이용하여 각 클래스에 대한 로그 오즈를 확률 $p$로 변환
3. 가장 큰 확률을 가지는 클래스를 관심 클래스로 분류

<img src = "https://p.ipic.vip/xueako.png">



< **로그 오즈 (Log odds) = $ln{{p} \over {1-p}}$ **>

☞ 로그 오즈 $> 0$ 이면, 어떤 일이 발생할 가능성이 발생하지 않을 가능성보다 높음.

​	$({{1} \over {1-p}} > 1 → p > 1 - p)$

☞ 로그 오즈 $< 0$ 이면, 어떤 일이 발생할 가능성이 발생하지 않을 가능성보다 낮음.

​	$({{1} \over {1-p}} < 1 → p < 1 - p)$

☞ 로그 오즈 $= 0$ 이면, 어떤 일이 발생할 가능성이 발생하지 않을 가능성보다 같음.

​	$({{1} \over {1-p}} = 1 → p = 1 - p)$



< **오즈 $= {{p} \over {1 - p}} = exp(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_kx_k)$** >

☞ 사건이 발생할 가능성이 발생하지 않을 가능성보다 얼마나 더 높은지 알려줌

☞ <u>예측 변수의 단위가 한 단위 증가하는 것이 확률 $p$에 어떤 영향을 미치는지</u> 알고 싶을 때 유용

​	↓

- **오즈비**

  ☞ 특정 예측 변수를 제외한 다른 예측 변수들의 값을 상수로 고정하고, **특정 예측변수($x_j$)의 값을 한 단위 증가할 때 오즈비**

  ​	⇥ 오즈비 = $exp(\beta_j)$ : $x_j$가 한 단위 증가할 때 사건이 발생할 오즈비 $exp(\beta_j)$배 만큼 증가

  ☞ 오즈비 $> 1$이면, 예측 변수의 값이 증가할수록 사건이 발생할 가능성이 높아짐

  ☞ 오즈비 $< 1$이면, 예측 변수의 값이 증가할수록 사건이 발생할 가능성이 적어짐

  	- 오즈비 $< 1$일때, '${{1} \over {오즈비}}$'를 이용하여 해석 가능

```python
from sklearn.linear_model import LogisticRegression

# Logistic Regression 모형 정의(파라미터 기본값)
LogisticRegression(
    solver = 'liblinear',          # 최적화 문제 해결하는 알고리즘
    max_iter = 100,                # 최적화 알고리즘에서 반복하는 최대 횟수
    penalty = 'l2',                # 페널티 부과에 사용될 정규화 기법
    C = 1.0,                       # 정규화 강도 조절
    multi_class = 'ovr',           # 다중 범주를 분류하는 방식
    class_weight = None,           # 'balanced' or 각 범주별 가중치가 적힌 딕셔너리
)
```

<details>
    <summary>Parameters</summary>

- `solver` : 최적화 문제를 해결하는 알고리즘
    - 'lbfgs(Limited-memory Broyden-Fletcher_Goldfarb-Shnno)'
      - 뉴튼-랩슨 방ㅂ버을 약간 변형한 방법
        - 2차 도함수를 정확히 계산하는 것이 아니라 근사치를 사용하여 속도를 개선
      - 사용할 수 있는 규제화 방법 : 'l2' & None
      -  scikit-learn 0.22부터 기본값
    - 'liblinear'
      - scikit-learn 0.21까지 기본값
      - 경사하강법과 비슷한 방법
      - 한 번에 하나의 파라미터만 업데이트
      - 사용할 수 있는 규제화 방법 : 'l1' & 'l2'
    - '뉴튼 랩슨 newton-cg(Newton-Raphson)'
      - 경사하강법과 마찬가지로 여러 번의 업데이트를 통해서 비용함수를 최소화하는 파라미터 값을 찾는 방법
      - 2차 도함수를 계산하는 것이 필요하다는 단점
      - 사용할 수 있는 규제화 방법 : 'l2' & None
    - 'sag(Stochastic Average Gradient decent)'
      - 경사하강법과 유사하게 작동
        - 이전 업데이트 단계에서 경사값을 현재 업데이트에 사용 → 학습속도가 더 빨라짐
      - 사용할 수 있는 규제화 방법 : 'l2' & 'None'
    - 'saga'
      - 사용할 수 있는 규제화 방법 : 'l1', 'l2', 'elasticnet', 'None'
- `max_iter` : 최적화 알고리즘에서 반복하는 최대 횟수
- `penalty` : 페널티 부과에 사용될 정규화 기법
    - 'l1' : L1 규제 
    - 'l2' : L2 규제
- `C` : 정규화 강도 조절
- `multi_class` : 다중 범주를 분류하는 방식
    - 'ovr' : 이진 분류(scikit-learn V 0.22이전 기본값)
    - 'multinomial' : 다중 분류
    - 'auto' : 자동으로 이진 분류 또는 다중 분류 중 하나를 선택(scikit-learn V 0.22이후 기본값)  
- `class_weight` : target의 각 유니크 값 가중치
    - 'balanced' : 자동으로 유니크 값 불균형을 고려한 가중치를 설정 
    - ex : {0 : weight_for_class_0, 1 : weight_for_class_1}

</details>



![image-20231120172909934]()
