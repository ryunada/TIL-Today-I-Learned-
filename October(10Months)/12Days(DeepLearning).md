#7장 퍼셉트론과 인공지능의 시작


```python
!git clone https://github.com/taehojo/data.git   
```

    fatal: destination path 'data' already exists and is not an empty directory.


# 8장 다층 퍼셉트론

## 1.다층 퍼셉트론의 등장

'성냥 개피 여섯 개로 정삼각형 네 개를 만들 수 있는가?'
<center>
<img src="https://drive.google.com/uc?id=1Y0M0Ze_vxqgLhU83ufiBVCMTgdqitGMm" width=200>
</center>
<br>
<br>
<br>

은닉층을 활용하여 XOR 문제를 어떻게 해결하는지 그림(8-4)에 소개 합니다. 

<center>
(그림. 8-4) 은닉층이 XOR 문제를 해결하는 원리 <br>
<img src="https://drive.google.com/uc?id=16c4BNbmKI_DwLB4Ow37Slfx2tpNGX98O"><br>
<img src="https://drive.google.com/uc?id=1TbocrRdc5ZW6zg5srR9WFiUsOnjGaMER">
</center>

## 2. 다층 퍼셉트론의 설계
다층 퍼셉트론이 입력층과 출력층 사이에 숨어 있는 은닉층을 만드는 것을 그림으로 나타내면 그림(8-5)와 같습니다.  

<br>
<center>
(그림. 8-5) 다층 페셉트론의 내부<br><br>
<img src="https://drive.google.com/uc?id=1XZKqQoYGm68UVPIyX4YrmzVSacS9d5xM" width = 500>
</center>   

가운데 점선으로 표시한 부분이 은닉층입니다. $x_1$과 $x_2$는 입력값입니다. 각 입력값에 가중치($w$)를 곱하고 바이어스($b$)를 더해 은닉층으로 전송합니다. 이 값들이 모이는 은닉층의 중간 정거장을 노드(node)라고 하며, 여기서는 $n_1$과 $n_2$로 표시 되었습니다. 은닉층에 취합된 값들은 활성화 함수를 통해 다음으로 보내지는데, 만약 앞서 배운 시그모이드 함수($\sigma(x)$)를 활성화 함수로 사용한다면 $n_1$과 $n_2$에서 계산되는 식은 각각 다음과 같습니다.  

$$n_{1} = \sigma ( w_{11} x_{1} + w_{21} x_{2} + b_1 )$$
$$n_{2} = \sigma ( w_{12} x_{1} + w_{22} x_{2} + b_2 )$$  

위 두 식의 결과값($n_1 , n_2$)이 출력층의 방향으로 보내지고 출력층으로 전달된 값은 마찬가지로 활성화 함수를 사용해 예측값 $y$를 결정하게 된다. 이 값을 $y_{out}$라고 할 때 이를 식으로 표현하면 다음과 같습니다. 

$$y_{out} = \sigma (w_{31}n_{1} + w_{32}n_{2} + b_3) $$  

이제 각각의 가중치($w$)와 편향($b$)를 값을 정할 차례입니다. 2차원 배열로 늘어 놓으면 다음과 같이 표현할 수 있습니다. 은닉층을 포함해 가중치 여섯개와 편향 세 개가 필요합니다. 

$$W^{(1)} = 
\begin{bmatrix} 
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}\;\;\;\;\;\;
B^{(1)}
=\begin{bmatrix} 
b_1 \\
b_2
\end{bmatrix}
$$  

$$W^{(2)} = 
\begin{bmatrix} 
w_{31} \\
w_{32} 
\end{bmatrix}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
B^{(2)}
=\begin{bmatrix} 
b_3
\end{bmatrix}
$$


#### 여기서 잠깐

$$n_{1} = w_{11} x_{1} + w_{21} x_{2} + b_1$$
$$n_{2} = w_{12} x_{1} + w_{22} x_{2} + b_2$$  

위 두 식을 하나의 행렬식으로 바꾸면 아래와 같습니다.  

$$
\begin{bmatrix} 
n_1 \\
n_2 
\end{bmatrix}
=
\begin{bmatrix} 
w_{11} & w_{21} \\
w_{12} & w_{22}
\end{bmatrix}
\begin{bmatrix} 
x_1 \\
x_2
\end{bmatrix}
+
\begin{bmatrix} 
b_1 \\
b_2
\end{bmatrix}
$$  
아래 식을 행렬로 바꾸면  
$$y_{out} = w_{31}n_1 + w_{32}n_2 + b_3$$  

$$y_{out} = 
\begin{bmatrix} 
w_{31} &
w_{32}
\end{bmatrix}
\begin{bmatrix} 
n_{1} \\ n_{2}
\end{bmatrix} + b_3$$


### 3. XOR 문제의 해결
앞서 우리에게 어떤 가중치와 편향(바이어스)이 필요한지 알아보았습니다. 이를 만족하는 가중치값과 편향값의 조합은 무수히 많습니다. 이를 구하는 방법은 9장에서 소개할 예정입니다. 지금은 다음과 같이 각 변수값을 정하고 이를 이용해서 XOR 문제를 해결하는 과정을 알아보겠습니다. 

$$
W^{(1)} =
\begin{bmatrix} 
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}
=  
\begin{bmatrix} 
-2 & 2 \\
-2 & 2
\end{bmatrix}\;\;\;\;\;\;
B^{(1)}
=\begin{bmatrix} 
3 \\
-1
\end{bmatrix}
$$  

$$W^{(2)} = 
\begin{bmatrix} 
1 \\ 1 
\end{bmatrix}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
B^{(2)}
=\begin{bmatrix} 
-1
\end{bmatrix}
$$

위 값을 그림에 적용하면 그림(8-6)과 같습니다.  
<br>
<center>
(그림. 8-6) 다층 퍼셉트론의 내부에 변수 채우기<br>
<img src = "https://drive.google.com/uc?id=1daMf5co7mufH4b-oJmRhN7s-po64Zs9Y" width = 400>
</center>  

이제 $x_1$과 $x_2$ 값을 각각 입력해 우리가 원하는 $y$ 값이 나오는지 점검해 보겠습니다.   


#### [과제]
과제 1 - 시그모이드 함수를 그리시오.  
과제 2 - 아래 표와 같이 동작하는지 확인하세요

<br>
<center>
(표. 8-1) XOR 다층 문제 해결<br>
<img src="https://drive.google.com/uc?id=1Cnqr3xMAJ3_ewwMH00SrK4A8LFrC4mPa">
</center>  

## 4. 코딩으로 XOR 문제 해결하기


```python
import numpy as np

# 가중치와 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

# 퍼셉트론
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

# NAND 게이트
def NAND(x1,x2):
    return MLP(np.array([x1, x2]), w11, b1)

# OR 게이트
def OR(x1,x2):
    return MLP(np.array([x1, x2]), w12, b2)

# AND 게이트
def AND(x1,x2):
    return MLP(np.array([x1, x2]), w2, b3)

# XOR 게이트
def XOR(x1,x2):
    return AND(NAND(x1, x2),OR(x1,x2))


# x1, x2 값을 번갈아 대입해 가며 최종 값 출력
for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = XOR(x[0], x[1])
    print("입력 값: " + str(x) + " 출력 값: " + str(y))      
```

    입력 값: (0, 0) 출력 값: 0
    입력 값: (1, 0) 출력 값: 1
    입력 값: (0, 1) 출력 값: 1
    입력 값: (1, 1) 출력 값: 0


#### [과제]
과제 1. 위 코드에서 사용한 활성화 함수는 무엇인가?  
과제 2. 은닉층과 출력층에서 활성화 함수로 시그모이드 함수를 사용하여 XOR 동작을 구현하려고 한다. 위 코드를 수정하시요


```python
# 과제 1 관련

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x)) # 시그모이드 수식

x = np.arange(-4.0, 4.0, 0.1) # -5.0 ~ 5.0까지 0.1씩 증가
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) # y축 값의 범위 설정
plt.grid()
plt.show()

```


![스크린샷 2022-10-12 오후 9 01 42](https://user-images.githubusercontent.com/87309905/195337459-e876e553-6ee0-48ce-a484-940a6ae6b2c8.png)


```python
# 과제 1 관련

import numpy as np
import matplotlib.pylab as plt

# def sigmoid(x):
#     return 1/(1+np.exp(-x)) # 시그모이드 수식

def sigmoid(x):
  a = 20
  b = -10
  return 1/(1+np.exp(-(a*x + b)))

x = np.arange(-4.0, 4.0, 0.1) # -5.0 ~ 5.0까지 0.1씩 증가
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) # y축 값의 범위 설정
plt.grid()
plt.show()

```

![스크린샷 2022-10-12 오후 9 01 32](https://user-images.githubusercontent.com/87309905/195337419-5e0265f9-37fe-4509-8fa6-4b9b9d418953.png)

    



```python
# 과제 2 관련
# 위에 제시한 시그모드이 함수를 활성화 함수로 사용할 경우 제시한 표(8-1)의 결과를 얻을 수 없다.
# 따라서 다음과 같은 시그모이드 함수를 사용했다. 

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  a = 20
  b = -10
  return 1/(1+np.exp(-(a*x + b)))


x = np.linspace(-1, 3, 100)
h = sigmoid(x)

plt.figure(figsize=(8,8))
plt.plot(x,h)
plt.axhline(0,color='black')
plt.axvline(0,color='black')
plt.ylim([0, 1])
plt.xlim([-1,3])
plt.xlabel('x')
plt.ylabel('h')
plt.grid()
plt.show()
```

![스크린샷 2022-10-12 오후 9 01 21](https://user-images.githubusercontent.com/87309905/195337381-deaed5a2-1269-4560-ae52-68d69aa9f293.png)




```python
# 과제 2관련

import numpy as np

# 가중치와 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

# 퍼셉트론
def MLP(x, w, b):
    y = np.sum(w * x) + b
    return sigmoid(y)

# NAND 게이트
def NAND(x1,x2):
    return MLP(np.array([x1, x2]), w11, b1)

# OR 게이트
def OR(x1,x2):
    return MLP(np.array([x1, x2]), w12, b2)

# AND 게이트
def AND(x1,x2):
    return MLP(np.array([x1, x2]), w2, b3)

# XOR 게이트
def XOR(x1,x2):
    return AND(NAND(x1, x2),OR(x1,x2))


# x1, x2 값을 번갈아 대입해 가며 최종 값 출력
for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = XOR(x[0], x[1])
    print("입력 값: " + str(x) + " 출력 값: " + str(y))      
```

    입력 값: (0, 0) 출력 값: 4.539786870251923e-05
    입력 값: (1, 0) 출력 값: 0.999954519621495
    입력 값: (0, 1) 출력 값: 0.999954519621495
    입력 값: (1, 1) 출력 값: 4.539786870251923e-05


우리가 원하는 XOR 문제의 정답이 도출되었습니다. 이렇게 퍼셉트론 하나로 해결되지 않던 문제를 은닉층을 만들어 해결했습니다. 하지만 퍼셉트론의 문제가 완전히 해결된 것은 아니었습니다. 다층 퍼셉트론을 사용할 경우 XOR 문제는 해결되었지만, 은닉층에 들어 있는 가중치를 데이터를 통해 학습하는 방법이 아직 없었기 때문입니다. 다층 퍼셉트론의 적절한 학습 방법을 찾기까지 그 후로 약 20여 년의 시간이 더 필요했습니다. 이 시간을 흔히 인공 지능의 겨울이라고 합니다. 이 겨울을 지나며 데이터 과학자들은 크게 두 분류로 나뉩니다. 하나는 최적화된 예측선을 잘 그려주던 아달라인을 발전시켜 SVM이나 로지스틱 회귀 모델을 만든 그룹이입니다. 또 두 번째 그룹에 속해 있던 제프리 힌튼(Geoffrey Hinton) 교수가 바로 딥러닝의 아버지로 칭속 받는 사람입니다. 힌튼 교수는 여러 가지 혁신적인 아이디어로 인공지능의 겨울을 극복해 냅습니다. 첫 번째 아이디어는 **1986년에 발표한 오차 역전파**입니다. 

***

# 9장 역전파에서 딥러닝으로
해결되지 않았던 XOR 문제를 다층 퍼셉트론으로 해결했습니다. 하지만 한 가지 문제를 만났습니다. 은닉층에 포함된 가중치를 업데이트할 방법이 없었던 것입니다. 이 기나긴 인공지능의 겨울을 지나 오차 역전파라는 방법을 만나고 나서야 해결됩니다.

## 딥러닝의 태동, 오차역전파

<center>
(그림. 9-1) 단일 퍼셉트론에서 오차 수정<br>
<img src="https://drive.google.com/uc?id=1TqvVVKW8xx3PZL01j0lnI1_Yb9XV3fUA" width=300>
</center><br>

<br><center>
(그림. 9-2) 다층 퍼셉트론에서 오차 수정<br>
<img src="https://drive.google.com/uc?id=1V9FcJotbm-oG2y-grUCH3MGSX9tE82vZ" width=300>
</center><br>

<br><center>
(그림. 9-2) 다층 퍼셉트론에서 오차 수정<br>
<img src="https://drive.google.com/uc?id=1zbppX4Vr_JQDakjI72b6i5nN7AmRVIWe" width=500>
</center><br>

***

# 10장 딥러닝 모델 설계하기  (2장 딥러닝의 핵심 미리 보기)
 둘째 마당과 셋째 마당을 마스터한 독자 여러분! 축하합니다. 가끔 어려운 수기이 나오기도 했지만 모든 개념이 머리 속에 정리되었다면 여러분을 딥러닝의 세계로 성큼성큼 안내해 줄 텐서플로, 케라스와 함께 지금부터 초고속으로 전진할 것입니다. 

 지금부터는 딥러닝의 기본 개념들이 실정에서는 어떤 방식으로 구현되는지, 왜 우리가 그 어려운 개념들을 익혀야 했는지 공부하겠습니다. 

 2장에서 소개한 '페암 환자의 생존율 예측하기" 에제를 기억하시나요? 당시에는 딥러닝 모델 부분을 자세히 설명할 수 없었지만 이제는 설명할 수 있습니다. 여러분은 앞서 배운 내용이 이 짧은 코드 안에 모두 들어 있다는 사실에 감탄할지도 모릅니다. 머리 속에 차곡차곡 들어찬 딥러닝의 개념들이 어떻게 활용되는지 지금부터 함께 알아 보겠습니다.

## 1. 모델의 정의


```python
from tensorflow.keras.models import Sequential  # 텐서플로의 케라스 API에서 필요한 함수들을 불러옵니다.
                                                # 텐서프로의 케라스라는 API에 있는 모델(model) 클래스로부터 Sequential() 함수를 불러오라는 의미
from tensorflow.keras.layers import Dense       # 데이터를 다루는 데 필요한 라이브러리를 불러옵니다.
                                                # 텐서프로의 케라스라는 API에 있는 레이어(layers)에서 Dense() 함수를 불러오라는 의미
import numpy as np
################################################################################

Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")  # 수술 환자 데이터를 불러옵니다.
X = Data_set[:,0:16]                                                 # 환자의 진찰 기록을 X로 지정합니다. 16개의 특성(속성;attribute)
y = Data_set[:,16]                                                   # 수술 1년 후 사망/생존 여부를 y에 할당
################################################################################

model = Sequential()                                                  # 딥러닝 모델의 구조를 결정합니다.
# 입력 (속성)16개, 출력 30개 = 은닉층의 노드가 30개
model.add(Dense(30, input_dim=16, activation='relu'))                 # model.add() 함수로 작업 진행 층을 추가합니다.
# 출력이 1개, 활성화 함수는 'sigmoid')
model.add(Dense(1, activation='sigmoid'))
################################################################################

# 손실 함수(loss function) = binary_crossentropy  (상황에 따라 손실함수로 mse를 쓸 수 있음)
# 옵티마이저로 'adam'을 쓰고 있음. 이전 예제에서는 sgd(확율적 경사하강법)을 썼었음
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 딥러닝 모델을 실행합니다.

history=model.fit(X, y, epochs=5, batch_size=16)
```

    Epoch 1/5
    30/30 [==============================] - 1s 2ms/step - loss: 0.5185 - accuracy: 0.8362
    Epoch 2/5
    30/30 [==============================] - 0s 2ms/step - loss: 0.4897 - accuracy: 0.8511
    Epoch 3/5
    30/30 [==============================] - 0s 2ms/step - loss: 0.4840 - accuracy: 0.8511
    Epoch 4/5
    30/30 [==============================] - 0s 2ms/step - loss: 0.4714 - accuracy: 0.8511
    Epoch 5/5
    30/30 [==============================] - 0s 2ms/step - loss: 0.4717 - accuracy: 0.8489


딥러닝의 모델을 설정하고 구동하는 부분은 **model**이라는 객체를 선언하면서 시작됩니다. 먼저 model = Sequential()로 시작되는 부분은 딥러닝의 구조를 짜고 층을 설정하는 부분입니다. 이어서 나오는 model.compile() 부분은 앞에서 정한 모델을 컴퓨터가 인식할 수 있도록 해석하는 부분입니다. 그리고 model.fit()으로 시작하는 부분은 모델을 실제로 수행하는 부분입니다. 지금부터 이 세 가지 부분에 대해 하나씩 살펴보겠습니다. 

## 2. 입력층, 은닉층, 출력층
먼저 딥러닝의 구조를 짜고 층을 설정하는 부분을 사펴보면 다음과 같습니다. 
```
model = Sequential()
model.add( Dense(30, input_dim=16, activation='relu'))
model.add( Dense(1, activation='sigmoid'))
```
세번째 마당에서 딥러닝이란 입력층과 출력층 사이에 은닉층들을 차곡차곡 추가하면서 학습시키는 것임을 배웠습니다. 이 층들이 케리스에서는 sequential() 함수를 통해 쉽게 구현됩니다. sequential() 함수를 model로 선언해 놓고 model.add()라는 라인을 추가하면서 새로운 층이 만들어집니다 

위 코드에서는 model.add()로 시작되는 라인이 두 개 있으므로 층을 두 개 가진 모델을 만드는 것입니다. 맨 마지막 층은 결과를 출력하는 '출력층'이 됩니다. 나머지는 모두 '은닉층'의 역활을 합니다. 따라서 지금 만들어진 두 개의 층은 각각 은닉층과 출력층입니다. 

각각의 층은 Dense() 함수를 통해 구체적인 구조가 결정됩니다. 이제 model.add(Densse(30, inpu_dim=16)) 부분을 살펴 보겠습니다. model.add() 함수를 통해 새로운 층을 만들고 나면 Dense() 함수의 첫번째 인자에 몇 개의 노드를 이 층에 만들 것인지 숫자를 적어 줍니다. 노드란 앞서 소개한 '가중합'에 해당하는 것으로 이전 층에서 전달된 변수와 가중치, 바이어스가 하나로 모이게 되는 곳입니다. 하나의 층에 여러 개의 노드를 적절히 만들어 주어야 하는데, 30이라고 되어 있는 것은 이 층에 노드를 30 만들겠다는 것입니다. 이어서 input_dim이라는 변수가 나옵니다. 이는 입력데이터에서 몇 개의 값을 가져올지 정하는 것입니다. 케라스(keras)는 입력층을 따로 만드는 것이 아니라 처 번째 은닉층에 input_dim을 적어 줌으로써 첫 번째 Dense가 은닉층과 입력층의 역활을 같이 합니다. 우리가 다루고 있는 폐암 수술 환자의 생존 여부 데이터에는 입력 값이 16개 있습니다. 따라서 데이터에서 값을 16개 받아 은닉층의 노드 30개로 보낸다는 의미입니다.  
<br>
<center>
(그림. 10-1) 첫 번째 Dense는 입력층과 첫 번째 은닉층을, 두 번째 Dense는 출력층을 의미<br>
<img src="https://drive.google.com/uc?id=1REQGBZ4O2fgZ1gm4js1ZLRzzYIWw9cXz" width=400>
</center><br>  

이제 두 번째 나오는 model.add(Dense(1, activation='sigmoid'))를 보겠습니다. 마지막 층이므로 출력층이 됩니다. 출력값을 하나로 정해서 보여 줘야 하므로 출력층의 노드 수는 한 개입니다. 그리고 이 노드에서 입력받은 값은 활성화 함수를 거쳐 최종 출력값으로 나와야 합니다. 여기서는 활성화 함수로 시그모이드(sigmoid) 함수를 사용했습니다.



## 3. 모델 컴파일
다음으로 model.compile() 부분입니다. 
```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

model.compile 부분은 앞서 지정한 모델이 효과적으로 구현될 수 있게 여러 가지 환경을 설정해 주면서 컴파일 하는 부분입니다. 먼저 어떤 오차 함수를 사용할지 정해야 합니다. 우리는 5장에서 손실 함수에는 두 가지 종류가 있음을 배웠습니다. 바로 선형회귀에서 사용한 평균 제곱 오차와 로지스틱 회귀에서 사용한 교차 엔트로피 오차입니다. 페암 수술 환자의 생존율 예측은 생존과 사망, 둘 중 하나를 예측하므로 교차 엔트로피 오차 함수를 적용하기 위해 binary_crossentropy를 선택했습니다. 

손실함수는 최적의 가중치를 학습하기 위해 필수적인 부분입니다. 올바른 손실 함수를 통해 계산된 오차는 옵티마이저를 적절히 솰용하도록 만들어 줍니다. 케라스는 쉽게 사용 가능하 여러 가지 손실 함수를 준비해 놓고 있습니다. 크게 평균 제공 오차 계열과 교차 엔프로피 계열 오차로 나뉘는데 이를 표(10-1)에 정리해 놓았습니다. 선형 회귀 모델은 평균 제곱 계열 중 하나를, 이항 분류를 위해서는 binary_crossentropy를 그리고 다항 분류에서는 categorical_crossentropy를 사용한다는 것ㅇ르 기억합시다.  
<br>
<center>
(표. 10-1) 대표적인 오차 함수<br>
<img src="https://drive.google.com/uc?id=15jnYFz_cYGnCXrPg8vnceJPtZ0e9YoXb">
</center>
<br>  

이어서 옵티마이저를 선택할 차례입니다. 앞 장에서 현재 가장 많이 쓰는 옵티마이저는 adam이라고 했습니다. optimizer에 'adam'을 할당하는 것으로 실행할 준비가 되었습니다. metrics는 모델이 컴파일 될 때 모델 수행의 결과를 나타내게끔 설정하는 부분입니다. accuracy라고 설정한 것은 학습셋에 대한 정확도에기반해 결과를 출력하라는 의미입니다. 




## 4. 모델 실행하기
모델을 정의하고 컴파일하고 나면 이제 실행키길 차례입니다. 앞서 컴파일 단계에서 정해진 환경을 주어진 데이털를 불러 실행시킬 때 사용되는 함수는 다음과 같이 model.fit() 부분입니다.

```
history = model.fit(X, y, epochs=5, batch_size=16)
```
이 부분을 설명하기 앞서 용어를 다시 한번 정리해 보겠습니다. 주어진 폐암 환자의 수술 후 생존 여부 데이터는 총 470명의 환자에게서 16개의 정보를 정리한 것입니다. 이때 각 정보를 '속성'이라고 합니다. 그리고 생존 여부를 클래스, 가로 한 줄에 해당하는 각 환자의 정보를 '샘플'이라고 합니다. 주어진 데이터에는 총 470개의 샘픔이 가가각 16개의 속성을 가지고 있는 것이라고 앞서 설명한 바 있습니다.  

<br><center>
(그림. 10-2) 폐암 환자 생존율 예측 데이터의 샘플, 속성, 클래스 구분<br>
<img src="https://drive.google.com/uc?id=1AWqFKt9uhi6cZhjhGCdZ2PmglmXkcr69" width=500>
</center><br>  

참고  
이 용어는 문서(책)마다 조금씩 다를 수 있습니다. 예를 들어 샘플을 instance 또는 example이라고도 하며 속성 대신 피처(feature) 또는 특성이라고도 합니다. 이 책에서는 '속성'과 '샘플'로 통일해서 사용하겠습니다. 

학습 프로세스가 모든 샘플에 대해 한 번 실행되는 것을 1 epoch('에포크'라고 읽음)라고 합니다. 코드에서 epochs=5로 지정한 것은 각 샘플이 처음부터 끝까지 다섯 번 재사용될 때까지 실행을 반복하라는 의미입니다. 

batch_size는 샘플을 한 번에 몇 개씩 처리할지 정하는 부분으로 batch_size=16은 전체 470개의 샘플을 16개씩 끊어서 집어 넣으라는 의미입니다. batch_size가 너무 크면 학습 속도가 느려지고 너무 작으면 각 샐행 값의 편차가 생겨서 전체 결과값이 불안정해질 수 있습니다. 따라서 자신의 컴퓨터 메모리가 감당할 만큼의 batch_size를 찾아 설정해 주는 것이 좋습니다. 




### 1. 환경 준비 


```python
# 텐서플로 라이브러리 안에 있는 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense   

# 데이터를 다루는 데 필요한 라이브러리를 불러옵니다.
import numpy as np
```

### 2. 데이터 준비


```python
# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git   

# 준비된 수술 환자 데이터를 불러옵니다.
Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")  
X = Data_set[:,0:16]    # 환자의 진찰 기록을 X로 지정합니다.
y = Data_set[:,16]      # 수술 1년 후 사망/생존 여부를 y로 지정합니다.
```

    fatal: destination path 'data' already exists and is not an empty directory.


### 3. 구조 결정


```python
# 딥러닝 모델의 구조를 결정합니다.
model = Sequential()                                                   
model.add(Dense(30, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### 4. 모델 실행 


```python
# 딥러닝 모델을 실행합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
history=model.fit(X, y, epochs=5, batch_size=16)
```

    Epoch 1/5
    30/30 [==============================] - 1s 4ms/step - loss: 7.2116 - accuracy: 0.1511
    Epoch 2/5
    30/30 [==============================] - 0s 4ms/step - loss: 0.6488 - accuracy: 0.7298
    Epoch 3/5
    30/30 [==============================] - 0s 4ms/step - loss: 0.5050 - accuracy: 0.8489
    Epoch 4/5
    30/30 [==============================] - 0s 3ms/step - loss: 0.4617 - accuracy: 0.8468
    Epoch 5/5
    30/30 [==============================] - 0s 3ms/step - loss: 0.4582 - accuracy: 0.8489


***

# 11장 데이터 다루기

## 1. 딥러닝과 데이터
세월이 흐르면서 쌓인 방대한 양의 데이터가 있습니다. 이런 '빅데이터'는 분명히 머신 러닝과 딥러닝을 통해 사람에 버금가는 판단과 지능을 가질 수 있게 됬습니다. 하지만 데이터 양이 많다고 해서 무조건 좋은 결과를 얻을 수 있는 것은 아닙니다. 데이터 양도 중요하지만 데이터 안에 '필요한' 정보가 얼마나 있는가도 중요하기 때문입니다. 그리고 준비된 데이터가 우리가 사용하려는 머신 러님과 딥러닝에 효율적으로 사용될 수 있도록 사전에 잘 가공 되어 있느냐도 중요합니다. 

머신 러닝 프로젝ㅌ의 성공과 실패는 얼마나 좋은 데이터를 가지고 시작하느냐에 영향을 많이 받습니다. 여기서 좋은 데이터란 한쪽으로 치우지지 않고 불필요한 정보가 적게 포함되어 있어야 하며 왜곡 되지 않은 데이터를 말합니다. 이러한 양질의 데이터를 얻기 위해 머신 러인, 딥러닝 개발자들은 데이터를 직접 들여다 보고 데이터의 적합성, 적절성을 분석할 수 있어야 합니다. 이루고 싶은 목표에 맞추어 가능한 많은 데이터를 모았다면 이를 머신 러닝과 딥러닝에 사용할 수 있게 잘 정제된 데이터 형식으로 바꿀수 있어야 합니다. 이 작업은 머신 러닝, 딥러닝 프로젝트의 첫 단계이자 매우 중요한 작업니다. 가장 중요한 작업이다라고 말해도 과언이 아닐 것입니다. 

지금부터 데이터 분석에 가장 많이 사용되는 파이썬 라이브러리 판다스(pandas)와 맷플롯립(matplotlib) 등을 사용해 우리가 다룰 데이터가 어던 내용을 담고 있는지 확인하면서 딥러닝의 핵심 기술들을 하나씩 구현해 보겠습니다.

## 2. 피마인디언 데이터 분석하기
<center>
(그림. 11-1) 피마 인디언의 옛 모습<br>
<img src="https://drive.google.com/uc?id=1VHeiCzs0wXbPpgitVG3y_cEP3ym_Ctf3" width=500>
</center>

비만은 유전일까요? 아니면 식습관 조절에 실패한 자신의 탓일까요? 비만이 유전 및 화경, 모두의 탓이라는 것을 증명하는 좋은 사례가 바로 미국 남서부에 살고 있는 피마 인디언의 사례입니다. 피마 인디어은 1950년대까지마 해도 비만인 사람이 단 1명도 없는 민족이었습니다. 그런데 지금은 전체 부족의 60%가 당뇨, 80%가 비만으로 고통 받고 있습니다. 이는 생존하기 위해 영양분을 체내에 저장하는 뛰어난 능력을 물려받은 인디언들이 미국의 기름진 패스트푸드 문화를 만나면서 벌어진 일입니다. 피마 인디언을 대상으로 당뇨병(diabetes) 여부를 측정한 데이터는 data폴더에서 찾을 수 있습니다(data/pima-indians-diabetes3.csv).

이제 준비된 데이터의 내용을 들여다 보겠습니다. 768명의 인디언으로부터 여덟 개의 정보(속성)와 한 개의 클래스를 추출한 데이터임을 알 수 있습니다. 

<br><center>
(그림. 11-2) 피마 인디언 데이터 샘플, 속성, 클래스 구분<br>
<img src="https://drive.google.com/uc?id=1RaxsvsagRXixb7xb0fLGYtTjTXUJiz4T" width=600>
</center><br>

* 샘플 수 : 768
* 속성 : 8 가지
  - 정보 1 (pregnant): 과거 임신 횟수
  - 정보 2 (plasma): 포도당 부하 검사 2시간 공복 혈당 농도(mm Hg)
  - 정보 3 (pressure): 확장기 혈압(mm Hg)
  - 정보 4 (thickness): 삼두근 피부 주름 두께(mm)
  - 정보 5 (insulin): 혈청 인슐린(2-hour, mu U/ml)
  - 정보 6 (BMI): 제질량 지수(BMI, weight in kg/(height in m)<sup>2</sup>)
  - 정보 7 (pedigree):당뇨병 가족력
  - 정보 8 (age): 나이
* 클래스: 
  - 당뇨병 여부 (diabetes): 1(당뇨), 0 (당뇨병 아님)  

데이터의 각 정보가 의미하는 의학, 생리학 배경 지식을 모두 알 피료는 없지만 딥러닝을 구동하려면 반드시 속성과 틀래스를 먼저 구분해야 합니다. 또한 모델의 정확도를 향상시키기 위해서는 데이터를 추가하거나 재가공해야 할 수도 있습니다. 따라서 데이터의 내용과 구조를 파악하는 것이 중요합니다. 

<br>
🚀 본인의 제질량지수 확인 ^^  

[나의 체질량지수(BMI)](https://health.seoulmc.or.kr/healthCareInfo/myBMIPopup.do)  

BMI가 23부터 과제충이고 25부터는 비만이네요. 




## 3. 판다스를 활용한 데이터 조사
데이터를 잘 파악하는 것이 딥러닝을 다루는 기술의 1단계라고 했습니다. 그런데 데이터의 크기가 커지고 정보량이 많아지면 데이터를 불러 오고 내용을 파악하 루 있는 효과적인 방법이 필요합니다. 이때 가장 유용한 방법이 데이터를 시각화해서 눈으로 직접 확인해 보는 것입니다. 지금부터 데이터를 불러와 그래프로 표현하는 방법을 알아보겠습니다.
데이터를 다룰 때는 데이터를 다루기 위해 만들어진 라이브러리를 사용하는 것이 좋습니다. 지금까지 넘파이 라이브러리를 불러와 사용했는데, 넘파이의 기능을 포함하면서도 다양한 포멧의 데이터를 다루게 해 주는 판다스 라이브러리를 사용해서 데이터를 조사해 보겠습니다.  
<br>
📌 잠깐만요  
이 실습에는 판다스(pandas)와 시본(seaborn) 라이브러리가 필요합니다. 코랩은 기본 제공됩니다.
<br>  





```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 
df = pd.read_csv('./data/pima-indians-diabetes3.csv')
```

    fatal: destination path 'data' already exists and is not an empty directory.


판다스 라이브러리의 read_csv() 함수로 csv 파일을 불러와 df라는 이름의 데이터 프레임으로 저장했습니다. 파일 확장자 csv는 comma separated values의 약자로 해당 파일이 쉼표(,)로 구분된 텍스트 데이터 파일이란 의미입니다. 통상 csv 파일은 데이터를 각 열의 의미를 설명하는 정보가 파일 맨 위에 나타납니다. 이를 헤더(header)라고 합니다. 

이제 불러온 데이터의 내용을 간단히 확인하고자 head() 함수를 이용해 헤더와 함께  데이터의 첫 다섯 행을 불러 오겠습니다.


```python
# 처음 5줄을 봅니다.
df.head(5)
```





  <div id="df-faf8ffa1-9573-47b9-bd6a-8f9fcb93a64e">
    <div class="colab-df-container">
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
      <th>pregnant</th>
      <th>plasma</th>
      <th>pressure</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>pedigree</th>
      <th>age</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-faf8ffa1-9573-47b9-bd6a-8f9fcb93a64e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-faf8ffa1-9573-47b9-bd6a-8f9fcb93a64e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-faf8ffa1-9573-47b9-bd6a-8f9fcb93a64e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




당뇨에 걸리지 않은 정상인과 당뇨 환자가 각각 몇 명인지 조사해 봅시다. 데이터 프레임의 특정 칼럼을 불러오려면 df['컬럼명']이라고 입력하면 됩니다. value_counts() 함수를 이용하면 각 칼러의 값이 몇 개씩 있는지 알려줍니다. 


```python
# 정상과 당뇨 환자가 각각 몇 명씩인지 조사해 봅니다.
# 1 : 당뇨병, 0 : 당뇨병 아님

df["diabetes"].value_counts()
```




    0    500
    1    268
    Name: diabetes, dtype: int64



데이터 프레임이 가지고 있는 데이터에 대한 통계적 특성을 describe() 함수를 통해 확인합니다. 다음과 같은 내용이 출력됩니다. 정보별 샘플 수(count), 평규(mean) 등을 확인할 수 있습니다.  
<br>
🚶 강사가 혼자 하는 말  
임신 횟수의 평균 값이 3.8회면 대단하네요.


```python
# 각 정보별 특징을 좀 더 자세히 출력합니다.
df.describe()
```





  <div id="df-ae3db51e-ddb7-4d27-baf4-260b84c9ef53">
    <div class="colab-df-container">
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
      <th>pregnant</th>
      <th>plasma</th>
      <th>pressure</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>pedigree</th>
      <th>age</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ae3db51e-ddb7-4d27-baf4-260b84c9ef53')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ae3db51e-ddb7-4d27-baf4-260b84c9ef53 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ae3db51e-ddb7-4d27-baf4-260b84c9ef53');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




각 항목이 어느 정도의 상관 관계를 가지고 있는지 알고 싶다면 다음과 같이 입력합니다.  

🚀 여기서 잠깐!
 - 상관 관계
 - 상관 계수  

의 개념을 확인하고 가시죠  
[네이버 지식백과](https://terms.naver.com/entry.naver?docId=2073705&cid=47324&categoryId=47324)




```python
# 각 항목이 어느 정도의 상관 관계를 가지고 있는지 알아봅니다. 
df.corr()       # 상관계수(correlation) 출력
```





  <div id="df-ea7eda31-3e15-47cc-b5fc-7af40a333ffe">
    <div class="colab-df-container">
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
      <th>pregnant</th>
      <th>plasma</th>
      <th>pressure</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>pedigree</th>
      <th>age</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pregnant</th>
      <td>1.000000</td>
      <td>0.129459</td>
      <td>0.141282</td>
      <td>-0.081672</td>
      <td>-0.073535</td>
      <td>0.017683</td>
      <td>-0.033523</td>
      <td>0.544341</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>plasma</th>
      <td>0.129459</td>
      <td>1.000000</td>
      <td>0.152590</td>
      <td>0.057328</td>
      <td>0.331357</td>
      <td>0.221071</td>
      <td>0.137337</td>
      <td>0.263514</td>
      <td>0.466581</td>
    </tr>
    <tr>
      <th>pressure</th>
      <td>0.141282</td>
      <td>0.152590</td>
      <td>1.000000</td>
      <td>0.207371</td>
      <td>0.088933</td>
      <td>0.281805</td>
      <td>0.041265</td>
      <td>0.239528</td>
      <td>0.065068</td>
    </tr>
    <tr>
      <th>thickness</th>
      <td>-0.081672</td>
      <td>0.057328</td>
      <td>0.207371</td>
      <td>1.000000</td>
      <td>0.436783</td>
      <td>0.392573</td>
      <td>0.183928</td>
      <td>-0.113970</td>
      <td>0.074752</td>
    </tr>
    <tr>
      <th>insulin</th>
      <td>-0.073535</td>
      <td>0.331357</td>
      <td>0.088933</td>
      <td>0.436783</td>
      <td>1.000000</td>
      <td>0.197859</td>
      <td>0.185071</td>
      <td>-0.042163</td>
      <td>0.130548</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.017683</td>
      <td>0.221071</td>
      <td>0.281805</td>
      <td>0.392573</td>
      <td>0.197859</td>
      <td>1.000000</td>
      <td>0.140647</td>
      <td>0.036242</td>
      <td>0.292695</td>
    </tr>
    <tr>
      <th>pedigree</th>
      <td>-0.033523</td>
      <td>0.137337</td>
      <td>0.041265</td>
      <td>0.183928</td>
      <td>0.185071</td>
      <td>0.140647</td>
      <td>1.000000</td>
      <td>0.033561</td>
      <td>0.173844</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.544341</td>
      <td>0.263514</td>
      <td>0.239528</td>
      <td>-0.113970</td>
      <td>-0.042163</td>
      <td>0.036242</td>
      <td>0.033561</td>
      <td>1.000000</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>0.221898</td>
      <td>0.466581</td>
      <td>0.065068</td>
      <td>0.074752</td>
      <td>0.130548</td>
      <td>0.292695</td>
      <td>0.173844</td>
      <td>0.238356</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ea7eda31-3e15-47cc-b5fc-7af40a333ffe')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ea7eda31-3e15-47cc-b5fc-7af40a333ffe button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ea7eda31-3e15-47cc-b5fc-7af40a333ffe');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




조금 더 알아 보기 쉽게 이 상관관계를 그래프로 표현 보겠습니다. 맷플롯립(matplotlib)은 파이썬에서 그래프를 그릴 때 가장 많이 사용되는 라이브러리입니다. 이를 기반으로 조금 더 정교한 그래프를 그리게 해주는 시본(seaborn)라이브러리까지 사요해서 각 정보 간 상관관계를 가시화해 보겠습니다. 


```python
# 데이터 간의 상관 관계를 그래프로 표현해 봅니다.
colormap = plt.cm.gist_heat   # 그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12))   # 그래프의 크기를 정합니다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울수록 밝은색으로 표시되게 합니다.
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()
```

![스크린샷 2022-10-12 오후 9 00 27](https://user-images.githubusercontent.com/87309905/195337218-ca94a034-fa38-4952-ab17-9adc1c49ef4a.png)

    
    


위 그림(그림 11-3)에서 가장 눈여겨 보아야할 부분은 당뇨병 발병 여부를 가리키는 (1)diabetes 항목입니다. diabetes 항목을 보면 pregrnant부터 age까지 상관도가 숫자로 표시되어 있고 숫자가 높을수록 밝은 색으로 채워져 있습니다.

<center>
(그림 11-3) 정보 간 상관관계 그래프<br>
<img src="https://drive.google.com/uc?id=1ETXI_D6icnh7Us7Le-eUvwEUlXMg__HE">
</center>

❓여기서 질문!  
위 상관계수 표를 보고 답하시오. 위에서 고려한 8개 특성 중 당뇨병 발병과 가장 밀접한 관계가 있는 속성은 무엇인가?

## 4. 중요한 데이터 추출하기

그림(11-3)을 살펴보면 plasma 항목(공복 혈당 농도)과 BMI(제질량 지수)가 우리가 예측하고자 하는 diabetes 항목과 상관관계가 높다는 것을 알 수 있다. 즉, 이 항목들이 예측 모델을 만드는데 중요한 역활을 할 것으로 기대할 수 있습니다. 이제 이 두 항목만 따로 떼어 내어 당뇨의 발병 여부와 어떤 관계가 있는지 알아보겠습니다. 

먼저 plasma를 기준으로 각각 정상과 당뇨 여부가 어떻게 분포되는지 살펴봅시다. 다음과 같이 히스토그램을 그려주는 맷플롯립 라이브러리의 hist() 함수를 이용합니다.

데이터 프레임(df)에서 가져오게 될 칼럼을 hist() 함수안에 $x$축으로 지정합니다. df의 plasma 칼럼 중 diabetes 값이 0인 것과 1인 것을 구분해 불러오겠 했습니다. bins는 x 축을 몇 개의 막대 그래프로 쪼개어 보여줄 것인지 정하는 변수입니다. barstacked 옵션은 여러 데이터가 쌓여 있는 형태의 막대바를 생성하는 옵션입니다.추출한 데이터의 라벨을 각각 normal(정상)과 diabetes(당뇨)로 할당했습니다. 


```python
import warnings
warnings.filterwarnings('ignore')

# plasma를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포하는지 살펴봅니다.
plt.hist(x = [df.plasma[df.diabetes == 0], df.plasma[df.diabetes == 1]], bins =30, histtype = 'barstacked', label = ['normal','diabetes'])
plt.legend()
plt.xlabel('plasma')
plt.ylabel('number of sample')
plt.show()
```


![스크린샷 2022-10-12 오후 9 00 11](https://user-images.githubusercontent.com/87309905/195337162-6f79f1a8-5708-4701-b64c-170fe962376f.png)
  
    


위 그래프를 보고 plasma(공복 혈당 농도)가 높을 수록 정상인 샘플 수 보다 당뇨인 샘플의 수가 더 많음을 알 수 있습니다. 이와 같은 방법으로 BMI를 기준으로 각각 정상과 단요가 어느 정도 비율로 분포하는지 살펴보겠습니다.


```python
# BMI를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포하는지 살펴봅니다.
plt.hist(x = [df.bmi[df.diabetes == 0], df.bmi[df.diabetes == 1]], bins = 30, histtype = 'barstacked',label=['normal','diabetes'])
plt.legend()
plt.xlabel('BMI')
plt.ylabel('number')
plt.show()
```

![스크린샷 2022-10-12 오후 8 59 58](https://user-images.githubusercontent.com/87309905/195337121-49196ac5-2a83-4e69-bcf2-c1a59d1d0f20.png)

    


<br>
<center>
<img src = "https://drive.google.com/uc?id=1TOD17dScr_FKzjUc3ULLnWRkGJFC3LtJ">
</center>

앞서서, BMI가 23부터 과제충이고 25부터는 비만으로 평가된다는 것을 확인했습니다. BMI에 따른 과체중부터 비만인 경우 당뇨 발병 샘플이 정상인 샘플 보다 많다는 것을 확인 할 수 있습니다. 그리고 분석하고 있는 데이터에 나타난 결과를 보면 저체중인 경우에도 당뇨 발병 샘플이 정상 샘플보다 조금 더 많은 것으로 나타났습니다.

이렇게 결과에 미치는 영향이 큰 항목을 찾는 것이 **데이터 전처리** 과정 중 하나입니다. 이 밖에도 데이터가 누락된 곳이 있다면 평균이나 중앙값으로 대치하거나, 흐름에서 크게 벗어나는 이상한 값을 제거하는 과정 등이 데이터 전처리에 포함될 수 있습니다.

## 5. 피마 인디언 당뇨병 예측 실행
이제 텐서플로의 케라스를 이용해서 예측을 실행해 봅시다. 판다스 라이브러리를 사용하기 때문에 iloc() 함수를 사용해 속성 $X$와 클래스 $y$를 각각 저장합니다. iloc는 데이터 프레임에서 대괄호 안에 정한 범위만큼 요소를 가져와 저장하게 합니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas 라이브러리를 불러옵니다.
import pandas as pd

# 깃허브에 준비된 데이터를 가져옵니다. 이미 앞에서 가져왔으므로 주석 처리합니다. 5번 예제만 별도 실행 시 주석 해제 후 실습하세요.
!git clone https://github.com/taehojo/data.git

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 
df = pd.read_csv('./data/pima-indians-diabetes3.csv')

```

    fatal: destination path 'data' already exists and is not an empty directory.



```python
# 세부 정보를 X로 지정합니다.
X = df.iloc[:, 0:8]

# 당뇨병 여부를 y로 지정합니다.
y = df.iloc[:,8]
```

다음과 같이 모델 구조를 설정합니다. 이전과 달라진 점은 은닉층이 하나 더 추가되었다는 것입니다. 그리고 층과 층의 연결을 한 누에 볼 수 있게 해주는 model.summary()이 추가 되었습니다. model.summary()의 실행 결과는 다음과 같습니다.


```python
# 모델을 설정합니다.
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu', name = 'Dense_1'))
model.add(Dense(8, activation = 'relu', name ='Dense_2'))
model.add(Dense(1, activation = 'sigmoid', name = 'Dense_3'))
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Dense_1 (Dense)             (None, 12)                108       
                                                                     
     Dense_2 (Dense)             (None, 8)                 104       
                                                                     
     Dense_3 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 221
    Trainable params: 221
    Non-trainable params: 0
    _________________________________________________________________


<br><center>
<img src="https://drive.google.com/uc?id=1rWlrqvQAbVzfopA4juXsDK_a_iPHoc0S" width=500>
</center><br>

(1) Layer 부분은 층의 이름과 유형을 나타낸다. 각 층의 이름은 자동으로 정해지는데, 따로 이름을 만들려면 Dense() 함수 안에 name="층 이름"을 추가해 주면 됩니다. 입력층과 첫 번째 은닉층을 연결해 주는 Dense_1층, 첫 번째 은닉층과 두 번째 은닉층을 연결하는 Dense_2층 그리고 두 번째 은닉층과 출력층을 연결하는 Dense_3층이 만들어졌음을 알 수 있습니다. 

(2) Output Shape 부분은 각 층에 몇 개의 출력이 발행하는지 나타냅니다. 쉽표(,)를 사이에 두고 괄호의 앞은 행(샘플)의 수, 뒤는 열(속성)의 수를 의미합니다. 행의 수는 batch_size에 정의한 만큼 입력되므로 딥러닝 모델에서는 이를 특별히 세지 않습니다. 따라서 괄호의 앞은 None으로 표시됩니다. 여덟 개의 입력이 첫 번째 은닉층을 지나며 12개 출력이 되고 두 번째 은닉층을 지나며 여덟 개가 되었다가 출력층에서는 한 개의 출력을 만든다는 것을 보여주고 있습니다. 

(3) Param 부분은 파라미터 수, 즉 총 가중치와 바이어스 수의 합을 나타냅니다. 예를 들어 첫 번째 층의 경우 입력 값 8개가 층안에서 12개의 노드로 분산되므로 가추치가 8x12=96개가 되고 각 노드에 바이어스가 한 개씩 있으니 전체 파라미터 수는 96 + 12 = 108이 됩니다.

(4) 부분은 전체 파라미터를 합산한 값입니다. Trainable params는 학습을 진행하면서 업데이트가 된 파라미터들이고, Non-trainable params는 업데이트가 되지 않은 파라미터의 수를 나타냅니다. 

이 모델을 그림으로 표현하면 그림(11-6)과 같습니다.
<br><center>
(그림. 11-6) 피마 인디언 당뇨병 예측 모델의 구조
<img src="https://drive.google.com/uc?id=1yXLXLw6v6guSf2uBoKGIIp1gms_0tC86">
</center><br>  

$✅ \color{red} {\text{은닉층의 개수를 왜 두개로 했나요? 그리고 노드의 수를 왜 각각 12개와 8개로 했나요?} }$

입력과 출력의 수는 정해져 있지만, 은닉층은 몇 층으로 할지, 은닉층 안의 노드는 몇 개로 할지에 대한 정답은 없습니다. 자신의 프로젝트에 따라 설정해야 합니다. 여러 번 반복하면서 최적값을 찾아내는 것이 좋으며, 여기서는 읨의 수로 12와 8을 설정했고 설명의 편의성을 위해 두 개의 은닉층을 만들었습니다. 여러분이 직접 노드의 수와 은닉층의 개수를 바꾸어 보면서 더 높은 저확도가 나오는지 테스트해보세요.





```python
# 모델을 컴파일합니다.
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])

# 모델을 실행합니다.
history = model.fit(X, y, epochs = 100, batch_size = 5)
```

    Epoch 1/100
    154/154 [==============================] - 2s 4ms/step - loss: 2.2396 - accuracy: 0.5013
    Epoch 2/100
    154/154 [==============================] - 1s 4ms/step - loss: 0.9221 - accuracy: 0.5495
    ...
    154/154 [==============================] - 0s 2ms/step - loss: 0.5011 - accuracy: 0.7513
    Epoch 98/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5097 - accuracy: 0.7565
    Epoch 99/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5135 - accuracy: 0.7669
    Epoch 100/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5103 - accuracy: 0.7539


### [ 과제 ]
상관도가 낮은 속성 2개를 제외하고 학습을 시켜보십시오. 위 모델(위 설정과 동일)의 예측 결과는 어떻게 달라지는가?


```python
df
```





  <div id="df-8aabc06f-a93f-49fe-bef1-a7743abbc07c">
    <div class="colab-df-container">
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
      <th>pregnant</th>
      <th>plasma</th>
      <th>pressure</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>pedigree</th>
      <th>age</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8aabc06f-a93f-49fe-bef1-a7743abbc07c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8aabc06f-a93f-49fe-bef1-a7743abbc07c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8aabc06f-a93f-49fe-bef1-a7743abbc07c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test = df.drop(['pressure','thickness'], axis = 1 )
test

X_test = test.iloc[:,0:6]
y_test = test.iloc[:,6]

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid',name='Dense_3'))
model.summary()

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X_test, y_test, epochs=100, batch_size=5)
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Dense_1 (Dense)             (None, 12)                84        
                                                                     
     Dense_2 (Dense)             (None, 8)                 104       
                                                                     
     Dense_3 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 197
    Trainable params: 197
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    154/154 [==============================] - 1s 2ms/step - loss: 5.4550 - accuracy: 0.5599
    Epoch 2/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.8785 - accuracy: 0.6120
    Epoch 3/100
    ...
    154/154 [==============================] - 0s 2ms/step - loss: 0.5333 - accuracy: 0.7461
    Epoch 98/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5234 - accuracy: 0.7383
    Epoch 99/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5302 - accuracy: 0.7500
    Epoch 100/100
    154/154 [==============================] - 0s 2ms/step - loss: 0.5214 - accuracy: 0.7396


***

# 12장 다중 분류 문제 해결하기
## 1. 다중 분류 문제
아리리스는 그 꽃봉오리가 마치 먹물을 머금은 붓과 같다 하여 우리나라에서는 '붓꽃'이라고 불리는 아름다운 꽃입니다. 아이리스는 꽃잎의 모양과 기링에 따라 여러 가지 품종으로 나뉩니다. 사진을 보면 서로 비슷해 보입니다. 과연 딥러닝을 사용하여 이들을 구별해 낼 수 있을까요?
<br><center>
(그림. 12-1) 아이리스의 품종<br>
<img src="https://drive.google.com/uc?id=10IA-RltFVLEn2BSsPRfR5nnDsRcqJJCe"><br>
<img src="https://drive.google.com/uc?id=1U8SR_6E9uuVO0vodKIg3l1zEOd2giP-z">
</center><br>  

아이리스 품종 예측 데이터는 예제 파일의 data 폴더에서 찾을 수 있습니다(data/iris3.csv). 데이터 구조는 다음과 같습니다.  
<br><center>
(그림. 12-2) 아이리스 데이터의 샘플, 속성, 클래스 구분<br>
<img src="https://drive.google.com/uc?id=1EEgYhS0DXYRtkKb5NXRzQXrosazQpZmq">
</center><br>  

* 샘플 수 : 150
* 속성의 수
 - 정보 1 : 꽃받침 길이(sepal length, 단위: cm)
 - 정보 2 : 꽃받침 너비(sepal width, 단위: cm)
 - 정보 3 : 꽃잎 길이(petal length, 단위: cm)
 - 정보 4 : 꽃잎 너비(petal width, 단위: cm)
* 클래스: iris-setosa, Iris-versicolor, Iris-virginica

속성을 보니 우리가 앞서 다루었던 데이터셋과 눈에 띠는 차이가 있습니다. 바로 클래스가 두 개가 아니라 세 개입니다. 즉, 참(1)과 거짓(0)을 해결하는 것이 아니라, 여러 개 중에 어떤 것이 답인지 예측하는 문제입니다. 

이렇게 여러 개의 중에 하나로 분류하는 것을 **다중 분류**(multi classification)라고 합니다. 다중 분류 문제는 둘 중에 하나를 고르는 이항 분류(binary classification)와는 접근 방식이 조금 다릅니다. 지금부터 아이리스 품종을 예측하는 실습을 통해 다중 분류 문제를 해결해 보겠습니다. 



### 2. 상관도 그래프

먼저 데이터를 불러와 데이터셋의 일부를 확인하겠습니다.

데이터 프레임에 들어 있는 데이터셋(샘플들)은 같은 품종끼리 정렬이 되어 있습니다. 품종(species)에 따른 샘플이 몇 개씩 있는지 확인해 보겠습니다.

데이터셋에 setosa, versicolor, virginica 각각 50개 샘플이 저장되어 있는 것을 확인했습니다. 이제 상관도 그래프를 그려보겠습니다. 시본(seaborn) 라이브러리에 있는 pairplot() 함수를 써서 전체 상관도를 제시합니다.

위 상관도를 통해 각 속성별 데이터 분포와 속성 간의 관계를 한눈에 볼 수 있습니다. pairplot() 함수 설정 중 hue 옵션은 주어진 데이터 중 어떤 카테고리를 중심으로 그래프를 그릴지 정해 주게 되는데, 우리는 품종(species)에 따라 보여지게끔 지정했습니다.
```
sns.pairplot(df, hue='species')
```
그래프 각가의 가로추과 세로축은 서로 다른 속성을 나타내며, 이러한 속성에 따라 품종이 어떻게 분포하는지 보여 줍니다. 이렇한 분석을 통해 시진 상으로 비슷해 보이던 꽃잎과 꽃받침의 크기와 너비가 품종별로 어떤 차이가 있는지 알 수 있습니다. 

예를 들어, 위 그림의 맨 왼쪽 맨 아래 그래프를 보겠습니다. 이 그래프에서 파란색 원형 점과 주황색 원형 점 그리고 녹생 원형 점이 표시되어 세 품종에 대한 정보를 담고 있습니다. 파란색은 원형 점은 setosa 품종에 대한 정보를, 주황색 점들은 versicolor 품종에 대한 정보를 그리고 녹색 점들은 viriginica 품종에 대한 정보를 보이고 있습니다. 파란색 점들은 setosa 품종의 분꽃들의 꽃받침(sepal)의 길이와 꽃잎(petal)의 너비와의 상관관계를 보이고 있습니다. 꽃받침(sepal)의 길이 변화 대비 꽃잎(petal)의 너비 변화가 크지 않다고 판단됩니다. 

위 서브 그래프 중, setosa 붓꽃의 경우 꽃받침의 폭(sepal width) 변화 대비 꽃잎의 길이(petal length) 변화가 그리 크지 않다 또는 관계가 깊지 않다고 판단할 수 있는 정보를 주는 그래프는 어떤 것일까요?  

<br><center>
상과 관계가 매우 낮음(없음)<br>
<img src="https://drive.google.com/uc?id=1SCa732olmGpw-Nhfv0YC1mhbZyFS7xio">
</center><br>  



## 3. 원-핫 인코딩
이제 케라스를 이용해 아이리스의 품종을 예측해 보겠습니다. 먼저 데이터 프레임(df)를 X와 y로 나누겠습니다.

$y$의 값이 숫자가 아닌 문자입니다. 품종(species)에 대한 정보가 "Iris-setosa" 등 문자로 되어 있기 때문이네요. 원활한 계산을 위해 문자를 숫자형으로 바꾸어 표현하겠습니다.  

아이리스 꽃의 품종은 세 가지 입니다. 그러면 각각의 이름으로 세 개의 열을 만든 후 품종이 일치하는 경우 1로 아닌 경우는 0으로 변경합니다.
<br><center>
원-핫 인코딩<br>
<img src="https://drive.google.com/uc?id=1x3dOjXtJG7ZquDfVlEXT27k_EjkoNMy2" width=500>
</center><br>  

이렇게 여러 개의 값으로 된 문자열을 0과 1로만 이루어진 형태로 만들어 주는 과정을 **원-핫 인코딩**(one-hot encoding)이라고 합니다. 원-핫 인코딩은 판다스가 제공하는 get_dummies() 함수를 사용하면 간단하게 만들 수 있습니다. 


## 4. 소프트맥스

이제 모델을 만들어 줄 차례입니다. 다음 코드를 보면서 이전에 실행했던 피마 인디언의 당뇨병 예측과 무엇이 달라졌는지 찾아 보세요.


```python
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # 모델 설정
# model = Sequential()
# model.add(Dense(12,  input_dim=4, activation='relu'))
# model.add(Dense(8,  activation='relu'))
# model.add(Dense(3, activation='softmax'))
# model.summary()

# # 모델 컴파일
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 모델 실행
# history = model.fit(X, y, epochs=30, batch_size=5)
```

적확도가 무려 약 97%가 나왔네요. 이전에 실행했던 피마 인디언의 당뇨병 예측과 무엇이 달라졌는지 찾아 보셨습니까? 세 가지가 달라졌습니다. 출력 층의 노드 수가 3으로 변경되었습니다. 그리고 활성화 함수가 softmax로 변경되었습니다. 또 컴파일 부분에서 손실 함수가 categorical_crossentropy로 바뀌었습니다. 

먼저 출력 부분에 대해 알아보겠습니다. 이전까지 우리는 출력이 0\~1 사이의 한 값으로 나왔습니다. 예를 들어 당뇨인지 아니지에 대한 예측 값이 시그로임드 함수를 거쳐 0\~1 사이의 값 중 하나로 변환되어 0.5 이상이면 당뇨로, 이하면 정상으로 판단했지요. 그런데 이번 예제에서는 예측해야할 값이 세 가지 늘었습니다. 즉, 각 샘플마다 이것이 setosa일 확율, versicolor일 확율 그리고 virginica일 확율을 따로따로 구해야 한다는 것이지요. 예를 들어 예측 결과는 그림(12-5)또는 표 (12-1)와 같은 형태로 나타납니다.  

<br><center>
(그림 12-5) 소프트 맥스<br>
<img src = "https://drive.google.com/uc?id=1cfHTDo5ZnsxIODeYyS3nHv3NQtSxxhp-"><br>

(표. 12-1) 소프트맥스<br>  

|setona일 확율|versicolor일 확율|virginca일 확율|
|---|---|---|
|0.2|0.7|0.1|
|0.8|0.1|0.1|
|0.2|02.|0.6|

</center><br>

이렇게 세 가지의 확율을 모두 구해야 하므로 시그모이드 함수가 아닌 다른 함수가 필요합니다. 이때 사용되는 함수가 바로 소프트맥스 함수입니다. 소프트 맥스 함수는 표(12-1)과 같이 각 항목당 예측 확율이 0과 1 사이의 값으로 나타내 주는데, 이때 각 샘플당 예측 확율의 총합이 1인 형태로 바꾸어 주게 됩니다. 

마찬가지로 손실 함수도 이전과 달라졌습니다. 이항 분류에서 binary_crossentropy를 썼다면, 다항 분류에서는 categorical_crossentropy를 쓰면 됩니다. 

## 5. 아이리스 품족 예측 실행

```model.summary()```를 사용해서 두 개의 은닉층에 각각 12개와 여덟 개의 노드가 만들어졌고 출력은 세 개임을 확인할 수 있습니다. 결과는 30번 반복했을 정확도가 약 96%(실행 시킬 때 마다 차이가 발생할 수 있음) 나왔습니다. 꽃의 너비와 길이를 담은 150개의 데이터 중 144개의 꽃 종류를 맞추었다는 의미입니다. 이제부터는 이렇게 측정된 정확도를 어떻게 신뢰할 수 있을지, 예측 결과의 신뢰도를 높이는 방법에 대해 알아보겠습니다. 

***

# 13장 모델의 성능 검증하기

1986년 제프리 힌튼 교수가 오차 역전파를 발표한 직후, 존스 홉킨스의 세즈노프스키(Sejnowski) 교수는 오차 역전파가 은닉층의 가중치를 실제로 업데이트시키는 것을 확인하고 싶었습니다. 그는 **광석과 일방 암석에 수중 음파 탐지기를 쏜 후 결과를 모아 데이터셋을 준비했고 음파 탐지기의 수신 결과만 보고 광석인지 일반 암석인지를 구부하는 모델을 만들었습니다.** 그가 측정한 결과의 정확도는 얼마였을까요?

<br><center>
<img src="https://drive.google.com/uc?id=1G-cPV2ET-IrDbdzQFh430UCMz7CHuPpL" width=400>
</center><br>


이 장에서는 세즈노프스키 교수가 했던 초음파 광물 예측 실험을 텐서플로로 재현해보고 이렇게 구해진 실험 정확도를 평가하는 방법과 성능을 향상시키는 중요한 머신 러닝 기법들에 대해 알아보겠습니다.

##  1. 데이터의 확인과 예측 실행


```python
import pandas as pd

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 광물 데이터를 불러옵니다.
df = pd.read_csv('./data/sonar3.csv', header=None)

# 첫 5줄을 봅니다. 
df.head()
```

    fatal: destination path 'data' already exists and is not an empty directory.






  <div id="df-cc751462-fd11-4abb-a81f-8a90049cff6f">
    <div class="colab-df-container">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cc751462-fd11-4abb-a81f-8a90049cff6f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cc751462-fd11-4abb-a81f-8a90049cff6f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cc751462-fd11-4abb-a81f-8a90049cff6f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




첫번째 열(0)부터 60번째(59)열까지는 음파의 에너지를 0에서 1 사이의 숫자로 표시하고 있습니다. 이제 일반 암서과 광석이 각각 몇 개나 데이터셋에 포함되어 있는지 확인해보겠습니다.


```python
# 일반 암석(0)과 광석(1)이 몇 개 있는지 확인합니다.
df[60].value_counts()
```




    1    111
    0     97
    Name: 60, dtype: int64



광석 샘플이 111개 암석 샘플이 97개 따라서 샘플 수는 총 111+97= 208개의 샘플이 데이터 셋을 구성하고 있습니다. 

1\~60번째(0~59) 열을 변수 $X$에 저장하고 광물의 종류를 $y$로 표현하겠습니다.


```python
# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:, 0:60]
y = df.iloc[:,60]
```

이후 앞서 했던 그대로 딥러닝을 실행하겠습니다. 출력 $y$는 하나이며 은닉층...


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X, y, epochs=200, batch_size=10)
```

    Epoch 1/200
    21/21 [==============================] - 1s 2ms/step - loss: 0.6970 - accuracy: 0.5192
    Epoch 2/200
    21/21 [==============================] - 0s 2ms/step - loss: 0.6864 - accuracy: 0.5337
    Epoch 3/200
    ...
    21/21 [==============================] - 0s 2ms/step - loss: 0.0570 - accuracy: 0.9952
    Epoch 199/200
    21/21 [==============================] - 0s 2ms/step - loss: 0.0576 - accuracy: 0.9904
    Epoch 200/200
    21/21 [==============================] - 0s 2ms/step - loss: 0.0570 - accuracy: 0.9904


200번 반복되었을 때의 결과를 보니 정확도가 100%입니다. 이 모델의 예측 정확도가 100%라는 것을 믿을 수 있습니까? 정말로 광석인지 일반 암석인지  100%의 확율로 판별해 내는 모델이 만들어진 것이라요? 다음 섹션에서 이 의문에 대한 답을 찾아 보겠습니다. 

## 2. 과적합 이해하기
이제 과적합 문제가 무엇인지 알아보고 이를 어떻게 해결하는지 살펴보겠습니다. 과적합(overfitting)이란 모델이 학습 데이터셋 안에서는 일정 수준 이상의 예측 정보를 보이지만 새로운 데이터에 적용하면 잘 맞지 않는 것을 의미합니다. 

그림(13-1)의 그래프에는 두 종류의 데이터가 있습니다. 원 안이 검은 색인 것과 원 안이 흰색인 두 종류의 데이터가 보입니다. 이 두 종류를 완벽하게 또는 매우 정확하게 분류하기 위해 구분 선을 심한 곡선으로 그린 것(과적합)과 그 외 두 가지의 직선 행태의 구분선이 보입니다.   
<br><center>
(그림. 13-1) 과적합<br>
<img src="https://drive.google.com/uc?id=1bPEOjpG1MqG1eF_WSZjHwDUpczsAnAU2" width=500>
</center><br>  

과적합 구분 선은 주어진 샘플들에만 최적화 되어 있습니다. 새로운 데이터가 주어졌을 때 과적합된, 그러니까 기존 데이터에 overfitting된 구분선으로 새로운 데이터를 정확히 불류하기 어렵다는 것입니다. 

과적합(overfitting)은 층이 너무 많거나 변수가 복자해서 살생하기도 하고 테스트셋과 학습셋이 중복될 때 생기기도 합니다. 특히 딥러닝은 학습 단계에서 입력층, 은닉층, 출력층의 노드들에 상당히 많은 변수가 투입됩니다. 따라서 딥러닝을 진행하는 동안 과적합에 빠지 않게 늘 주의해야 합니다. 


## 3. 학습셋과 테스트셋
그렇다면 과적합을 방지하려면 어떻게 해야할까요? 먼저 학습을 하는 데이터셋과 이를 테스트할 데이터셋을 완전히 구분한 후 학습과 동시에 테스트를 변행하며 진행하는 것이 한 방법입니다. 예를 들어 데이터세이 총 100개의 샘플로 이루어져 있다면 다음과 같이 두 개의 셋으로 나눕니다. 

<br><center>
(그림. 13-2) 학습셋과 테스트셋 구분<br>
<img src="https://drive.google.com/uc?id=1MBgSLj0CodjyAHGZx4o4EIvKDFLolpM1">
</center><br>

(전체 데이터셋의 70\~75%를 학습셋으로 사용하고 30\~25%를 테스트 셋으로 나눕니다.) 신경망을 만들어 70개의 샘플로 학습을 진행한 후 이 학습의 결과를 저장합니다. 이렇게 저당된 파일을 '모델'이라고 합니다. 모델은 다른 셋에 적용할 경우 학습 단계에서 각인되었던 그대로 다시 수행합니다. 따라서 나머지 30개의 샘플로 테스트해서 정확도를 살펴보면 학습이 얼마나 잘 되었는지 알 수 있을 것입니다. 딥러닝 같은 알고리즘을 충분히 조절해 가장 나은 모델이 만들어지면 이를 실생활에 대입해 활용하는 것이 바로 머신 러닝의 개발 순서입니다.  

<br><center>
(그림. 13-3) 학습셋과 테스트셋<br>
<img src="https://drive.google.com/uc?id=1BOHkL4fJVuVrCdIpfC9P02t-TUB8_f92" width=400>
</center><br>  
지금까지 우리는 테스트셋을 만들지 않고 모든 데이터셋을 이용해 학습시켰습니다. 그런데로 매번 정확도(accuracy)를 계산할 수 있었지요. 어떻게 가능했을까요?  

**"지금까지 학습데이터를 이용해 정확도를 측정한 것은 데이터셋에 들어있는 모든 샘플을 그대로 테스트에 활용한 결과입니다." → 지금까지 모든 데이터셋을 이용해 학습했고 모든 데이터셋(데이터 샘플)을 가지고 테스트한 결과입니다.** 

이를 통해 학습이 진행되는 상황을 파악할 수는 있지만 새로운 데이터에 적용했을 때 어느 정도의 성능이 나올지 알수 없습니다. 머신 러닝의 최종 목적은 과거의 데이터를 토대로 새로운 데이터를 예측하는 것입니다. 즉, 새로운 데이터에 사용할 모델을 만드는 것이 최종 목적이므로 테스트셋을 만들어 정확한 평가를 병행하는 것이 매우 중요합니다. 

학습셋만 가지고 평가할 때, 층을 더하거나 에포크(epoch) 값을 높여 실행 횟수를르리면 정확도가 계소해서 올라갈 수 있습니다. 하지만 학습 데이터셋만으로 평가한 예측 성공률이 테스트셋에서도 그래도 나나타지는 않습니다. 즉, 학습이 깊어져 학습셋 내부에서 성공률은 높아져도 테스트셋에서는 효과가 없다면 과적합이 일어난 것이지요. 이를 그래프로 표현하면 그림(13-4)와 같습니다.

<br><center>
(그림. 13-4) 학습이 계속되면 학습셋에서는 에러는 계속해서 작아져 과적합 발생<br>
<img src="https://drive.google.com/uc?id=1ZrnHjrDC6TXjPww9R5JFyYrwHrDyNIME" width=400>
</center><br>  

학습을 진행해도 테스트 결과가 더 이상 좋아지지 않는 시점에서 학습을 멈춰야 합니다. 이때 학습 정도가 가장 적절한 것으로 볼 수 있습니다. 

우리가 다루는 초음파 광물 예측 모델을 만든 세즈노프스키 교수가 실험 결과를 발표한 논무의 일부를 가져와 보겠습니다. 

<br><center>
(그림. 13-5) 학습셋과 테스트셋 정확도 측정의 예(RP Gorman et.al., 1998)<br>
<img src="https://drive.google.com/uc?id=1pnJsvvIZ_RJwDhyQ63nGsgOIKofvPv9L">
</center><br>  

여기서 눈여겨보아야 할 부분은 은닉층(Number of Hidden Units) 개수가 올라감에 따라 학습셋의 예측율(Average Performacne on Training Sets)과 데스트셋의 예측률(Average Performance on Testing Sets)이 어떻게 변하는지입니다. 이 부분만 따로  뽑아서 정리하면 표(13-2)와 같습니다. 

<br><center>
(표. 13-2) 은닉층 개수의 변화에 따른 학습셋의 예측률<br>

|은닉층 개수|학습셋의 예측률|테스트셋의 예측율|
|:---:|---:|---:|
|0|79.3|73.1|
|2|96.2|85.7|
|3|98.1|87.6|
|6|99.4|89.3|
|12|99.8|90.4|
|24|100|89.2|

</center><br>  

은닉층이 늘어날수록 학습셋의 에측률이 점점 올라가다가 결국 24개 층에 이르면 100% 예측률을 보입니다. 우리가 조금 전에 실행했던 결과와 같습니다. 그런데 이 모델을 토대로 테스트한 결과는 어떤가요? 테스트셋 예측률은 은닉층의 개수가 12개일 때 90.4%로 최고를 이루다 24개째에서는 다시 89.2%로 떨어지고 맙니다. 즉, 식이 복잡해지고 학습량이 늘어날수록 학습 데이터를 통한 예측률은 계속해서 올라가지만 은닉층의 수를 적절하게 조절하지 않을 경우 테스트셋을 이용한 예측률은 오히려 떨어지는 것을 확인할 수 있습니다. 

그러면 예제에 주어진 데이터를 학습 데이터셋과 테스트셋으로 나누는 예제를 만들어 보겠습니다. 

🚀 여기서 잠깐.  
이 실습에서는 사이킷런(scikit-learn) 라이브러리가 필요합니다.

저장된 $X$ 데이터와 $y$ 데이터에서 각각 정해진 비율(%)만큼 학습 데이터셋과 테스트 데이터셋으로 분리시키는 함수가 사이킷런의 ```train_test_split()```함수입니다. 따라서 다음과 같이 학습 데이터셋과 테스트 데이터셋을 만들 수 있습니다. 총 데이터셋에서 학습 데이터셋을 70%, 테스트 데이터셋을 30%로 나눌 때의 코드  
예입니다. 
```
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)
```
위 코드에서 ```test_size```은 테스트 데이터셋의 비율입니다. 0.3은 전체 데이터셋의 30%를 테스트 데이터셋으로 사용하겠다는 것으로 나머지 70%를 학습 데이터셋으로 사용하게 됩니다. 이렇게 나누어진 학습 데이터셋과 테스트 데이터셋으로 각각 ```X_train, y_train``` 그리고 ```X_test, y_test```에 저장됩니다. 

모델은 앞서 만든 구조를 그대로 유지하고 모델에 성능 평가를 위해 테스트 함수(```model.evaluate()```)를 추가했습니다. 

```
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```
```model.evaluate()``` 함수는 loss와 accuracy 두 가지를 계산해 출력합니다. 이를 score에 저장하고 accuracy를 출력하도록 했습니다. 

이제 전체 코드를 실행해 보겠습니다.





```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd
```


```python
# 깃허브에 준비된 데이터를 가져옵니다. 앞에서 이미 데이터를 가져왔으므로 추석 처리합니다. 3번 예제만 별도 실행 시 주석을 해제하여 실습하세요.
!git clone https://github.com/taehojo/data.git
```

    fatal: destination path 'data' already exists and is not an empty directory.



```python
# 광물 데이터를 불러옵니다.
df = pd.read_csv('./data/sonar3.csv', header = None)
```


```python
# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:,0:60]
y = df.iloc[:,60]
```


```python
# 학습셋과 테스트셋을 구분합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, shuffle = True)
```


```python
# 모델을 설정합니다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X_train, y_train, epochs=200, batch_size=10)
```

    Epoch 1/200
    7/7 [==============================] - 1s 3ms/step - loss: 0.6952 - accuracy: 0.5323
    Epoch 2/200
    7/7 [==============================] - 0s 2ms/step - loss: 0.6818 - accuracy: 0.5484
    Epoch 3/200
    7/7 [==============================] - 0s 2ms/step - loss: 0.6713 - accuracy: 0.5484
    ...
    7/7 [==============================] - 0s 3ms/step - loss: 0.0248 - accuracy: 1.0000
    Epoch 199/200
    7/7 [==============================] - 0s 3ms/step - loss: 0.0246 - accuracy: 1.0000
    Epoch 200/200
    7/7 [==============================] - 0s 3ms/step - loss: 0.0239 - accuracy: 1.0000



```python
# 모델을 테스트셋에 적용해 정확도를 구합니다.
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])
```

    5/5 [==============================] - 0s 3ms/step - loss: 0.6374 - accuracy: 0.7740
    Test accuracy:  0.7739726305007935


학습 데이터셋(X_train과 y_train)을 이용해 200번의 학습을 진행했을 때 모델이 판단한 학습 데이터셋에 대한 정확도와 생성된 모델을 테스트셋을 적용했을 때 보인 정확도가 다르다는 것입니다. 테스트 데이터셋을 적용했을 때의 정확도가 학습 데이터셋를 활용해 생성한 모델이 학습 데이터셋를 가지고 판단한 정확도 보다 낮습니다. 


머신러인, 딥러닝의 목표는 학습 데이터셋에서만 잘 동작하는 모델을 만드는 것이 아니라. 새로운 데이터에 대해 높은 정확도를 안정되게 보여주는 모델을 만드는 것이 목표입니다. 어떻게 하면 그러한 모델을 만들 수 있을까요? 모델 성능의 향상을 위한 방법에는 크게 데이터를 보강하는 방법과 알고리즘을 최적화 하는 방법이 있습니다. 

데이터를 이용해 성능을 향상시키려면 우선 충분한 데이터를 가져와 구가하면 됩니다. 많이 알려진 아래 그래프는 특히 딥러닝의 경우 샘플 수가 많을 수록 성능이 좋아짐을 보여줍니다.  

<br><center>
(그림. 13-6) 데이터의 증가와 딥러닝, 머신러닝 성능의 상관관계<br>
<img src="https://drive.google.com/uc?id=1WELFk7zpYiBYWmMnd7eJ7tlGXR36xrTL" width=400>
</center><br>  

하지만 데이터를 추가하는 것 자체가 어렵거나 데이터 추가만으로 성능 개선에 한계가 있을 수 있습니다. 딸서 가지고 있는 데이터를 적절히 보완해 주는 방법을 사용합니다. 예를 들어 사진의 경우 크기를 확대/축소한 것을 데이터 셋에 추가해 보거나 위 아래로 조금씩 움직인 사진을 데이터셋에 추가하는 것입니다.(이 내용은 20장에서 다룹니다) 테이블형 테이터의 경우 너무 크거나 낮은 이상치가 모델에 영향을 줄 수 없도록 크기를 적절히 조절할 수 있습니다. 시그모이드 함수를 사용해 전체를 0~1사이의 값으로 변환하는 것이 좋은 예입니다. 또 교차 검증 방법을 사용해서 가지고 있는 데이터를 충분히 이용하는 방법도 있습니다. 이는 잠시 후에 설명할 것입니다. 

다음으로 알고리즘을 이용해 성능을 향상하는 방법은 먼저 다른 구조로 모델을 바꾸어 가며 최적의 구조를 찾는 것입니다. 예를 들어 은닉층의 개수라든지, 그 안에 들어갈 노드의 수, 최적화 함수의 종류를 바꾸어 보는 것입니다. 앞서 이야기한 바 있지만 딥러닝 설정에 정답은 없습니다. 자신의 상황에 맞는 구조를 계속해서 테스트 해보며 찾는 것이 중요합니다. 그리고 데이터에 따라서는 딥러닝이 아닌 랜덤 포레스트, XGBoost, SVM 등 다른 알고리즘이 더 좋은 결과를 보일 때도 있습니다. 일반적인 머신 러닝과 딥러닝을 합해서 더 좋은 결과를 만드는 것도 가능하지요. 많은 경험을 통해 최적의 성능을 보이는 모델을 만드는 것이 중요합니다. 

**이제 현재 모델을 저장하고 불러오는 방법에 대해 알아보겠습니다.**

## 4. 모델 저장과 재사용
학습이 끝난 후 지금 만든 모델을 저장하면 언제든지 이를 불러와 다시 사용할 수 있습니다. 학습 결과를 저장하려면 ```model.save()```함수를 이용해 모델을 저장할 수 있습니다.


```python
# 모델 이름과 저장할 위치를 함께 지정합니다.
model.save('./data/model/my_model.hdf5')
```

hdf5 파일 포멧은 주로 과학 기술 데이터 작업에서 사용되는데, 크고 복잡한 데이터를 저장하는데 사용됩니다. 이를 다시 불러오려면 케라스 API의 ```load_model()```함수를 사용합니다. 앞서 ```Sequential()``` 함수를 불러온 모델 클래스 안에 함께 들어 있으므로 ```Sequential``` 뒤에 ```load_model```을 추가합니다.


```python
from tensorflow.keras.models import Sequential, load_model
```

좀전에 만든 모델을 메모리에서 삭제하겠습니다


```python
# 테스트를 위해 조금 전 사용한 모델을 메모리에서 삭제합니다.
del model
```

저장된 모델을 불러옵니다.


```python
# 모델을 새로 불러옵니다.
model = load_model('./data/model/my_model.hdf5') 

# 불러온 모델을 테스트셋에 적용해 정확도를 구합니다. 
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```

    5/5 [==============================] - 0s 3ms/step - loss: 0.6374 - accuracy: 0.7740
    Test accuracy: 0.7739726305007935


테스트 데이터셋을 가지고 정확도 검사를 다시 해봤습니다. 이전과 같은 결과를 얻은 것을 확인할 수 있습니다.

### [과제]
과제 1 - 데이터셋의 65%를 학습 데이터셋으로하고 35%를 데트트 데이터셋으로 나누어서 위 과정을 수행하십시요. 생성된 모델의 학습데이터셋에 대한 정확도와 테스트 데이터셋에 대한 정확도를 제시하십시요.

과제 2 - 데이터셋의 80%를 학습 데이터셋으로하고 20%를 데트트 데이터셋으로 나누어서 위 과정을 수행하십시요. 생성된 모델의 학습데이터셋에 대한 정확도와 테스트 데이터셋에 대한 정확도를 제시하십시요.



```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd

# 광물 데이터를 불러옵니다.
df = pd.read_csv('./data/sonar3.csv', header=None)
# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:,0:60]
y = df.iloc[:,60]

# 학습셋과 테스트셋을 구분합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True)

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X_train, y_train, epochs=200, batch_size=10)

print('---'*24)
# 모델을 테스트셋에 적용해 정확도를 구합니다. 
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```

    Epoch 1/200
    14/14 [==============================] - 0s 2ms/step - loss: 0.7037 - accuracy: 0.4074
    Epoch 2/200
    14/14 [==============================] - 0s 2ms/step - loss: 0.6908 - accuracy: 0.5630
    Epoch 3/200
    14/14 [==============================] - 0s 3ms/step - loss: 0.6849 - accuracy: 0.5630
    Epoch 4/200
    ...
    Epoch 197/200
    17/17 [==============================] - 0s 2ms/step - loss: 0.0168 - accuracy: 1.0000
    Epoch 198/200
    17/17 [==============================] - 0s 2ms/step - loss: 0.0170 - accuracy: 1.0000
    Epoch 199/200
    17/17 [==============================] - 0s 2ms/step - loss: 0.0163 - accuracy: 1.0000
    Epoch 200/200
    17/17 [==============================] - 0s 2ms/step - loss: 0.0169 - accuracy: 1.0000
    ------------------------------------------------------------------------
    2/2 [==============================] - 0s 9ms/step - loss: 0.3629 - accuracy: 0.8571
    Test accuracy: 0.8571428656578064


## 5. k겹 교차 검증

데이터가 충분히 많아야 모델 성능을 향상된다고 앞서 말했습니다. 이는 학습과 테스트를 위한 데이터를 충분히 확보할수록 세상에 나왔을 때 더 잘 동작하기 때문입니다. 하지만 실제 프로젝트에서는 데이터를 확보하는 것이 쉽지 않거나 많은 비용이 발생하는 경우도 있습니다. 따라서 가지고 있는 데이터를 십분 활용하는 것이 중요합니다. 특히 학습셋을 70%, 테스트셋을 30%로 설정할 경우 30%의 테스트셋은 학습에 이용할 수 없다는 단점이 있습니다.

이를 해결하기 위해 고안된 방법이 k겹 교차 검증(k-fold cross validation)입니다. k겹 교차 검증이란 먼저 데이터셋을 k 개로 나누고 그중 하나를 테스트셋으로 사용하고 테스트셋으로 선정하지 않은 나머지 데이터셋를 모두 합해서 학습셋으로 사용하여 정확도를 구합니다. 다시 k 개의 데이테셋에서 테스트셋으로 선택되지 않은 데이터셋을 테스트셋으로 사용하고 나머지 테이터셋을 모아 학습 데이터셋으로 사용하여 정확도를 구합니다. k개의 데이터셋을  다 한번씩 테스트 셋으로 두고 정확도를 구해서 얻은 k개의 정확도의 평균을 구해 최종 정확도를 판단합니다.

이렇게 하면 가지고 있는 데이터의 100%를 학습셋으로 사용할 수 있고 또 동시에 테스트셋으로도 사용할 수 있습니다. 예를 들어 5겹 교차 검증(5-fold cross validation)의 예가 그림(13-7)에 설며되어 있습니다.   
<br><center>
(그림. 13-7) 5겹 교차 검증 방법<br>
<img src="https://drive.google.com/uc?id=1UXPDWkDSPC0hhLCwxNwgWPuSfVPHw5Xm" width=500>
</center><br>  

이제 초음파 광물 예측 예제를 통해 5겹 교차 검증을 실히새 보겠습니다. 데이터를 원하는 수만큼 나우어 각각 학습셋과 테스트 셋으로 사용하게 하는 함수는 사이킷런 라이브러리의 ```KFold()```함수입니다. 실습 코드에서 ```KFold()```를 활용하는 부부만 뽑아 보며 다음과 같습니다. 

```
k=5
kfold = KFold(n_splits=k, shuffle=True)
acc_score=[]

for tranin_index, test_index in kfold.split(X):
   X_train, X_text = X.iloc[train_index, :], X.iloc[test_index,:]
   y_train, y_text = y.iloc[trian_index], y.iloc[test_index]
```

데이터셋을 몇개로 나눌 것인지 정해서 ```k```변수에 할당합니다. 사이킷런의 ```KFold()``` 함수를 불러 옵니다. ```shuffle```에 ```True```를 할당하면 데이터셋을 섞습니다. _k_번의 정확도 계산 결과(정확도 값)를 ```acc_score``` 리스트에 할당할 예정입니다. ```split()``` 함수에 의해 k개의 학습셋과 테스트 셋으로 분리되며 ```for``` 문에 의해 _k_번 반복됩니다. 

반복되는 매 학습 과정 마다 정확도를 구해 다음과같이 ```acc_score``` 리스트에 붙입니다.

```
accuracy = model.evaluate(X_test,y_test)   # 정확도를 구합니다.
acc_score.append(accuracy[1])              # acc_score 리스트에 저장합니다.
```

_k_번의 학습이 끝나면 각 정확도를 취합해 모델 성능 평가를 합니다. 아래에는 완성된 코드를 보입니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import pandas as pd

# 깃허브에 준비된 데이터를 가져옵니다. 앞에서 이미 데이터를 가져왔으므로 추석 처리합니다. 3번 예제만 별도 실행 시 주석을 해제하여 실습하세요.
# !git clone https://github.com/taehojo/data.git

# 광물 데이터를 불러옵니다.
df = pd.read_csv('./data/sonar3.csv', header=None)

# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:,0:60]
y = df.iloc[:,60]
```


```python
# 몇 겹으로 나눌 것인지를 정합니다. 
k=5

# KFold 함수를 불러옵니다. 분할하기 전에 샘플이 치우치지 않도록 섞어 줍니다.
kfold = KFold(n_splits=k, shuffle=True)

# 정확도가 채워질 빈 리스트를 준비합니다.
acc_score = []

# 모델 구조 생성
def model_fn():
    model = Sequential() # 딥러닝 모델의 구조를 시작합니다.
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# K겹 교차 검증을 이용해 k번의 학습을 실행합니다. 
for train_index , test_index in kfold.split(X):  # for 문에 의해서 k번 반복합니다. spilt()에 의해 k개의 학습셋, 테스트셋으로 분리됩니다.
    X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]  
    y_train , y_test = y.iloc[train_index], y.iloc[test_index]

    model = model_fn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0) 
    
    accuracy = model.evaluate(X_test, y_test)[1]  # 정확도를 구합니다.
    acc_score.append(accuracy)  # 정확도 리스트에 저장합니다.

# k번 실시된 정확도의 평균을 구합니다.
avg_acc_score = sum(acc_score)/k

# 결과를 출력합니다.
print('정확도:', acc_score)
print('정확도 평균:', avg_acc_score)
```

    2/2 [==============================] - 0s 9ms/step - loss: 0.7299 - accuracy: 0.8095


    WARNING:tensorflow:5 out of the last 13 calls to <function Model.make_test_function.<locals>.test_function at 0x7f4387504a70> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    2/2 [==============================] - 0s 7ms/step - loss: 0.3557 - accuracy: 0.8333


    WARNING:tensorflow:6 out of the last 15 calls to <function Model.make_test_function.<locals>.test_function at 0x7f4382a65cb0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    2/2 [==============================] - 0s 11ms/step - loss: 0.8184 - accuracy: 0.8571
    2/2 [==============================] - 0s 9ms/step - loss: 0.5467 - accuracy: 0.8293
    2/2 [==============================] - 0s 5ms/step - loss: 0.8833 - accuracy: 0.7561
    정확도: [0.8095238208770752, 0.8333333134651184, 0.8571428656578064, 0.8292682766914368, 0.7560975551605225]
    정확도 평균: 0.8170731663703918


학습이 진행되는 과정을 화면에 출력되지 않게 하려고 ```model.fit()``` 함수의 파라메타 ```verbose```에 0을 할당했습니다.

🚀 잠깐만요.  
텐서플로 함수가 for문에 포함되는 경우 다음과 같은 WARNING 메시지가 나오는 경우가 있습니다. 텐서프로 구동에는 문제가 없으므로 그냥 진해하면 됩니다.  
WARNING:tensorflow: 5 out of the last 9 call to <function Model.make_test_function.<locals>.test_function at ....> triggered tf.function retracing...


<br>
이렇게 해서 가지고 있는 데이터를 모두 사용해 학습과 테스트를 진행했습니다. 이제 다음 장에서 학습 과정을 시각화해 보는 방법과 학습을 몇 번 반복할지(epochs) 스스로 판단하게 하는 방법 등을 알아보며 모델 성능을 향상시켜 보겠습니다. 
