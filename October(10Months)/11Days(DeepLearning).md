
http://github.com/taehojo/deeplearning

## 1. 환경 준비


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
```

### 데이터 경로


```python
# from google.colab import drive
# drive.mount('/content/drive')
```

## 2. 데이터 준비


```python
!git clone https://github.com/taehojo/data.git   # 깃허브에 준비된 데이터를 가져옵니다.

Data_set = np.loadtxt('./data/ThoraricSurgery3.csv',delimiter=',')
X = Data_set[:,0:16]
y = Data_set[:,16]
```

    fatal: destination path 'data' already exists and is not an empty directory.


## 3. 구조 결정


```python
model = Sequential()                                                  # 딥러닝 모델의 구조를 결정합니다.
model.add(Dense(30, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## 4. 모델 실행


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 딥러닝 모델을 실행합니다.
history=model.fit(X, y, epochs=5, batch_size=16)
```

    Epoch 1/5
    30/30 [==============================] - 1s 3ms/step - loss: 2.4827 - accuracy: 0.8468
    Epoch 2/5
    30/30 [==============================] - 0s 3ms/step - loss: 1.3840 - accuracy: 0.8255
    Epoch 3/5
    30/30 [==============================] - 0s 3ms/step - loss: 0.5801 - accuracy: 0.8319
    Epoch 4/5
    30/30 [==============================] - 0s 3ms/step - loss: 0.4476 - accuracy: 0.8511
    Epoch 5/5
    30/30 [==============================] - 0s 3ms/step - loss: 0.4401 - accuracy: 0.8532


# 3장_basic_math


```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
a = 1
b = 2
y = a*x + b

plt.plot(x,y)
plt.grid()
plt.show()
```

<img width="364" alt="스크린샷 2022-10-11 오후 5 33 21" src="https://user-images.githubusercontent.com/87309905/195040268-30cd46d4-2497-4789-8b0d-c258dec95441.png">



```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
a = 1
b = 2
y = a*x + b

plt.plot(x,y)
plt.xlabel('x')         # x label 추가
plt.ylabel('y')         # y label 추가
plt.title('$y = ax+b$') # 타이틀에 수식 넣기

plt.grid()
plt.show()
```



<img width="394" alt="스크린샷 2022-10-11 오후 5 33 40" src="https://user-images.githubusercontent.com/87309905/195040325-c4f8e4e5-41dc-45b9-98b3-34026709f769.png">
    



```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
a = 1
b = 2
y = a*x + b

plt.plot(x,y)
plt.xlabel('x')         # x label 추가
plt.ylabel('y')         # y label 추가
plt.title('$y = ax+b$') # 타이틀에 수식 넣기

plt.axhline(0, color = 'black') # 수평 좌표선
plt.axvline(0, color = 'black') # 수직 좌표선

plt.grid()
plt.show()
```


<img width="403" alt="스크린샷 2022-10-11 오후 5 33 56" src="https://user-images.githubusercontent.com/87309905/195040392-c778cc0e-af17-42cf-acae-90cbd550295a.png">
    




```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
a = 1
b = 2
y = a*x + b

plt.plot(x,y)
plt.xlabel('x')         # x label 추가
plt.ylabel('y')         # y label 추가
plt.title('$y = {}x+{}$'.format(a,b)) # 타이틀에 수식 넣기

plt.axhline(0, color = 'black') # 수평 좌표선
plt.axvline(0, color = 'black') # 수직 좌표선

plt.grid()
plt.show()
```



<img width="387" alt="스크린샷 2022-10-11 오후 5 34 11" src="https://user-images.githubusercontent.com/87309905/195040440-9c279ed8-76fd-4298-9561-b574d0304c02.png">
   


***

# 4장. 가장 훌륭한 예측선

## 최소 제곱법

 이제 우리 목표는 가장 정확한 선을 긋는 것입니다. 더 구체적으로 정확한 기울기 $a$와 정확한 $y$ 절편 $b$를 알아내면 되는 것입니다. 만일 우리가 **최소제곱법**(method of least squares) 공식을 알고 적용한다면 일차 함수의 기울기 $a$와 $y$ 절편 $b$를 바로 구할 수 있습니다.

기울기 $a$는 다음 식으로 계산할 수 있습니다.  
(식. 4-1) 
$$ a = \frac{\sum_{i=0}^{N-1}(x_i-\bar{x})(y_i - \bar{y}) } { \sum_{i=0}^{N-1}(x_i-\bar{x})^2} $$  
  


여기서 $\bar{x}$는 $x$의 평균이며 $\bar{y}$는 $y$의 평균입니다. $y$절편 $b$는 아래 식으로 구할 수 있습니다.  

$$ b = \bar{y} - (\bar{x}a)$$


## 1. 환경 준비


```python
import numpy as np
import matplotlib.pyplot as plt
```

## 2. 데이터 준비


```python
# 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
```

## $x$와 $y$의 평균값


```python
#x의 평균값을 구합니다.
mx = np.mean(X)

#y의 평균값을 구합니다.
my = np.mean(y)

# 출력으로 확인합니다.
print("x의 평균값:", mx)
print("y의 평균값:", my)
```

    x의 평균값: 5.0
    y의 평균값: 90.5


#### [기울기 공식의 분모와 분자]


```python
# 기울기 공식의 분모(divisor) 부분입니다.
divisor = sum([(x_i - mx)**2 for x_i in X])

# 기울기 공식의 분자(dividend) 부분입니다.
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (X[i] - mx) * (y[i] - my)
    return d
    
dividend = top(X, mx, y, my)

# 출력으로 확인합니다.
print("분모:", divisor)
print("분자:", dividend)
```

    분모: 20.0
    분자: 46.0


## 3. 기울기($a$) $y$절편($b$)구하기


```python
# 기울기 a를 구하는 공식입니다.
a = dividend / divisor

# y 절편 b를 구하는 공식입니다.
b = my - (mx*a)

# 출력으로 확인합니다.
print("기울기 a =", a)
print("y절편 b =", b)
```

    기울기 a = 2.3
    y절편 b = 79.0


### 4. 과제
[과제1]  
공부한 시간 대비 취득 성적을 그래프로 표현하시오. 그리고 주어진 데이터들의 특징(경향)을 가장 잘 나타낼 것으로 **예측한 선($\hat{y})$**을 그리시오.
$$ \hat{y} = 2.3x+79 $$

[과제2]  
공부한 시간($x$) 대비 취득 성적($y$)이 아래 표와 같을 때, 주어진 데이터들의 특징(경향)을 가장 잘 나타낼 것으로 예측한 선을 그리시오.

|공부한 시간($x$)|2|3|4|6|8|
|---|---|---|---|---|---|
|성적($y_i$)|81|83|93|91|97|


```python
# [과제 1]
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

a = 2.3
b = 79
y_hat = a * X + b
plt.plot(X, y_hat)
```




    [<matplotlib.lines.Line2D at 0x7f38ca016210>]




<img width="375" alt="스크린샷 2022-10-11 오후 5 34 30" src="https://user-images.githubusercontent.com/87309905/195040508-e46800de-e17e-459e-8487-e0aa98ca8097.png">
    



```python
# [ 과제 2 ]
X = np.array([2,3,4,6,8])
y = np.array([81,83,93,91,97])

# X의 평균값
mx = np.mean(X)
# y의 평균값
my = np.mean(y)

print(f"x의 평균 : {mx}, y의 평균 : {my}")

# 기울기 공식의 분모(divisor)
divisor = sum([(x_i - mx)**2 for x_i in X])

# 기울기 공식의 분자(dividend)
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (X[i] - mx) * (y[i] - my)
    return d
    
dividend = top(X, mx, y, my)

print(f"분자 : {dividend}, 분모 : {divisor}")

# 기울기
a = dividend/divisor
# y절편
b = my - (mx*a)

print(f"기울기 a : {a}, y절편 : {b}")

# 그래프
y_hat = a * X + b
# 시간 당 성적 ( 점으로 표현 )
plt.plot(X,y,'o')
# 예측 선
plt.plot(X, y_hat)

```

    x의 평균 : 4.6, y의 평균 : 89.0
    분자 : 58.0, 분모 : 23.2
    기울기 a : 2.5, y절편 : 77.5





    [<matplotlib.lines.Line2D at 0x7f38b5632510>]



<img width="396" alt="스크린샷 2022-10-11 오후 5 34 45" src="https://user-images.githubusercontent.com/87309905/195040561-308b372b-002b-41a0-a71a-37a261abe8b4.png">

    


## 5. 평균 제곱 오차

 최소 제곱법을 이용해 기울기 $a$와 $y$ 절편 $b$을 편리하게 구했지만 이 공식 만으로는 앞으로 만나게 될 모든 상황을 해결하기 어렵습니다. 여러 개의 입력($x_0, x_1, ..., x_N$)을 처리하기에는 무리가 있기 때문입니다. 앞에서 살펴본 예에서는 성적에 영향을 주는 요소로 '공부한 시간' 하나만 고려했지만 2장에서 살펴본 폐암 수술 환자의 생존율 데이터를 보면 입력 데이터의 종류가 16개나 됩니다.  

 예측선을 구하기 위해 가장 많이 사용하는 방법은 '일단 그리고 조금씩 수정해 가기' 방식입니다. 가설을 하나 세운 후 이 값이 주어진 요건을 충족하는지 판단하면서 조금씩 변화를 주고 이 변화가 긍정적이면 오차가 최소가 될 때까지 이 과정을 계속 반복하는 방법입니다. 이는 딥러닝을 가능하게 하는 가장 중요한 원리 중 하나입니다.   

 그런데 선을 긋고 나서 수정하는 과정에서 빠지면 안되는 것이 있습니다. 나중에 그린 선이 먼저 그린 선보다 더 좋은지 나쁜지 판단하는 방법입니다. 즉, 각 선의 오차를 계산할 수 있야하고 오차가 작은 쪽으로 바꾸는 알고리즘이 필요한 것입니다. 이를 위해 주어진 선의 오차를 평가하는 방법이 필요합니다. 오차를 구할 때 가장 많이 사용되는 방법이 **평균 제곱 오차(Mean Square Error, MSE)**입니다.  

지금부터 평균 제곱 오차를 구하는 방법을 알아보겠습니다. 먼저 앞서 공부한 내용에서 언급한 '공부한 시간($x$)과 성적($y$)의 관계도'를 보겠습니다.




```python
# 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

plt.plot(X,y,'o')
plt.xlabel('study time')
plt.grid()
```

<img width="391" alt="스크린샷 2022-10-11 오후 5 34 56" src="https://user-images.githubusercontent.com/87309905/195040601-bf320c06-5aa5-458e-819f-814cd1eb06f3.png">

    


우리는 위에서 최소 제곱법을 이용해 점들의 특성(경향)을 가장 잘 나타내는 최적의 직선이 $y=2.3x+79$임을 구했지만 이번에는 최소 제곱법을 사용하지 않고 임의의 직선을 선택해보겠습니다. 즉 기울기 $a$와 $y$ 절편 $b$를 임의로 선택해 보겠습니다. 임의로 결정한 선이 제시하는 공부 시간에 따른 취득 예상 점수와 실제 학생들이 취득한 점수(점)와의 차이를 최소화하는 방법으로 선의 기울기 $a$와 $y$절편 $b$값을 구해 보려고 합니다.  

 먼저 임의의 선을 하나 긋기 위해 기울기 $a$와 $y$ 절편 $b$를 3과 76이라고 가정해 보겠습니다. 이 때 임의의 선을 나타내는 1차 방정식은 다음과 같습니다. 
$$y=3x+76$$  
실제 데이터와 임의로 생성한 예측 선을 그래프로 표현해 보면 아래와 같습니다.


```python
# 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

# 공부한 시간은 2시간부터 8시간까지만 고려할 생각임.
x = np.arange(2, 9) 

# 임의의 직선 생성(임의의 예측선 생성)
a = 3     # 기울기(a)
b = 76    # 절편(b)
y_hat = a*x + b

# 실 데이터와 예측한 데이터 출력
plt.plot(X,y,'o')         # 공부 시간 vs. 취득 성적 그래프(점(o)로 표현)
plt.plot(x, y_hat)        # 모델이 예상한 '공부 시간 vs. 취득 예상 성적' 그래프
plt.xlabel('study time')
plt.grid()

```


<img width="408" alt="스크린샷 2022-10-11 오후 5 35 10" src="https://user-images.githubusercontent.com/87309905/195040651-23292801-62c9-45dc-9b89-35f1fccd8543.png">

    


임의의 선이 예측한 성적과 실제 취득한 점수 사이에 차가 어느 정도 있는지 확인하기 위해 각 점과 직선 사이의 거리를 측정합니다. 


```python
# 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

# 공부한 시간은 2시간부터 8시간까지만 고려할 생각임.
x = np.arange(2, 9) 

# 임의의 직선 생성(임의의 예측선 생성)
a = 3
b = 76
y_hat = a*x + b

# 실 데이터와 예측한 데이터 출력
plt.plot(X,y,'o')         # 공부 시간 vs. 취득 성적 그래프(점(o)로 표현)
plt.plot(x, y_hat)        # 모델이 예상한 '공부 시간 vs. 취득 예상 성적' 그래프

plt.xlabel('study time')
plt.grid()

#오차 그래프 그리기
for i in range(4):
  plt.plot( [X[i], X[i]],[ y[i], a*X[i]+b], 'r' )

```


    
<img width="405" alt="스크린샷 2022-10-11 오후 5 35 21" src="https://user-images.githubusercontent.com/87309905/195040689-eaeb5b06-0395-414e-a1b5-bc06634b9b1f.png">
    


위 그래프에서 빨간색 선의 길이를 근거로 예측선이 잘 그어졌는지 판단할 수 있습니다. 이 빨간색 선과 (실 데이터)점과의 거리가 오차이며 이 오차의 합이 작을 수록 예측 선이 잘 그어졌다고 판단합니다. 


```python
# 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
X = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

# 공부한 시간은 2시간부터 8시간까지만 고려할 생각임.
x = np.arange(2, 9) 

# 임의의 직선 두개 생성(임의의 예측선 두개 생성, 각기 기울기가 다르게)
# 
a1 = 4
b = 76
pre_y1 = a1*x + b

a2 = 5
pre_y2 = a2*x + b


f = plt.figure( figsize=(14,5) )

# 두 개의 그래프 영역을 만들어 첫 번째 영역에 pre_y1을 출력하고
# 두 번째 영역에 pre_y2를 출력

plt.subplot(1,2,1)
plt.plot(X,y,'o')         # 공부 시간 vs. 취득 성적 그래프(점(o)로 표현)
plt.plot(x, pre_y1)       # 모델이 예상한 '공부 시간 vs. 취득 예상 성적' 그래프
plt.title('$\hat{y}=4x+79$')
plt.xlabel('study time')
plt.ylim(ymin=80, ymax=120)
plt.grid()

#오차 그래프 그리기
for i in range(4):
  plt.plot( [X[i], X[i]],[ y[i], a1*X[i]+b], 'r' )


plt.subplot(1,2,2)
plt.plot(X,y,'o')         # 공부 시간 vs. 취득 성적 그래프(점(o)로 표현)
plt.plot(x, pre_y2)       # 모델이 예상한 '공부 시간 vs. 취득 예상 성적' 그래프
plt.title('$\hat{y}=5x+79$')
plt.xlabel('study time')
plt.ylim(ymin=80, ymax=120)
plt.grid()

#오차 그래프 그리기
for i in range(4):
  plt.plot( [X[i], X[i]],[ y[i], a2*X[i]+b], 'r' )
```

<img width="832" alt="스크린샷 2022-10-11 오후 5 35 38" src="https://user-images.githubusercontent.com/87309905/195040745-9c78f574-c6ab-4117-aeaa-92eafce35602.png">

    


 예측 선의 기울기가 잘못될수록 빨간색 선들의 길이 합, 즉 오차의 합도 커집니다. 만일 기울기가 무한대로 커지면 오차도 무한대로 커지는 상관관계가 있는 것을 알 수 있습니다.   
  
 빨간색 선의 길이는 예측한 값과 실제 값 사이의 오차입니다. 각 빨간색 선의 길이(오차)의 합을 실제로 계산해 보겠습니다. 예를 들어 2시간($x=2$) 공부했을 때 취득한 점수($y$)는 81점인데 예측 선 식($\hat{y}=3x+76$)에 $x=2$를 대입했을 때 갖게되는 값은 82점입니다. 실제 값과 예측 값 사이의 차가 오차입니다. 오차($e_i$)를 구하는 식을 다음과 같이 표현할 수 있습니다.  

$$ e_i= y_i - \hat{y_i} $$  

여기서 아래 첨자 $i$는 인덱스이며 $y_i$는 실제 취득한 $i$번째 점수이며 $\hat{y_i}$는 예측선($\hat{y}=3x+76$)에 의해 구한 $i$번째 예측 점수입니다. 위 식에 주어진 데이터를 대입해서 얻을 수 있는 모든 오차 값을 아래 표에 정리했습니다.

|오차|-1|5|-3|-3|
|---|---|---|---|---|
|공부한 시간($x$)|2|4|6|8|
|성적($y_i$)|81|93|91|97|
|예측 성적($\hat{y_i}$)|82|88|94|100|  

$$-1+5+(-3)+(-3) = -2$$  

이렇게 해서 구한 오차($e_i$)를 모두 더하면 $-2$가 됩니다. 

$$ error = \sum_{i=0}^{N-1} e_i = \sum_{i=0}^{N-1}(y_i - \hat{y_i})$$  

그런데 위의 식과 같이 구한 구한 오차 값($error, 오차의\;단순\;합$)으로는 실제로 오차가 어느 정도 되는지 가늠하기 어렵습니다. 각 오차값에 양수와 음수가 섞여 있어서 오차 값($e_i$)을 단순히 더해 버리면 합($error$)이 $0$이 될 수도 있기 때문입니다. 수식으로 표현해서 설명하며 아래와 같이 방식으로 오차($error$)를 구하면 오차가 얼마나 발생했을지 제대로 가늠하기 어렵다는 뜻입니다. 각 오차($e_i$)의 부호를 없애야 정확한 오차를 구할 수 있습니다. 따라서 오차의 합 평가할 때는 각 오차 값을 제곱한 후 합산을 합니다. 이 과정을 식으로 표현하면 아래와 같습니다. 

$$평균제곱오차(MSE) = \frac{1}{N}e=\frac{1}{N} \sum_{i=0}^{N-1}(y_i - \hat{y_i})^2$$

여기서 $N$은 총 데이터 샘플의 개수 입니다. 위 식을 적용하여 오차의 합을 계산하면 $1+25+9+9=44$가 됩니다. 우리가 구하고자 하는 **평균 제곱 오차**는 위에서 구한 오차의 합($e$)을 샘플 개수 $N$으로 나눈 것입니다. 위 식은 머신 러닝과 딥러닝을 학습할 때 자주 등장하는 중요한 식입니다. 위 식을 조금 단순한 형태로 아래와 같이 표현하겠습니다. 

$$평균제곱오차(MSE) = \frac{1}{N} \sum (y_i - \hat{y_i})^2$$  

앞서 구한 오차의 합, 44와 $x$의 총 개수 4를 위 식에 대입해서 얻은 평균제곱오차(MSE)는 $44/4 =11$입니다. 이로써 우리가 임의의 만든 예측 선의 평균제곱오차가 11이라는 것을 알았습니다. 이제 우리의 다음 일은 평균제곱오차가 11보다 작은 예측 선을 만들기 위해 기울기 $a$와 $y$절편 $b$를 찾는 것입니다.   

**선형회귀**란 임의의 직선을 그어 이에 대한 평균 제곱 오차를 구하고 이 평균 제곱 오차(MSE) 값을 가장 작게 만들어 주는 예측 선의 기울기 $a$값과 $y$절편 $b$값을 찾아가는 작업입니다. 





## 6. 파이썬 코딩으로 확인하는 평균 제곱 오차

### 1. 환경 준비


```python
import numpy as np
```

### 2. 데이터 준비


```python
# 가상의 기울기 a와 y 절편 b를 정합니다.
fake_a=3
fake_b=76

# 공부 시간 x와 성적 y의 넘파이 배열을 만듭니다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
```

### 3. 평균 제곱 오차 구하기


```python
# y=ax + b에 가상의 a,b 값을 대입한 결과를 출력하는 함수입니다.
def predict(x):
    return fake_a * x + fake_b

# 예측 값이 들어갈 빈 리스트를 만듭니다.
predict_result = []

# 모든 x 값을 한 번씩 대입하여 predict_result 리스트를 완성합니다.
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))
```

    공부시간=2, 실제점수=81, 예측점수=82
    공부시간=4, 실제점수=93, 예측점수=88
    공부시간=6, 실제점수=91, 예측점수=94
    공부시간=8, 실제점수=97, 예측점수=100



```python
# 평균 제곱 오차 함수를 각 y 값에 대입하여 최종 값을 구하는 함수입니다.
n=len(x)  
def mse(y, y_pred):
    return (1/n) * sum((y - y_pred)**2)

# 평균 제곱 오차 값을 출력합니다.
print("평균 제곱 오차: " + str(mse(y,predict_result)))
```

    평균 제곱 오차: 11.0


임의로 정한 예측선이 $y=3x+76$일 때 평균 제곱 오차(MSE)가 11.0이라는 것을 알게 되었습니다. 이제 남은 것은 이 평균 제곱 오차(MSE)를 줄이면서 새로운 선을 긋는 것입니다. 이를 위해서 예측 선의 기울기$a$값과 $y$절편 $b$값을 적절히 조절하면서 평균 제곱 오차(MSE)의 변화를 살펴보고 그 오차가 최소가 되는 a값과 b값을 구해야 합니다. 

다음 장에서는 오차를 줄이는 방법에 대해 알아보겠습니다. 

## 4. 과제
[과제 1]  
임의의 예측 선($\hat{y}$)을 다음과 같이 가정합니다. 이 때의 평균 제곱 오차(MSE)를 구하는 코드를 작성하세요.
$$ \hat{y} = 2.3x+80$$  
[과제 2]  
앞서 **'최소 제곱법'**으로 구한 예측 선($\hat{y}$)의 **'평균 제곱 오차(MSE)'** 값은 얼마인지 계산하는 프로그램을 작성하시오
$$\hat{y} = 2.3x+79$$

오차($e_i$)는 아래와 같이 정의되었음. 
$$ e_i= y_i - \hat{y_i} $$  

|오차|-3.6|3.8|-2.8|-1.4|
|---|---|---|---|---|
|공부한 시간(x)|2|4|6|8|
|성적($y_i$)|81|93|91|97|
|예측값($\hat{y_i}$)|84.6|89.2|93.8|98.4|  


```python
# [ 과제 1 ]
a = 2.3
b = 80

X = np.array([2,4,6,8])
y = np.array([81,93,91,97])

def predict(x):
  return a * x + b

predict_result = []

# 예측 점수 
for i in range(len(X)):
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

# 평균 제곱 오차 함수
n = len(X)
def mse(y, y_pred):
    return (1/n) * sum((y - y_pred)**2)

# 평균 제곱 오차 값
print(f"{str(mse(y,predict_result))}")
```

    공부시간=2, 실제점수=81, 예측점수=85
    공부시간=4, 실제점수=93, 예측점수=89
    공부시간=6, 실제점수=91, 예측점수=94
    공부시간=8, 실제점수=97, 예측점수=98
    9.299999999999983



```python
# [ 과제 2 ]
a = 2.3
b = 79

X = np.array([2,4,6,8])
y = np.array([81,93,91,97])

def predict(x):
  return a * x + b

predict_result = []

# 예측 점수 
for i in range(len(X)):
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

# 평균 제곱 오차 함수
n = len(X)
def mse(y, y_pred):
    return (1/n) * sum((y - y_pred)**2)

# 평균 제곱 오차 값
print(f"{str(mse(y,predict_result))}")
```

    공부시간=2, 실제점수=81, 예측점수=84
    공부시간=4, 실제점수=93, 예측점수=88
    공부시간=6, 실제점수=91, 예측점수=93
    공부시간=8, 실제점수=97, 예측점수=97
    8.299999999999985


***

# 5장. 선형 회귀 모델: 먼저 긋고 수정하기

(식1)  
$$ e = \sum_{i=0}^{N-1} {e_i}^2 = \sum_{i=0}^{N-1}(y_i - \hat{y_i})^2$$  
(식2)  
$$평균제곱오차(MSE) = \frac{1}{N} \sum (y_i - \hat{y_i})^2$$  

 4장을 학습하면서 기울기 $a$를 적절하게 잡지 못하면 오차($e$)가 커지는 것을 확인했습니다. 기울기를 크게 잡거나 너무 작게 잡아도 오차($e$)가 커집니다. 기울기 $a$와 오차($e$) 사이에는 이렇게 상관관계가 있습니다. 이때 기울기가 무한대로 커지거나($+\infty$)  또는 무한대로 작아지면($-\infty$) y축과 나란한 직선이 됩니다.그러면 오차($e$)도 무한대로 커지겠됩니다. 기울기 $a$와 오차($e$ 또는 MSE)의 관계를 다시 말하면 기울기 $a$와 오차 사이에는 아래 그림과 같이 **2차 함수의 관계**가 있다는 뜻입니다.  
<br><center>
(그림. 5-1) 기울기 $a$와 오차와의 관계:적절한 기울기를 찾을 때 오차가 최소화된다<br>
<img src="https://drive.google.com/uc?id=11TN59OhxqSl2QOGKeMfDt88KXY_epR-t" width=400>
</center><br>


```python
import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-1, 9, 100)
w = 4
m = 4
e0 = 2
e = w*(a-m)**2 + e0
plt.plot(a, e)
plt.xlabel('$gradient, a$')
plt.ylabel('$error,e$')
plt.grid()
```


    
![png](output_58_0.png)
    


 위 그래프에서 가로 축은 기울기 $a$이며 세로 축은 오차 $e$를 나타내고 있습니다. 변수 $a$에 따른 오차 $e$를 그린 그래프이다. 이 그래프에서 **오차가 가장 작을 때는 언제인가?** 그래프의 가장 아래쪽으로 볼록한 부분에 이르렀을 때입니다. 가속도 $a$가 4일 때입니다. 위 그래프는 다음 식(3)과 같이 표현할 수 있습니다.  
(식3)  
$$e=w(a-m)^2+e_0$$  

  
위 그래프는 $m=4$, $e_0=2$ 그리고 $w=4$ 일 때를 그린 것입니다. 이 그래프에서 오차($e$)가 최소가 되는 지점은 기울기($a$)의 값이 $m$이 될 때입니다. $a=m$일 때 오차($e$)는 최소 오차값인 $e_0$이 됩니다. 


 좀 익숙한 기호를 이용하여 위 식을 눈에 익숙한 형태로 바꾸면 위 식(3)은 아래 식(4)와 같습니다. 가로 축은 $x$이며 세로 축을 $y$라고 가정합니다.$a$는 가속도가 아닌 임의의 상수입니다.  
(식4)  
$$y=a(x-x_0)^2+y_0 =a(x-4)^2+y_0$$
$$y=4(x-4)^2+2=4(x^2-8x +16) +2=4x^2-32x+68$$




 우리는 4장에서 임의의 기울기($a_1$)를 선택하여 예측 선($\hat{y} =a_1x+b$)을 만들고 예측한 값과 실제값 사이의 평균 제곱 오차(MSE)를 구해봤습니다. 이 때 임의로 정한 기울기 값을 적절히 변경해가면서 오차값이 최소 오차($e_0$)가 되는 최적의 기울기($m$)를 찾게 되는 것입니다. 기울기($a$)값이 변경됨에 따라 오차($e$)가 최소 오차($e_0$)에 가까워지고 따라서 기울기가($a$)가 최적 기울기 값 $m$에 가까워진다는 것을 컴퓨터가 판단해야할 것 입니다. 이러한 판단을 하게 하는 방법이 바로 **경사하강법(gradient decent)**입니다.


## 1.경사하강법(Gradient decent)
참고
 - gradient : (특히 도로, 철도의) 경사도, 기울기. 예)  steep gradient(급경사도)
 - descent : 내려오기, 하강, 내리막($\leftrightarrow$ascent)

% gradient descent = 내리막 경사?

식(5)  
$$y=4(x-4)^2+2 $$  
식(5)에서 변수$x$를 변수 $a$로 변경하여 표현하면 식(6)과 같이 표현할 수 있습니다.   
식(6)
$$e=w(a-m)^2+e_0 →e=4(a-4)^2+2$$  

위 식(6)의 그래프를 다시 살펴보겠습니다.



```python
a = np.linspace(-1, 9, 100)
w = 4
m = 4
e0 = 2
e = w*(a-m)**2 + e0
plt.plot(a, e)
plt.xlabel('$gradient, a$')
plt.ylabel('$error,e$')
plt.grid()
```


    
<img width="420" alt="스크린샷 2022-10-11 오후 5 36 31" src="https://user-images.githubusercontent.com/87309905/195040955-77d2e69c-483c-4b6e-931c-12c09bf50f21.png">
    


<br><center>
<img src="https://drive.google.com/uc?id=1fBNK0gJhR5WOzx84XY1kOgYJQPjgaxEV" width=400>
</center><br>

 예측선의 기울기($a$)가 $a < m$인 구간에서 오차 식($e$)의 기울기는 (-)음수이며 예측선의 기울기($a$)가 $a > m$인 구간에서는 오차 식($e$)의 기울기는 (+)양수이다. 최소 오차값($e_0$)얻기 위해서는 오차식($e$)에서 $a=m$이 될 때이며 이 때의 **오차 식($e$)의 기울기는 0입니다.**  따라서 우리가 해야할 일은 오차 식의 기울기가 0인 지점을 찾는 것이다. 다른 말로 오차 식($e$)의 기울기가 0일 때의 예측 선의 기울기($a$)를 찾는 것입니다. 이를 위해 다음 과정을 진행합니다. 

<center>
<img src="https://drive.google.com/uc?id=1Gtg9aEpWy_dMEqrWOF4wDtVFSj5eFMpa" width=400>
</center>

1. 임의의 위치($a_1$)에서 오차식($e$)의 기울기를 구한다.
2. 구한 기울기가 양수라면 $a_1$ 보다 더 작은 $a$ 값을 선택하고
   구한 기울기 값이 음수라면 $a_1$보다 더 큰 $a$값은 선택하여 기울기를 값을 구한다. 구한 기울기 값이 0에 매우가까울 때까지 2번 과정을 반복한다.

**경사하강법은 이렇게 반복적으로 기울기 $a$를 바꿔가면서 $m$ 값을 찾아가는 방법이다.**


***

 위 1~2 과정을 반복하면서 다음 기울기 조사 지점으로 이동할 때의 값 변화폭을 잘 선택해야함. 지나치게 크게 $a$값을 바꾸하면 기울기가 0인 지점을 못 찾을 수 있다. a값의 변화 폭을 어떻게 정하느냐가 **'학습율'**입니다. 정리해서 말하면 **경사하강법은 기울기($a$)의 변화에 따른 오차의 변화를 2차 식($e$)를 만들고 적절한 학습율을 설정해 오차 식의 기울기가 0인 지점을 찾는 것입니다.**  
  

 참고 - $y$절편 $b$의 값도 이와 같은 성질을 가지고 있습니다. $b$ 값이 너무 크면 오차도 함께 커지고 너무 작아도 오차가 커집니다. 그래서 최적의 $b$ 값을 구할 때 경사 하강법을 사용합니다. 


## 2. 파이썬 코딩으로 확인하는 선형 회귀
지금까지의 설명한 내용을 바탕으로 파이썬 코드를 작성해보려고 합니다. 먼저 평균 제곡 오차 식을 다시 살펴보겠습니다.

(식5)
$$평균제곱오차(MSE) = \frac{1}{N} \sum (y_i - \hat{y_i})^2$$

(식6)
$$\hat{y}=ax+b$$

식(5)에서 $\hat{y_i}$는 식(6)에서 $x=x_i$때의 $\hat{y}$의 값이다.따라서  $\hat{y_i}$은 식(7)과 같이 쓸 수 있다.

식(7)
$$\hat{y_i} = ax_i+b$$

식(5)에 식(7)을 대입하여 식(8)와 같이 쓸 수 있다.  
식(8)
$$평균제곱오차(MSE) = \frac{1}{N} \sum (y_i -(ax_i+b) )^2$$


먼저 식(5)을 식(9)와 같이 풀어쓸 수 있다.   
식(9)
$$\frac{1}{N} \sum (y_i - \hat{y_i})^2  =\frac{1}{N} \sum (y_i - \hat{y_i})(y_i - \hat{y_i}) =\frac{1}{N} \sum({y_i}^2 - 2y_i\hat{y_i} + \hat{y_i}^2) $$  

여기서 $({y_i}^2 - 2y_i\hat{y_i}+\hat{y_i}^2)$에 식(7)를 대입하여 정리해보면  

$${y_i}^2-2y_i(ax_i+b)+(ax_i+b)^2 = {y_i}^2 - 2y_i(ax_i+b)+(ax_i+b)^2= {y_i}^2 -2a{x_i} {y_i} - 2b{y_i} +(ax_i+b)(ax_i+b) ={y_i}^2 -2a{x_i} {y_i} - 2b{y_i} + a^2x_i^2 + 2abx_i+b^2$$

따라서 식(5)을 식(10)과 같이 표현할 수 있다.  
식(10)

$$평균제곱오차(MSE) = \frac{1}{N} \sum (y_i - \hat{y_i})^2 = \frac{1}{N} \sum ({y_i}^2 -2a{x_i} {y_i} - 2b{y_i} + a^2{x_i}^2 + 2abx_i+b^2)$$

식(10)을 $a$에대해 (편)미분하면 식(11)과 같다.  
식(11)
$$ \frac{2}{N} \sum -x_i(y_i-(ax_i+b))$$

식(10)을 $b$에 대해 (편)미분하면 식(12)과 같다.  
식(12)  
$$ \frac{2}{N} \sum -(y_i-(ax_i+b))$$






```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
N = len(x)
a = 3                     # 임의의 예측 선의 기울기
b = 70                    # 임의의 예측 선의 y절편
y_hat = a * x + b         # 임의의 예측 선 생성
```


```python
# 예측 선의 기울기 a가 3일 때 에러(e)의 기울기.
a_diff = (2/N) * sum(-x*(y - y_hat) )
b_diff = (2/N) * sum(-(y - y_hat))
```


```python
print( a_diff )
print( b_diff)
```

    -48.0
    -11.0


우리가 해야할 일은 오차 식의 기울기가 0인 지점을 찾는 것이다. 다른 말로 오차 식($e$)의 기울기가 0일 때의 예측 선의 기울기($a$)를 찾는 것입니다. 이를 위해 다음 과정을 진행합니다. 

1. 임의의 위치($a_1$)에서 오차식($e$)의 기울기를 구한다.
2. 구한 기울기가 양수라면 $a_1$ 보다 더 작은 $a$ 값을 선택하고
   구한 기울기 값이 음수라면 $a_1$보다 더 큰 $a$값은 선택하여 기울기를 값을 구한다. 구한 기울기 값이 0에 매우가까울 때까지 2번 과정을 반복한다.


```python
lr = 0.03
```


```python
# 이 코드 cell를 반복해서 실행시키면서 a_diff와 b_diff의 값 변화를 관찰하시요

a = a - lr * a_diff       # a_diff가 음수라면 다음 a 값은 이전 a 값 보다 커짐.
                          # a_diff가 양수라면 다음 a 값은 이전 a 값 보다 작아짐.
b = b - lr * b_diff
y_hat = a * x + b         # 임의의 예측 선 생성

a_diff = (2/N) * sum(-x*(y - y_hat) )
b_diff = (2/N) * sum(-(y - y_hat))

print(a_diff)
print(b_diff)
```

    41.699999999999974
    4.059999999999995


### 과제
 - 과제 1 : 식(10)을 $a$에 대해 편미분하는 과정을 식으로 보이십시요.
 - 과제 2 : 식(10)을 $b$에 대해 편미분하는 과정을 식으로 보이십시요.

과제 1에 대하여  
$\frac{\partial}{\partial a} \frac{1}{N} \sum ({y_i}^2 -2a{x_i} {y_i} - 2b{y_i} + a^2{x_i}^2 + 2abx_i+b^2)$  
$= \frac{1}{N} \sum{( -2{x_i} {y_i} + 2a{x_i}^2 + 2bx_i)}$  
$= \frac{1}{N} \sum{-2x_i(y_i-ax_i-b)}$  
$= \frac{2}{N} \sum{-x_i(y_i-(ax_i+b))}$  

과제 2에 대해서


***

### 1. 환경 준비


```python
import numpy as np
import matplotlib.pyplot as plt
```

### 2. 데이터 준비


```python
# 공부 시간 X와 성적 y의 넘파이 배열을 만듭니다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
```

### 3. 데이터 분포 확인


```python
# 데이터의 분포를 그래프로 나타냅니다.
plt.scatter(x, y)
plt.show()
```

<img width="370" alt="스크린샷 2022-10-12 오전 10 08 41" src="https://user-images.githubusercontent.com/87309905/195225913-b13e176c-ccd3-4f8e-8d2c-3999bf0e72c6.png">


### 4. 실행을 위한 변수 설정


```python
# 기울기 a와 y 절편 b의 값을 초기화합니다.
a = 0
b = 0

# 학습률을 정합니다.
lr = 0.03

# 몇 번 반복될지 설정합니다. 
epochs = 2001 
```

5. 경사 하강법


```python
# x 값이 총 몇 개인지 셉니다.
n=len(x)

# 경사 하강법을 시작합니다.
for i in range(epochs):                  # 에포크 수 만큼 반복
    
    y_pred = a * x + b                   # 예측 값을 구하는 식입니다. 
    error = y - y_pred                   # 실제 값과 비교한 오차를 error로 놓습니다.
    
    a_diff = (2/n) * sum(-x * (error))   # 오차 함수를 a로 편미분한 값입니다. 
    b_diff = (2/n) * sum(-(error))       # 오차 함수를 b로 편미분한 값입니다. 
    
    a = a - lr * a_diff     # 학습률을 곱해 기존의 a 값을 업데이트합니다.
    b = b - lr * b_diff     # 학습률을 곱해 기존의 b 값을 업데이트합니다.
    
    if i % 100 == 0:        # 100번 반복될 때마다 현재의 a 값, b 값을 출력합니다.
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))        
```

    epoch=0, 기울기=27.8400, 절편=5.4300
    epoch=100, 기울기=7.0739, 절편=50.5117
    epoch=200, 기울기=4.0960, 절편=68.2822
    epoch=300, 기울기=2.9757, 절편=74.9678
    epoch=400, 기울기=2.5542, 절편=77.4830
    epoch=500, 기울기=2.3956, 절편=78.4293
    epoch=600, 기울기=2.3360, 절편=78.7853
    epoch=700, 기울기=2.3135, 절편=78.9192
    epoch=800, 기울기=2.3051, 절편=78.9696
    epoch=900, 기울기=2.3019, 절편=78.9886
    epoch=1000, 기울기=2.3007, 절편=78.9957
    epoch=1100, 기울기=2.3003, 절편=78.9984
    epoch=1200, 기울기=2.3001, 절편=78.9994
    epoch=1300, 기울기=2.3000, 절편=78.9998
    epoch=1400, 기울기=2.3000, 절편=78.9999
    epoch=1500, 기울기=2.3000, 절편=79.0000
    epoch=1600, 기울기=2.3000, 절편=79.0000
    epoch=1700, 기울기=2.3000, 절편=79.0000
    epoch=1800, 기울기=2.3000, 절편=79.0000
    epoch=1900, 기울기=2.3000, 절편=79.0000
    epoch=2000, 기울기=2.3000, 절편=79.0000


 기울기 $a$의 값이 2.3에 수렵하는 것과 $y$절편 $b$가 79에 수렴하는 과정을 봤습니다. 최적의 기울기($a$)가 2.3이고 최적의 $y$절편 $b$가 79라는 것은 최소 제곱법을 이용해서 알고 있었습니다. 여기서는 최소 제곱법을 사용하지 않고 '평균제곱오차'와 '경사 하강법'을 이용해서 원하는 값을 구할 수 있다는 것을 확인했습니다. 이와 똑같은 방법을 $x$가 여러 개인 다중 선형 회귀에서도 사용할 수 있습니다. 

### 6. 그래프 확인


```python
# 앞서 구한 최종 a 값을 기울기, b 값을 y 절편에 대입하여 그래프를 그립니다.
y_pred = a * x + b      

# 그래프 출력
plt.scatter(x, y)
plt.plot(x, y_pred,'r')
plt.show()
```

<img width="379" alt="스크린샷 2022-10-12 오전 10 09 50" src="https://user-images.githubusercontent.com/87309905/195226033-4838f8ed-422d-4839-8195-351967b47a40.png">


***


[과제]   
변수가 늘었고 기호가 복잡해졌지만 기존에 해봤던 계산을 요구할 것입니다. 다음 식(1)을 $a_1$에 대해 편미분 하시오.  
(식1)  
$$MSE(e) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$  
여기서 $\hat{y_i}$는 식(2)에 주여졌습니다.  
(식2)  
$$\hat{y_i} = a_1x_{1,i} + a_2x_{2,i} + b$$


[풀이]  
식(1)과 식(2)에서, $n$은 데이터의 개수이며 $y_i$는 각각의 실제 데이터이고 $\hat{y_i}$는 각각의 예측 데이터이다.   
(식3)  
$$e = \frac{1}{n} \sum(y_i-\hat{y_i})(y_i-\hat{y_i}) =\frac{1}{n} \sum({y_i}^2 - 2y_i\hat{y_i} + \hat{y_i}^2)$$  

식(3)에 식(2)를 대입하여 정리하면 식(4)와 같습니다.  
(식4)  
$$e=\frac{1}{n}\sum({y_i}^2 - 2 {y_i} ({a_1}{x_{1,i}} + {a_2} {x_{2,i}} + b) + ({a_1}{x_{1,i}} + {a_2} {x_{2,i}} + b)^2)$$

식(5)와 같이 정의하면 식(6)과 같은 표현이 가능하다.  

(식5)  
$$ A=({a_1}x_{1,i} + {a_2}x_{2,i})$$

(식6)
$$ ({a_1}{x_{1,i}} + {a_2} {x_{2,i}} + b)^2 = (A +b)^2$$

식(6)을 풀어서 쓰면 식(7)과 같다. 식(7)에 식(5)를 대입해서 풀면 식(8)과 같다.  
(식7) 
$$(A+b)^2 = A^2 + 2Ab + b^2$$
(식8)  
$$(A+b)^2 = ({a_1}x_{1,i} + {a_2}x_{2,i})({a_1}x_{1,i} + {a_2}x_{2,i})+ 2({a_1}x_{1,i} + {a_2}x_{2,i})b + b^2 $$  
식(8)을 다시 고쳐 쓰면 식(9)와 같다.  
(식9)
$$(A+b)^2= {a_1}^2 x_{1,i}^2 + 2{a_1}{a_2} {x_{1,i}x_{2,i}}+{a_2}^2 x_{2,i}^2 
+2 {a_1} b x_{1,i} +2{a_2} b x_{2,i} 
+b^2$$  

식(5)와 식(9)를 이용하여 식(3)을 고쳐쓰면 식(10)과 같다.  
(식10)
$$ e = \frac{1}{n}\sum(B)$$  
여기서  
(식11)  
$$B = {y_i}^2 - 2 {y_i} ({a_1}{x_{1,i}} + {a_2} {x_{2,i}} + b) + {a_1}^2 x_{1,i}^2 + 2{a_1}{a_2} {x_{1,i}x_{2,i}}+{a_2}^2 x_{2,i}^2 
+2 {a_1} b x_{1,i} +2{a_2} b x_{2,i} 
+b^2$$

식(11)을 다시 정리하면  (식12)와 같습니다.  
(식12)  
$$B = {y_i}^2 - 2{a_1}x_{1,i}y_i +2{a_2} x_{2,i}y_i + 2b{y_i}+{a_1}^2 x_{1,i}^2 + 2a_1{a_2}x_{1,i}x_{2,i} + {a_2}^2 x_{2,i}^2+2{a_1}bx_{1,i}+ 2{a_2} b x_{2,i} +b^2$$

식(12)를 식(10)에 넣어서 표현하면 식(13)과 같다.  
(식13)
$$ e = \frac{1}{n}\sum({y_i}^2 - 2{a_1}x_{1,i}y_i +2{a_2} x_{2,i}y_i + 2b{y_i}+{a_1}^2 x_{1,i}^2 + 2a_1{a_2}x_{1,i}x_{2,i} + {a_2}^2 x_{2,i}^2+2{a_1}bx_{1,i}+ 2{a_2} b x_{2,i} +b^2)$$


질문은 식(13)을 $a_1$에 대해 편미분 하라는 것과 같다. 식(13)을 $a_1$에 대해 미분하면 식(14)을 얻게 된다. 

(식 14)  
$$ \frac{\partial{e}}{\partial{a_1}} = \frac{1}{n} \sum( -2x_{1,i}{y_i}+2{a_1} x_{1,i}^2 + 2{a_2} {x_{1,i}} {x_{2,i}} + 2b{x_{1,i}}) \\
= \frac{1}{n}\sum(-2x_{1,i}(y_i -{a_1}x_{1,i} -{a_2} x_{2,i}-b)) \\
= \frac{2}{n}\sum(-x_{1,i}(y_i - ({a_1}x_{1,i} +{a_2} x_{2,i}+b)))\\
= \frac{2}{n}\sum(-x_{1,i}(y_i - \hat{y_i}))$$  

**우리가 유도한 식(14)은 어떤 의미가 있는 것인가?**

## 3. 다중 선형 회귀의 개요


## 4. 파이썬 코딩으로 확인하는 다중 선형 회귀
 앞서 학생들이 공부한 시간에 따른 예측 선을 그리려고 기울기 $a$와 $y$ 절편 $b$를 구했습니다. 그런데 예측한 성적과 실제 학생들이 취득한 성적 사이에는 차이가 있습니다. 4시간 공부한 한색의 경우 88점을 예측했으나 실제 받은 점수는 93점을 받았고 6시간 공부한 학생은 93점을 받을 것으로 예측했으나 실제로는 91점을 받았습니다. 이러한 차이가 나는 이유는 공부한 시간 이외의 다른 요소가 성적에 영향을 주었다고 보는 것이 타당할 것입니다. 

 더 정확한 예측을 하려면 추가 정보를 고려해야하며 공부한 시간 외의 정보를 추가해서 새로운 예측값을 구해봐야할 것 입니다. 공부한 시간($x$)만 고려했던 기존 접근과 달리 '과외 수업 횟수'를 고려해보려고 합니다. 따라서 공부한 시간을 $x_1$이라고 하고 과외 수업 횟수'를 $x_2$라고 표기하겠습니다. 이 두 정보와 취득 점수를 아래 표로 정리했습니다.  

|취득 성적($y$)|81|93|91|97|
|---|---|---|---|---|
|공부한 시간($x_1$)|2|4|6|8|
|과외 수업 횟수($x_2$)|0|4|2|3|

 고려해야할 독립변수는 $x_1$과 $x_2$ 두 개이므로 취득 성적을 나타내는 종속 변수 $y$는 식(1)과 같이 나타낼 수 있습니다.   
(식1)
$$y = a_1x_1 + a_2x_2 + b $$  

그러면 $a_1$과 $a_2$를 어떻게 구할 수 있을까요? 앞서 배운 경사하강법을 그대로 적용하면 됩니다. 

 먼저 성적에 영향을 주는 두가지 정보,$x_1$과 $x_2$에 따른 성적($y$)을 그래프로 나타내보겠습니다. 이 그래프는 앞서 $x$와 $y$ 두 축이었던 것과 달리 $x_1$, $x_2$ 그리고 $y$ 이렇게 세 개의 축이 필요합니다. 




```python
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([91, 93, 91, 97])
```


```python
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.show()
```

<img width="576" alt="스크린샷 2022-10-12 오전 10 10 08" src="https://user-images.githubusercontent.com/87309905/195226067-3d9d5cce-49cb-4b10-a0f2-3e04e459800e.png">



코드의 형태는 크게 다르지 않습니다. 다만 고려할 사항이 2개로 늘어서 $x_1$, $x_2$ 두 개로 늘었고 따라서 기울기도 $a_1$과 $a_2$로 늘었습니다. 앞서 수행했던 방법대로 경사하강법을 적용해 보겠습니다.
$$\hat{y} = a_1 x_1 + a_2 x_2 + b$$


```python
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([91, 93, 91, 97])

# 임의로 a1, a2 그리고 b 값을 결정
a1 = 2
a2 = 2
b = 78

y_hat = a1 * x1 + a2*x2 + b      # 기울기와 절편 자리에 a1, a2, b를 각각 배치.
error = y - y_hat               # 에러, 즉 실제 값 - 예측 값
```

오차 함수를 $a_1, a_2, b$로 각각 편미분한 값을 a1_diff, a2_diff, b_diff라고 할 때 이를 구하는 코드는 아래와 같다고 받아들이겠습니다. 


```python
n = len(x1)     # n은 데이터 개수, 이전 예제에서는 변수 이름으로 N을 사용하기도 했음.
a1_diff = (2/n) * sum(-x1*error)      # 오차 함수를 a1로 편미분한 값입니다.
a2_diff = (2/n) * sum(-x2*error)      # 오차 함수를 a2로 편미분한 값입니다.
b_diff = (2/n) * sum(-error)          # 오차 함수를 b로 편미분한 값입니다.
```

학습률을 곱해 기존의 기울기와 절편을 업데이트한 값을 구합니다.


```python
lr = 0.003
```


```python
a1 = a1 - lr * a1_diff
a2 = a2 - lr * a2_diff
b = b - lr * b_diff

y_hat = a1 * x1 + a*x2 + b 
print(y-y_hat)
```

    [ 9.081      -2.03500016 -3.35100008 -3.56700012]


위 코드 셀을 정리함


```python
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([91, 93, 91, 97])

# 임의로 a1, a2 그리고 b 값을 결정
a1 = 2
a2 = 2
b = 78

n = len(x1)     # n은 데이터 개수, 이전 예제에서는 변수 이름으로 N을 사용하기도 했음.
lr = 0.03

for i in range(1700):
  y_hat = a1 * x1 + a2 * x2 + b       # 예측 식을 생성
  error = y - y_hat                   # 에러, 즉 실제 값 - 예측 값
 
  a1_diff = (2/n) * sum(-x1*error)    # a1_diff는 오차 함수를 a1로 편미분한 값입니다.
  a2_diff = (2/n) * sum(-x2*error)    # a2_diff는 오차 함수를 a2로 편미분한 값입니다.
  b_diff = (2/n) * sum(-error)        # b_diff는 오차 함수를 b로 편미분한 값입니다.

  a1 = a1 - lr * a1_diff              # 오차 함수의 a1에 대한 기울기가 0이 되는 지검을 찾기 위해 새로운 a1 값 생성
  a2 = a2 - lr * a2_diff              # 오차 함수의 a2에 대한 기울기가 0이 되는 지검을 찾기 위해 새로운 a2 값 생성
  b = b - lr * b_diff                 # 오차 함수의 b에 대한 기울기가 0이 되는 지검을 찾기 위해 새로운 b 값 생성
 
print(y-y_hat)
```

    [7.43350097e+136 2.05103771e+137 2.42559674e+137 3.26672005e+137]


***

## 1. 환경 준비


```python
import numpy as np
import matplotlib.pyplot as plt
```

## 2. 데이터 준비


```python
# 공부 시간 x1과 과외 시간 x2, 그 성적 y의 넘파이 배열을 만듭니다. 
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([81, 93, 91, 97]) 
```

## 3. 데이터 분포 확인


```python
# 데이터의 분포를 그래프로 나타냅니다.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, x2, y)
plt.show()
```

<img width="348" alt="스크린샷 2022-10-12 오전 10 10 23" src="https://user-images.githubusercontent.com/87309905/195226086-f0c6d1c5-9a5f-4c97-8faa-1741f82ca0a6.png">

   


## 4. 실행을 위한 변수 설정


```python
# 기울기 a와 절편 b의 값을 초기화합니다.
a1 = 0
a2 = 0
b = 0

# 학습률을 정합니다.
lr = 0.01 

# 몇 번 반복될지 설정합니다.
epochs = 2001 
```

## 5. 경사 하강법


```python
# x 값이 총 몇 개인지 셉니다. x1과 x2의 수가 같으므로 x1만 세겠습니다. 
n=len(x1)

# 경사 하강법을 시작합니다.
for i in range(epochs):                  # 에포크 수 만큼 반복
    
    y_pred = a1 * x1 + a2 * x2 + b       # 예측 값을 구하는 식을 세웁니다
    error = y - y_pred                   # 실제 값과 비교한 오차를 error로 놓습니다.
    
    a1_diff = (2/n) * sum(-x1 * (error)) # 오차 함수를 a1로 편미분한 값입니다. 
    a2_diff = (2/n) * sum(-x2 * (error)) # 오차 함수를 a2로 편미분한 값입니다. 
    b_diff = (2/n) * sum(-(error))       # 오차 함수를 b로 편미분한 값입니다. 
    
    a1 = a1 - lr * a1_diff  # 학습률을 곱해 기존의 a1 값을 업데이트합니다.
    a2 = a2 - lr * a2_diff  # 학습률을 곱해 기존의 a2 값을 업데이트합니다.
    b = b - lr * b_diff     # 학습률을 곱해 기존의 b 값을 업데이트합니다.
    
    if i % 100 == 0:        # 100번 반복될 때마다 현재의 a1, a2, b 값을 출력합니다.
        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" %(i, a1, a2, b))        
```

    epoch=0, 기울기1=9.2800, 기울기2=4.2250, 절편=1.8100
    epoch=100, 기울기1=9.5110, 기울기2=5.0270, 절편=22.9205
    epoch=200, 기울기1=7.3238, 기울기2=4.2950, 절편=37.8751
    epoch=300, 기울기1=5.7381, 기울기2=3.7489, 절편=48.7589
    epoch=400, 기울기1=4.5844, 기울기2=3.3507, 절편=56.6800
    epoch=500, 기울기1=3.7447, 기울기2=3.0608, 절편=62.4448
    epoch=600, 기울기1=3.1337, 기울기2=2.8498, 절편=66.6404
    epoch=700, 기울기1=2.6890, 기울기2=2.6962, 절편=69.6938
    epoch=800, 기울기1=2.3653, 기울기2=2.5845, 절편=71.9160
    epoch=900, 기울기1=2.1297, 기울기2=2.5032, 절편=73.5333
    epoch=1000, 기울기1=1.9583, 기울기2=2.4440, 절편=74.7103
    epoch=1100, 기울기1=1.8336, 기울기2=2.4009, 절편=75.5670
    epoch=1200, 기울기1=1.7428, 기울기2=2.3695, 절편=76.1904
    epoch=1300, 기울기1=1.6767, 기울기2=2.3467, 절편=76.6441
    epoch=1400, 기울기1=1.6286, 기울기2=2.3301, 절편=76.9743
    epoch=1500, 기울기1=1.5936, 기울기2=2.3180, 절편=77.2146
    epoch=1600, 기울기1=1.5681, 기울기2=2.3092, 절편=77.3895
    epoch=1700, 기울기1=1.5496, 기울기2=2.3028, 절편=77.5168
    epoch=1800, 기울기1=1.5361, 기울기2=2.2982, 절편=77.6095
    epoch=1900, 기울기1=1.5263, 기울기2=2.2948, 절편=77.6769
    epoch=2000, 기울기1=1.5191, 기울기2=2.2923, 절편=77.7260



```python
# 실제 점수와 예측된 점수를 출력합니다.
print("실제 점수:", y)
print("예측 점수:", y_pred)
```

    실제 점수: [81 93 91 97]
    예측 점수: [80.76387645 92.97153922 91.42520875 96.7558749 ]



```python
plt.plot(y, 'o')
plt.plot(y_pred, 's')
plt.title('y vs. $\hat{y}$')
plt.ylabel('y')
plt.grid()

```

<img width="382" alt="스크린샷 2022-10-12 오전 10 10 39" src="https://user-images.githubusercontent.com/87309905/195226107-65681215-ca2d-43e5-b028-f3b80a8da738.png">

 
    


## 5-1. 텐서플로에서 실행하는 선형 회귀 모델

 우리는 머신 러닝의 기본인 선형회귀에 대해 배우고 있습니다. 그런데 우리 목표는 딥러닝이지요. 2장에서 잠시 살펴보았지만, 앞으로 우리는 딥러닝을 실행하기 위해 텐서플로라는 라이브러리의 케라스 API를 불러와 사용할 것입니다. 따라서 지금까지 배운 선형 회귀의 개념과 팁러닝 라이브러리들이 어떻게 연결되는지 살펴볼 필요가 있습니다. 이를 통해 텐서플로우 및 케라스의 사용법을 익히는 것은 물론이고 딥러닝 자체에 대한 학습도 한걸음 더 나가게 될 것입니다. 
 선형 회귀는 현상을 분석하는 방법의 하나입니다. 머신 러닝은 이러한 분석 방법을 이용해 예측 모델을 만든는 것이지요. 따라서 두 분야에서 사용하는 용어가 약간 다릅니다. 예를 들어 함수 $y=ax+b$라는 공부한 시간과 성적의 관계를 유추하기 위해 필요했던 식이었습니다. 문제를 해결하기 위해 가정한 식을 머신 러닝에서는 **가설 함수(hypothesis)**하며 **$H(x)$**라고 표기합니다. 또 기울기 $a$는 변수 $x$에 어느 정도 가중치를 두는지 결정하므로 **가중치(weight)**라고 하며 $w$로 표시합니다. 절편 $b$는 데이터의 특성에 따라 따로 부여되는 값이므로 **편향(bias)**이라고 하며 $b$로 표시합니다. 따라서 우리가 앞서 사용한 $y=ax+b$를 머신 러닝에서는 식(1)과 같이 표기합니다.  
(식1)
$$ y = ax + b → H(x)=wx+b$$  

또한 평균 제곡 오차처럼 실제 값과 예측한 값 사이의 오차를 나타내는 식을 **손실 함수(loss function)**이라고 합니다. 

$$평균 제곱 오차 → 손실 함수(loss function)$$

최적의 기울기와 절편을 찾기 위해 앞서 경사 하강법을 배웠지요? 딥러닝에서는 이를 **옵티마이저(optimizer)**라고 합니다. 앞서 사용한 경사 하강법은 딥러닝에서 사용되는 여러 옵티마이저 중 하나였습니다. 경사 하강법 외에 옵티마이저에 대해서는 9장에서 상세히 배웁니다. 




이제부터는 손실 함수, 옵티마이저라는 용어를 사용해서 설명하겠습니다. 먼저 텐스플로에 포함된 케라스 API 중 필요한 함수들을 다음과 같이 불러 옵니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
x = np.array([2,4,6,8])
y = np.array([81,93,91, 97])
```

Sequential() 함수와 Dense() 함수는 2장에서 이미 소개한 바 있습니다. 이 함수를 불러와 선형 회귀를 실행하는 코드는 다음과 같습니다.



```python
# 이 코드 cell 실행 시 주석 처리된 아래 코드를 주석 해제하여 실행 시킬 것. 
# 이 코드 cell를 실행 키시면 다소 시간이 걸림.
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))       # 1
model.compile(optimizer='sgd', loss='mse')                  # 2
model.fit(x, y, epochs=2000)                                # 3
```

    Epoch 1/2000
    1/1 [==============================] - 1s 521ms/step - loss: 9921.2422
    Epoch 2/2000
    1/1 [==============================] - 0s 19ms/step - loss: 2283.4036
    Epoch 3/2000
    1/1 [==============================] - 0s 19ms/step - loss: 1156.2917
    Epoch 4/2000
    1/1 [==============================] - 0s 20ms/step - loss: 985.4776
    Epoch 5/2000
    ...
    Epoch 1998/2000
    1/1 [==============================] - 0s 7ms/step - loss: 8.3022
    Epoch 1999/2000
    1/1 [==============================] - 0s 5ms/step - loss: 8.3022
    Epoch 2000/2000
    1/1 [==============================] - 0s 6ms/step - loss: 8.3022





    <keras.callbacks.History at 0x7f38b5212150>



 위 코드 cell에서 주석 1, 2, 3의 세 줄의 코드에 앞서 공부한 모든 것이 담겨져 있습니다. 어떻게 설정하는지 살펴보겠습니다. 주석 1번에서, 가설 함수는 $H(x)=wx+b$입니다. 이때 출력되는 값(성적)이 하나씩이므로 Dense() 함수의 첫 번째 인자에 1이라고 설정합니다. 입력될 변수(공부 시간)도 하나뿐이므로 input_dim 역시 1이라고 설정합니다. 입력된 값을 다음 층으로 넘길 때 각 값을 어떻게 처리할지를 결정하는 함수를 활성화 함수라고 합니다. activation은 활성화 함수를 정의하는 옵션입니다. 우리는 지금 선형 회귀를 고려하고 있으므로 'linear'라고 적어주면 됩니다. 딥러닝 목적에 따라 다른 활성화 함수를 넣을 수도 있는데, 예를 들어 다음 절에서 배울 시그모이드(sigmoid) 함수가 사용할 것이라면 activation 옵션에 'sigmoid'를 할당하면 됩니다. 딥러닝에서 사용하는 여러 가지 활성화 함수에 대해서는 9장에서 상세히 배웁니다. 주석 2번에서, 앞서 배운 경사 하강법을 실행하려면 옵티마이저에 sgd라고 설정합니다. 손실 함수는 평균 제곱 오차를 사용할 것이기 때문에 mse라고 설정합니다. 주석 3번에서, 앞서 따로 코딩했던 epochs 숫자를 model.fit()함수에 전달합니다.  
 공부 시간($x$)이 입력되었을 때의 예측점수는 model.predict(x)로 알 수 있습니다. 예측 점수로 그래프를 그려보면 다음과 같습니다.




```python
plt.scatter(x,y)
plt.plot(x, model.predict(x), 'sr')
plt.show()
```


<img width="380" alt="스크린샷 2022-10-12 오전 10 10 55" src="https://user-images.githubusercontent.com/87309905/195226136-49fb5a6c-239f-4dda-97bd-433cde57f9f6.png">
    


## 1. 환경 준비


```python
import numpy as np
import matplotlib.pyplot as plt

# 텐서플로의 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## 2. 데이터 준비


```python
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
```

## 3. 모델 실행


```python
model = Sequential()

# 출력 값, 입력 변수, 분석 방법에 맞게끔 모델을 설정합니다. 
model.add(Dense(1, input_dim=1, activation='linear'))

# 오차 수정을 위해 경사 하강법(sgd)을, 오차의 정도를 판단하기 위해 평균 제곱 오차(mse)를 사용합니다. 
model.compile(optimizer='sgd', loss='mse')

# 오차를 최소화하는 과정을 2000번 반복합니다.
model.fit(x, y, epochs=2000)
```

    Epoch 1/2000
    1/1 [==============================] - 0s 326ms/step - loss: 7500.6973
    Epoch 2/2000
    1/1 [==============================] - 0s 9ms/step - loss: 1936.7855
    ...
    Epoch 1999/2000
    1/1 [==============================] - 0s 5ms/step - loss: 8.3022
    Epoch 2000/2000
    1/1 [==============================] - 0s 6ms/step - loss: 8.3023





    <keras.callbacks.History at 0x7f38b50a5950>



## 4. 그래프로 확인


```python
plt.scatter(x,y)
plt.plot(x, model.predict(x), 'sr-')
plt.show()
```

<img width="379" alt="스크린샷 2022-10-12 오전 10 11 09" src="https://user-images.githubusercontent.com/87309905/195226162-a2610162-2f36-48c9-ab38-7bbe4c6417cf.png">

   


## 5. 모델 테스트


```python
# 임의의 시간을 집어넣어 점수를 예측하는 모델을 테스트해 보겠습니다.

hour = 7
prediction = model.predict([hour])

print("%.f시간을 공부할 경우의 예상 점수는 %.02f점입니다" % (hour, prediction))
```

    7시간을 공부할 경우의 예상 점수는 95.12점입니다


# 5-2 텐서플로에서 실행하는 다중 선형 회귀 모델
## 1. 환경 준비


```python
import numpy as np
import matplotlib.pyplot as plt

# 텐서플로의 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## 2. 데이터 준비


```python
x = np.array([[2, 0], [4, 4], [6, 2], [8, 3]])
y = np.array([81, 93, 91, 97]) 
```

## 3. 모델 실행


```python
model = Sequential()

# 입력 변수가 2개(학습 시간, 과외 시간)이므로 input_dim에 2를 입력합니다. 
model.add(Dense(1, input_dim=2, activation='linear'))     # 1
model.compile(optimizer='sgd' ,loss='mse')                # 2
model.fit(x, y, epochs=2000)
```

    Epoch 1/2000
    1/1 [==============================] - 0s 307ms/step - loss: 7818.2349
    Epoch 2/2000
    1/1 [==============================] - 0s 6ms/step - loss: 1411.4215
    Epoch 3/2000
    1/1 [==============================] - 0s 5ms/step - loss: 955.3619
    ...
    Epoch 1995/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0744
    Epoch 1996/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0743
    Epoch 1997/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0743
    Epoch 1998/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0743
    Epoch 1999/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0743
    Epoch 2000/2000
    1/1 [==============================] - 0s 5ms/step - loss: 0.0743





    <keras.callbacks.History at 0x7f38b27c02d0>



주석 1번에서, 가설 함수는  H(x)=wx+b 입니다. 이때 출력되는 값(성적)이 하나씩이므로 Dense() 함수의 첫 번째 인자에 1이라고 설정합니다. 입력될 변수(공부 시간, 과외 학습 시간)가 두 개이므로 input_dim에 2을 할당합니다. 입력된 값을 다음 층으로 넘길 때 각 값을 어떻게 처리할지를 결정하는 함수를 활성화 함수라고 합니다. activation은 활성화 함수를 정의하는 옵션입니다. 우리는 지금 선형 회귀를 고려하고 있으므로 'linear'라고 적어주면 됩니다. 주석 2번에서, 앞서 배운 경사 하강법을 실행하려면 옵티마이저에 sgd라고 설정합니다. 손실 함수는 평균 제곱 오차를 사용할 것이기 때문에 mse라고 설정합니다. 주석 3번에서, 앞서 따로 코딩했던 epochs 숫자를 model.fit()함수에 전달합니다.

## 4. 모델 테스트


```python
# 임의의 학습 시간과 과외 시간을 집어넣어 점수를 예측하는 모델을 테스트해 보겠습니다.

hour = 7
private_class = 4
prediction = model.predict([[hour, private_class]])

print("%.f시간을 공부하고 %.f시간의 과외를 받을 경우, 예상 점수는 %.02f점입니다" % (hour, private_class, prediction))
```

    7시간을 공부하고 4시간의 과외를 받을 경우, 예상 점수는 97.53점입니다


***

# 6장. 로지스틱 회귀 모델: 참 거짓 판단하기

 법정 드라마나 영화를 보면 검사가 피고인을 다그치는 장면이 종종 나옵니다. 검사의 예리한 질문에 피고인이 당황한 표정으로 변명을 늘어놓을 때 검사가 이렇게 소리칩니다. "예, 아니오로만 대답하세요"

 때로 할 말이 많아도 "예" 혹은 "아니요"로만 대답해야할 때가 있습니다. 그런데 실은 이와 같은 상황이 딥러닝에서도 끊임없이 일어납니다. 전달받은 정보를 놓고 참과 거짓 중 하나를 판단해 다음 단계로 넘기는 장치들이 딥러닝 내부에서 쉬지 않고 작동하는 것이지요. 딥러닝을 수행하다는 것은 겉으로 드러나지 않는 '미니 판단 장치'들을 이용해서 복잡한 연산을 해낸 끝에 최적의 예측값을 내놓은 작업이라고 할 수 있습니다. 

 이렇게 참과 거짓 중 하나를 내놓은 과정은 로지스틱 회귀(logistic regression)를 거쳐 이루어집니다. 이제 회귀 분석의 또 다른 토대를 이루는 로지스틱 회귀에 대해 알아 보겠습니다. 

## 2.로지스틱 회귀의 정의
 5장에서 공부한 시간과 성적 사이의 관계를 그래프로 나타냈을 때, 그래프의 형태가 직선인 선형 회귀를 사용하는 것이 적절함을 보았다. 그런데 직선이 적절하지 않은 경우가 있습니다. 

|공부한시간|2시간|4시간|6시간|8시간|
|---|:---:|:---:|:---:|:--:|
|성적|81|93|91|97| 




```python
import numpy as np
import matplotlib.pyplot as plt
# 공부 시간 X와 성적 y의 넘파이 배열을 만듭니다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
a = 2.3
b = 79
y_pred = a * x + b      

# 그래프 출력
plt.scatter(x, y)
plt.plot(x, y_pred,'r')
plt.show()
```


<img width="375" alt="스크린샷 2022-10-12 오전 10 11 30" src="https://user-images.githubusercontent.com/87309905/195226194-35f97fa3-1b32-4dad-b2db-e3f8a0ee8c4e.png">
   


점수가 아니라 오직 합격과 불합격만 발표되는 시험이 있다고 합시다. 공부한 시간에 따른 합격 여부를 조사해 보니 아래 표와 같았습니다. 

|공부한 시간|2|4|6|8|10|12|14
|---|:---:|:---:|:---:|:--:|:---:|:---:|:---:|
|합격 여부|불합격|불합격|불합격|합격|합격|합격|합격| 

합격을 1, 불합격을 0이라고 이를 좌표 평면에 표시하면 아래와 같습니다. 선을 그어서 이 점들의 특성을 잘 나타내는 일차 방정식을 만들 수 있을까요? 


```python
x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1])
plt.plot(x,y, 'o')
plt.grid()
plt.show()
```



<img width="371" alt="스크린샷 2022-10-12 오전 10 11 43" src="https://user-images.githubusercontent.com/87309905/195226221-32e982d6-9de9-4fa2-a368-1d6a75fdac98.png">
   



```python
# https://zephyrus1111.tistory.com/103

x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1])
markers, stemlines, baseline = plt.stem(x, y, use_line_collection=True)
baseline.set_visible(False)
stemlines.set_visible(True)
plt.grid()
plt.show()
```


<img width="373" alt="스크린샷 2022-10-12 오전 10 11 56" src="https://user-images.githubusercontent.com/87309905/195226243-3ed2dd35-b500-45f3-9f44-8ff1a310d944.png">


점들의 특성을 정확하게 담아내려면 직선이 아니라 다음과 같은 S자 형태가 되어야 합니다.  
(식1)

$$ f(x) = \frac{1}{1+e^{-(ax+b)}} $$

위 식(1)을 **시그모이드** 함수라고 합니다.


```python
x = np.linspace(-4, 4, 100)
a = 1
b = 0
f1 = 1 / (1 + np.exp(-(a*x+b)) )
a = 2
f2 = 1 / (1 + np.exp(-(a*x+b)) )
a = 4
f3 = 1 / (1 + np.exp(-(a*x+b)) )
plt.plot(x, f1,label = 'a = 1')
plt.plot(x, f2, label = 'a = 2')
plt.plot(x, f3, label = 'a = 4')
plt.legend()
plt.show()
```

<img width="372" alt="스크린샷 2022-10-12 오전 10 12 08" src="https://user-images.githubusercontent.com/87309905/195226261-b486db91-79eb-4ffc-9a85-6bb06a21b6b1.png">

 


**위 그래프 해석**  
$a$값이 거질수록 급격하게 곡선이 꺽이는 것을 확일 할 수 있습니다. 즉, $a$값이 커질 수록 실측 데이터에 가까워진다고 볼 수 있겠습니다.


```python
x = np.linspace(-4, 4, 100)
a = 2
b = 0
f1 = 1 / (1 + np.exp(-(a*x+b)) )
b = -4
f2 = 1 / (1 + np.exp(-(a*x+b)) )
b = 4
f3 = 1 / (1 + np.exp(-(a*x+b)) )
plt.plot(x, f1,label = 'b = 0')
plt.plot(x, f2, label = 'b = -4')
plt.plot(x, f3, label = 'b = 4')
plt.legend()
plt.show()
```


<img width="371" alt="스크린샷 2022-10-12 오전 10 12 18" src="https://user-images.githubusercontent.com/87309905/195226278-8e5f59a3-ddb9-4fa8-9579-4e674d8eb191.png">
    




```python
x = np.linspace(2, 14, 100)
a = 2
b = -4
f1 = 1 / (1 + np.exp(-(a*x+b)) )
b = -8
f2 = 1 / (1 + np.exp(-(a*x+b)) )
b = -12
f3 = 1 / (1 + np.exp(-(a*x+b)) )
plt.plot(x, f1,label = 'b = -4')
plt.plot(x, f2, label = 'b = -8')
plt.plot(x, f3, label = 'b = -12')
plt.legend()
plt.show()
```


    
![png](output_156_0.png)
    


**위 두 그래프 해석**  
그래프는 $b$ 값이 커지면 왼쪽으로 이동하고 $b$값이 작아지면 오른쪽으로 이동한다는 것을 확인했습니다. 

![스크린샷 2022-10-11 오후 5.20.26.png](data:image/png;base64,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31794z/+Y/Unf/In1Stf+cpqjTXWqJZddtkkZydedX88UeQr0qiHm8g9Ho8//nh1/fXXVz/60Y+qf//3f6/+3//7f9WrXvWqap111qnodFTq96OPPlp997vfTXKS+5nPfGb1mte8plp//fWrZz3rWW29TCT/vYR9RiuxcQdLBDnrrLOq//3f/6123HHHgQnTi8AlzOzXgPL2n//5n9UXvvCF6sc//nH1p3/6p2M+wv/93/9drbzyytU73vGO6q/+6q/SM1oJ8Bi0hsgXwPHEE09UJ598cgIacuZEzg033LDabLPNUh7yZ03XURXzfNT9Iu08fj2MZxGOe//991df+cpXqn/9139t6yji/Nmf/Vm1zTbbtD8s4ubpux8k5XJ+73vfq84+++ynvW9hnvvc51a77LJL9Td/8zdt+SLuePJFXrvla9ymMSZ+V155ZfWLX/xivDTL86KBKWtAebvkkkuqBx98MFUKH+Agz1Teu+66q7riiiueBoKedyLP+vHDHx9ynX766W0rME9XpfvzP//z6jvf+U7KR4Tvln7wjbD4xzWXhZyHiWfhF+H5hx/r77TTThsDgmRjtXLxBD7/8A//0NZl8OVO9UeOJrlCRjK4ZgkytuKjlzKQ/f3TP/1T0vW//du/tWXKHne9jLRCjqbA41qEmPzXf/1XtdJKK1Wnnnpqtdpqq6WC2MSs+BUNTFUDytvvfve76uijj260olScIGEPOuig6jnPeU71hz/8IVW4eJa74qhowufx8zATvVapWIPf+ta3Eqg0xZce+su//MtqqaWWStdR8dNNh79FFlmk+uu//uvql7/8ZfXzn/88yYwX0FpvvfWSXgCZbgPN8kgH7xe+8IXVMssskzirt1dffXX1z//8z+k+wkWyoQv+mp2bbrppSgsYyV88j/BTcaWx6KKLpqZuE5/zzz+/euSRRxrfobhkIdOb3/zm6tWvfnUTi7ZfhA8PH05NbPHls265CzcuEIr88MMPV2uuuWb6sj372c+esIIIVmjuaiDev7Lkx1LqRqw9lpbCHxZC8BDPtUoOGDQ/heMXP2GiEsczlSGuPZ8KBZ96GniSIfzdR9jIB7+geBYuf9d0lBOegI+/axRx+KNItx7X84iTAjb84YXC4oz7hqBT8vKuIi8hU7j6LJvyHglGfoXHB01ETmGVmY997GPVi1/84mDbdrsO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)

![스크린샷 2022-10-11 오후 5.19.45.png](data:image/png;base64,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A0+YCwt+/etfN094whOaxz/+8c3jHve4uURVwy5lHHDQayZ1Jvfn+X/+53+ar3zlK80NN9zQ/Md//EeK4ylPeUqz9dZbN5tttlnzpCc9KXEyDzdu1kr7kUceab761a82t99+e/Pf//3fDbs//uM/brbbbrtmtdVWa37ndxbXT8aNV3r33Xdfwnj//fcnFv3v//5vs8oqqzTbb799wjqT8hkVb+FDzG9/+9vNpZde2jz00EOpvMmcddZZp3nFK17RPPWpT03+Ron1cVMgBjqWyDtAGPqxj32s+a//+q/mfe9734wqfMpN/VsiOBDVpl/ljLrSznDYh9l2z9/DD1OnfNxxxzUPPPBA8kLQsEfq41prrdXst99+ze/+7u9OpD4Glocffrg59thjm5/97GdJGEYe4CQwt91222bXXXdNygW7cVJgWbRoUXPCCSekpMMuTJave93rmo022mgxYZ67jxJzpKNDPP/885OApIiFvbSVt87x0EMPbZ73vOd1yjv3Mx3GmfpdvDubLtYp92gUzOuuu65ZfvnlO6EkWmnp4IDy9+tX5r3cc/t+4XEy/Gown/3sZ5sHH3ww2eWaGD8aEa3twgsvTO4aUcTNjJ8443kmpngirn7+w89//ud/Nscff3zzr//6rx0BA1+k+8QnPrG5/PLLmzvuuCNpxREuzH5pDMMNH2GDEa72TxrorLPOagj7wMVEw8AwXRzSkR5BjlftzlB4uHWap5xySvOrX/2qg4t95EE8vSji4D6d/4E1yYj00UcfbTbccMPmn/7pn5q11167UyF6gar2848DvSqPiohU0jap0ARBVFZD4jbFFA17jeEXv/hFx79wfssuu2wy+fm3f/u3NNSi+YgbwZbH/eQnPznZS592EZoHS/HR4GiguX0K0OdPuO9///tJqPTiRTs4AXTPPfek9hCChR9xBXn+vd/7veaFL3xhWI3V1NHE0FW+ciEECDsYV1xxxebZz352en/Ws561mEI0asDSv/jii9O0RZ4W++Br4N5hhx2addddN/c242dxPPOZz1ysfNqBZyUkgbzzzjubl770pc1dd93VPO1pT+ubSDvReNer+VUqiwO0IY3IjwD75S9/mcqXcNtxxx07PS+tjVAIAaICa1RbbrllyhD7s88+OwmoyCE/m266abPSSislq+9+97vN9ddfH87JJAj32GOPlCb/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)

![스크린샷 2022-10-11 오후 5.20.55.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZUAAAGxCAYAAACqZisbAAAKqmlDQ1BJQ0MgUHJvZmlsZQAASImVlgdQU+kWx79700NCS4iAlNA70gkgJfTQe7MRkkBCiTEQUCxYWFyBFUVEBJQFXRBBcC2ArAWxYFsUFbCgC7IoKOtiwYblXWAIu/vmvTfvzHw5v/nnfOc758797hwAyCS2SJQKywOQJswQh/m402Ni4+i4p4AIsAAPDIEpm5MuYoaEBADE5vzf7V0fgKb9HbPpXP/+/381BS4vnQMAFIJwAjedk4bwCWSNckTiDABQlYiuk5UhmuYOhKlipECE705z0iyPTnPCLH+eiYkI8wAAjXSFJ7HZ4iQASGqITs/kJCF5SIsRthByBUKEp+t1SUtbyUW4EWFDJEaE8HR+RsJf8iT9LWeCNCebnSTl2V5mDO8pSBelstf8n4/jf1taqmTuDH1kkfhi3zDEI3VB91JW+ktZmBAUPMcC7kz8DPMlvpFzzEn3iJtjLtvTX7o3NShgjhMF3ixpngxWxBzz0r3C51i8Mkx6VqLYgznHbPH8uZKUSKnO57Gk+bP5EdFznCmICprj9JRw//kYD6kuloRJ6+cJfdznz/WW9p6W/pd+BSzp3gx+hK+0d/Z8/Twhcz5neoy0Ni7P02s+JlIaL8pwl54lSg2RxvNSfaR6ema4dG8G8kLO7w2RPsNktl/IHINw4AkCgRmwAjbAAthk8FZnTDfhsVK0RixI4mfQmcjt4tFZQo65Kd3KwsoKgOm7OvsqvKHN3EGIdm1eW488G7cjiLhrXoumAtB4AADVg/Oadg4AikcBaJnkSMSZsxp6+geDfAXkABWoAA2gg3wLpmuzA07ADXgBPxAMIkAsWA44gA/SgBhkgXVgE8gDBWAH2A3KQRU4AA6BI+AYaAWnwXlwGVwHt0AveAgGwQh4ASbAOzAFQRAOIkMUSAXShPQgE8gKYkAukBcUAIVBsVA8lAQJIQm0DtoCFUDFUDlUDdVDP0OnoPPQVagHug8NQWPQa+gTjIJJMBVWh/XhRTADZsL+cAS8DE6CV8HZcC68HS6Da+BGuAU+D1+He+FB+AU8iQIoGRQNpYUyQzFQHqhgVBwqESVGbUDlo0pRNagmVDuqC3UHNYgaR31EY9EUNB1thnZC+6Ij0Rz0KvQGdCG6HH0I3YK+iL6DHkJPoL9iyBg1jAnGEcPCxGCSMFmYPEwpphZzEnMJ04sZwbzDYrE0rAHWHuuLjcUmY9diC7H7sM3YDmwPdhg7icPhVHAmOGdcMI6Ny8Dl4fbiGnHncLdxI7gPeBm8Jt4K742Pwwvxm/Gl+MP4s/jb+Gf4KYI8QY/gSAgmcAlrCEWEg4R2wk3CCGGKqEA0IDoTI4jJxE3EMmIT8RJxgPhGRkZGW8ZBJlRGILNRpkzmqMwVmSGZjyRFkjHJg7SUJCFtJ9WROkj3SW/IZLI+2Y0cR84gbyfXky+QH5M/yFJkzWVZslzZHNkK2RbZ27Iv5QhyenJMueVy2XKlcsflbsqNyxPk9eU95NnyG+Qr5E/J98tPKlAULBWCFdIUChUOK1xVGFXEKeoreilyFXMVDyheUBymoCg6FA8Kh7KFcpByiTJCxVINqCxqMrWAeoTaTZ1QUlSyUYpSWq1UoXRGaZCGounTWLRUWhHtGK2P9mmB+gLmAt6CbQuaFtxe8F55obKbMk85X7lZuVf5kwpdxUslRWWnSqvKI1W0qrFqqGqW6n7VS6rjC6kLnRZyFuYvPLbwgRqsZqwWprZW7YDaDbVJdQ11H3WR+l71C+rjGjQNN41kjRKNsxpjmhRNF02BZonmOc3ndCU6k55KL6NfpE9oqWn5akm0qrW6taa0DbQjtTdrN2s/0iHqMHQSdUp0OnUmdDV1A3XX6TboPtAj6DH0+Hp79Lr03usb6Efrb9Vv1R81UDZgGWQbNBgMGJINXQ1XGdYY3jXCGjGMUoz2Gd0yho1tjfnGFcY3TWATOxOByT6THlOMqYOp0LTGtN+MZMY0yzRrMBsyp5kHmG82bzV/uUh3UdyinYu6Fn21sLVItTho8dBS0dLPcrNlu+VrK2MrjlWF1V1rsrW3dY51m/UrGxMbns1+m3u2FNtA2622nbZf7OztxHZNdmP2uvbx9pX2/QwqI4RRyLjigHFwd8hxOO3w0dHOMcPxmOOfTmZOKU6HnUYXGyzmLT64eNhZ25ntXO086EJ3iXf50WXQVcuV7Vrj+sRNx43rVuv2jGnETGY2Ml+6W7iL3U+6v/dw9Fjv0eGJ8vTxzPfs9lL0ivQq93rsre2d5N3gPeFj67PWp8MX4+vvu9O3n6XO4rDqWRN+9n7r/S76k/zD/cv9nwQYB4gD2gPhQL/AXYEDQXpBwqDWYBDMCt4V/CjEIGRVyC+h2NCQ0IrQp2GWYevCusIp4SvCD4e/i3CPKIp4GGkYKYnsjJKLWhpVH/U+2jO6OHowZlHM+pjrsaqxgti2OFxcVFxt3OQSryW7l4wstV2at7RvmcGy1cuuLlddnrr8zAq5FewVx+Mx8dHxh+M/s4PZNezJBFZCZcIEx4Ozh/OC68Yt4Y7xnHnFvGeJzonFiaNJzkm7ksb4rvxS/rjAQ1AueJXsm1yV/D4lOKUu5VtqdGpzGj4tPu2UUFGYIry4UmPl6pU9IhNRnmhwleOq3asmxP7i2nQofVl6WwYVGYpuSAwl30mGMl0yKzI/ZEVlHV+tsFq4+sYa4zXb1jzL9s7+aS16LWdt5zqtdZvWDa1nrq/eAG1I2NCZo5OTmzOy0WfjoU3ETSmbft1ssbl489st0Vvac9VzN+YOf+fzXUOebJ44r3+r09aq79HfC77v3ma9be+2r/nc/GsFFgWlBZ8LOYXXfrD8oeyHb9sTt3cX2RXt34HdIdzRt9N156FiheLs4uFdgbtaSugl+SVvd6/YfbXUprRqD3GPZM9gWUBZ217dvTv2fi7nl/dWuFc0V6pVbqt8v4+77/Z+t/1NVepVBVWffhT8eK/ap7qlRr+m9AD2QOaBpwejDnb9xPipvla1tqD2S52wbvBQ2KGL9fb19YfVDhc1wA2ShrHGpY23jngeaWsya6pupjUXHAVHJUef/xz/c98x/2OdxxnHm07onag8STmZ3wK1rGmZaOW3DrbFtvWc8jvV2e7UfvIX81/qTmudrjijdKboLPFs7tlv57LPTXaIOsbPJ50f7lzR+fBCzIW7F0Mvdl/yv3TlsvflC13MrnNXnK+cvup49dQ1xrXW63bXW27Y3jj5q+2vJ7vtultu2t9su+Vwq71ncc/Z2663z9/xvHP5Luvu9d6g3p6+yL57/Uv7B+9x743eT73/6kHmg6mHGwcwA/mP5B+VPlZ7XPOb0W/Ng3aDZ4Y8h248CX/ycJgz/OL39N8/j+Q+JT8tfab5rH7UavT0mPfYredLno+8EL2YGs/7Q+GPypeGL0/86fbnjYmYiZFX4lffXhe+UXlT99bmbedkyOTjd2nvpt7nf1D5cOgj42PXp+hPz6ayPuM+l30x+tL+1f/rwLe0b99EbDF7ZhRAIQtOTATgdR0A5FgAKLcAIC6ZnaVnDJqd/2cI/CeenbdnzA6AWsRFdQDg6wZAObJ0NyIzCKJNj0MRbgC2tpauubl3ZkafNkM+AL/ra7jGOgzob/kA/mGz8/tf6v6nB9Ksf/P/AprNA9x7ADiRAAAAVmVYSWZNTQAqAAAACAABh2kABAAAAAEAAAAaAAAAAAADkoYABwAAABIAAABEoAIABAAAAAEAAAGVoAMABAAAAAEAAAGxAAAAAEFTQ0lJAAAAU2NyZWVuc2hvdOQbWY4AAAHWaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPGV4aWY6UGl4ZWxZRGltZW5zaW9uPjQzMzwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlBpeGVsWERpbWVuc2lvbj40MDU8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpVc2VyQ29tbWVudD5TY3JlZW5zaG90PC9leGlmOlVzZXJDb21tZW50PgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4K0zDRMAAAQABJREFUeAHtnQm4XEWdxUvZkhAIISQEEkjCTsIqq7IvIiAMiDKIyLAIiqOjDoKjjqIzqIOO6+CGoiyKCKIoimhACEvYIbIGSIAgCVs2khDIAr65v5LTVG66+3W/1/3e7e5T39dddWuvU3Xr1L+qbtWbujIVrIyAETACRsAINACBNzcgDkdhBIyAETACRiAiYFJxQzACRsAIGIGGIWBSaRiUjsgIGAEjYARMKm4DRsAIGAEj0DAETCoNg9IRGQEjYASMgEnFbcAIGAEjYAQahoBJpWFQOiIjYASMgBEoFKmkn8xgXrJkSfjud78b8vauNiNgBIyAESgmAoUilTe96U2RQEQis2bNCueff35YuHBhiVjwY2UEjIARMALFRKBQpAJEkIbI5ec//3mYOXNm+P3vf18ilWLC6FwZASNgBIwACBSKVCShkLEXX3wxXHbZZWH11VcPV111Vfj73/9uYgEYKyNgBIxAgREoFKloagtyueOOO8Jzzz0XVllllfDggw+GRYsWRQmmwFg6a0bACBiBjkegUKRCbUAokMuf//znMHDgwCihILVcccUV0dzxNWYAjIARMAIFRqBQpKLpr1deeSVMnDhxBcnkZz/7mUmlwA3JWTMCRsAIgEChSEUL9L/97W/D3LlzSyTy5je/OTz88MPh9ttv97qK260RMAJGoMAIFIpUkFSWLVsW2PUFkUhhv9pqq4VJkyaVdobJzboRMAJGwAgUB4E3eu6C5IlvUx544IG4QK8sIcFAKqyzsAtMC/pyt24EjIARMALFQKAwpCKyuPXWW0uL9ZoOE1SPP/54uPnmm0vTYrK3bgSMgBEwAsVAoDCkAoFwLMu5554bli5dGqfBXnvttSiVvPrqq4HFe+wvuOCCSDpMielXDCidCyNgBIyAEXhT1jEX4o56srF8+fIwbdq0WCuQzNSpU8Ppp58eLrroojB8+PBozzTYpptuusL0mKvRCBgBI2AEioHAqsXIxj+OZ4Ewxo8fH7MEycyZMydOdY0aNSqMGzcuLt5DNvxShd+8XepusxEwAkbACPQNAoUhFYorYhBJ6Jmv6tkNlu4IS+GRv9TOZiNgBIyAEeh7BAqzppIWXSQBuaCkp35sNgJGwAgYgeIhUEhSKR5MzpERMAJGwAjUgoBJpRaU7McIGAEjYARqQsCkUhNM9mQEjIARMAK1IGBSqQUl+zECRsAIGIGaEGgIqeQX0vPPykkle7lbNwJGwAgYgdZGoCGkwm4tEUa1GxrxI3+tDZtzbwSMgBEwAuUQqJlUuiMDEcu8efPCo48+GtPimBVIBkV4jq5/4oknTCwREf8ZASNgBNoPgZpJRd+OpBCkRIMZP/fdd1/YZ599wvnnnx/P6hLZiFTOOOOMSDRp2DROm42AETACRqB1EaiZVCgiRMDVvtwXP3v27IAkInIQeRxwwAHh+9//fvjCF74Q3vve94b58+dHssEfR7A8+eSTMVzrQuacGwEjYASMQCUEaiYVSOHOO+8Me+21V9hll13C9ttvH0499dRIKrjxkzRz1FFHhcmTJ4eRI0eGb3zjG9GNI1a23XbbsMYaa5SISJkirJURMAJGwAi0PgI1k8qCBQvCBz/4wTBixIjw7W9/O3DlL9LKj370oxKxAMdDDz0UpkyZEq8DPu6448Ipp5xSIhvIg3g44h4lMpGUEy39ZwSMgBEwAi2LQM0HSk6fPj0eS3/xxReHzTffPAwaNChsvPHGUSI57bTTSgB8+tOfDly0hUQyZsyY8NRTT4Urrrgi7LbbbvG4ek4iXn311UtEUwpogxEwAkbACLQ8AjWTCnedrLrqqlEKQRq58soro1SCxKKpL/TzzjsvSiNrrbVWvFhr7733DuwIQxphOmy99daLay0vvPBCWGeddcKFF14Y1l9/fZNMyzclF8AIGAEjEELNpLLddtuF/fffP3z2s5+Nl2ntvvvu4Te/+U3YYostSjhCHBtssEG8UGvhwoXh+OOPj2EOOuigSBqQDrc4rrvuuuGEE06IayxMp1kZASNgBIxAeyBQM6msueaa4dJLL407v374wx+Gp59+Omy55ZYlKQU4tEYya9ascOKJJwamzFg/uffee+P0lyA76aSTwh577BGJRov7crNuBIyAETACrYtAzQv1EAbrJEgZ99xzT9hqq60iiWCvDxwXL14cLrnkkvD2t789ut90003hsMMOCx/60IfCOeecE79bwS9bkXXploiodSF0zo2AETACRkAI1CypvPLKKwHSmDhxYvwyHlLgexWms1gTef/73x9+9rOfhS9+8Yvha1/7WvjABz4QSYfpMtzYJQah6BZHyAQpxZKKqsK6ETACRqD1EaiZVN75zndGUhk4cGA4+OCDA/ozzzwTEXjuuefibjDWSZBMxo4dG+0hHsiDXWJnn312vHOejydZdxGZiFyktz6kLoERMAJGoHMRqJlUrrrqqogSu7pQkIB0SRzY4Q6ZSIk8eEZSGTx4cBg9enSJVOQuXeGsGwEjYASMQOshUDOpQAapSklAZnSZU7+YsR8+fHi4+uqrw4ABA/LODX3OSz3554Ym5siMgBEwAkaghEDNpFKNLEqxdWNAghkyZMgKkkw3QXrkrLxKmiISE0uPoHQgI2AEjEBdCNRMKnXFWsVzOjVWxVuvnSARDrPUh5c8Dxs2LAwdOrTXcTsCI2AEjIARKI9An5IKEoSkB0kT5bPVGNupU6eGG2+8MU69sZ5z4IEHrvC9TGNScSxGwAgYASMgBPqUVEi0L8iEdCAvSUVKUzruVkbACBgBI9B4BN7YptX4uPs1RghEUlG/ZsSJGwEjYAQ6CIG2JRURivQOqlMX1QgYASPQbwi0LakgqXi6q9/alRM2AkagQxFoW1JJ6zOVViqZU/82GwEjYASMQM8QaHtSkbSCzhllVkbACBgBI9A8BPp891fzilI+Zkkm6DLLpwhHz9aNgBEwAkagdwi0vaQCPCaP3jUShzYCRsAI1IpA25OKCCUvpdQKkP0ZASNgBIxA7Qi0NankiUQEUzs89mkEjIARMAL1INC2pJInFEApZ1cPWPZrBIyAETAC1RFoW1LJSyX55+qw2NUIGAEjYAR6gkDbkgpSyYwZM3qCicMYASNgBIxADxFoS1LJT3MhpWBnaaWHrcTBjIARMAI1ItCWpJIvuwglTzZ5f342AkbACBiB3iHQlqQiiYQ7VKRMKELCuhEwAkageQi0JakILpGLnq0bASNgBIxAcxFoa1JpLnSO3QgYASNgBPIImFTyiPjZCBgBI2AEeoyASaXH0DmgETACRsAI5BEwqeQR8bMRMAJGwAj0GAGTSo+hc0AjYASMgBHII2BSySPiZyNgBIyAEegxAiaVHkPngEbACBgBI5BHwKSSR8TPRsAIGAEj0GMETCo9hs4BjYARMAJGII9A25IKX9MPHDiwVF5/XV+CwgYjYASMQNMQaFtSAbENN9wwAgeh+OyvprUhR2wEjIARKCHQtqTCYZIQiX6lEttgBIyAETACTUOgbUnlzW/+R9E87dW0tuOIjYARMAIrIdC2pKKSetpLSFg3AkbACDQfgbYmFQhFkor05kPqFIyAETACnYtA25KKCEV651axS24EjIAR6DsE2pZUUsnExNJ3DcopGQEj0NkItC2pQCQokYueO7u6XXojYASMQHMRaFtSEZmk8M2ePbtEMqm9zUbACBgBI9AYBNqWVMrBw7crVkbACBgBI9A8BNqeVNJpr3LSS/OgdcxGwAgYgc5DoO1JpfOq1CU2AkbACPQfAiaV/sPeKRsBI2AE2g6BtiUVTXsx5cVPz21Xgy6QETACRqBACLQtqaQYQyheT0kRsdkIGAEj0BwE2pZURCKphJKamwOnYzUCRsAIdDYCbUsqVOsWW2wRVllllVjDJpTObuguvREwAn2DQFuTytKlS0trKVpb6RtYnYoRMAJGoDMRaFtS0fRXXvcHkJ3Z0F1qI2AE+gaBtiUV4MtPec2ZM2elBfu8n76B3akYASNgBNoTgbYmFaQUSENTX8uWLYu1mBKJJJn2rF6XyggYASPQtwi0NamIPPK6iaRvG5lTMwJGoHMQaGtSkYSi6hS56Nm6ETACRsAINBaBtiQVyEM/iAUz+pvf3JbFbWyLcGxGwAgYgV4g0Ja9rCSUgQMHRiLhmV1fqaQi0ukFdg5qBIyAETACOQTaklRUxqFDh4aUWFJSEfHIr3UjYASMgBHoPQJtTSpIJ6+99lppKiyFy5JKiobNRsAIGIHGILBqY6IpZiypNIJ5/vz5pSkwnq2MgBEwAkagsQi0taQiqPIEwnM6FSZ/1o2AETACRqB3CLQ9qUAeTIOlRIJdnmh6B6NDGwEjYASMAAi0PalAHvpRYEsooGBlBIyAEWgOAm1LKpJGhg0bFpHjWXaSUkwwzWlUjtUIGIHORaBtSUVVmhIHO8HS71VELvJr3QgYASNgBHqHQNuSSkoYklJefvnl0vqKYEtJR3bWjYARMAJGoGcItC2pAIeIBZ0fx7SIRKTLT8/gcygjYASMgBFIEWjb71REGkx3pUrnf5lMUlRsNgJGwAg0BoG2lVQknQwZMiRKJ5AMv5RMRDyNgdKxGAEjYASMQNuSiqp2wIABJTKBRLi3XiolGNlZNwJGwAgYgZ4j0NakkkoiTINx8yNHtaBwS917DqFDGgEjYASMgBBo2zUVCqgpMEkkkEh+TQU7uQsU60bACBgBI9AzBDpCUpFEAnlg1jOQmVB61nAcyggYASNQDoG2JhUKPHLkyNK3KZCJJJVyYNjOCBgBI2AEeodA25OKpBLpM2bMKCEmu5KFDUbACBgBI9ArBNqaVJja0vSW9JRIZNcrBB3YCBgBI2AESgi0NalQyldffXWFKS9IBTJBTwmmhIgNRsAIGAEj0GME2ppUII0RI0bEK4VBKCUUiMWSSo/bjQMaASNgBMoi0PakkpYaEnnssceilaWUFBmbjYARMAKNQaCtSQWIJJGUI5Fydo2B1bEYASNgBDoTgbb/+HGVVVaJayqQCySy6qqretqrM9u6S20EjEAfINDWkgokss4664TVV1+9tJ6yYMGCeP4Xbvr1Ac5OwggYASPQEQi0NalIOqEmZV68eHFYvnx5aVoMeysjYASMgBFoDAJtTSpAhDRSTo+WibuerRsBI2AEjEDPEWhrUkEK4ej74cOHr4BQ/uKuFRz9YASMgBEwAj1GoK1JBVQgFhbnpTj+fuHChfE8MLnLzboRMAJGwAj0DoGOIBUkE6bB0vWT1Nw7CB3aCBgBI2AEhEBbk4p2d22wwQalhXrIxNNfqn7rRsAIGIHGItDWpCKoWFeRglR0+yN2WsiXu3UjYASMgBHoOQJtTSqa4kKXWVIKz/kpsZ7D6JBGwAgYASMAAm1NKhQQ8hg1alSJQPjCfsmSJThZGQEjYASMQIMRaGtS0ZoKu7/SGx/T6a8G4+nojIARMAIdjUBbk4pqNj3vC8kFacVrKULHuhEwAkagcQi0NaloHUXnfwm2F154QUbrRsAIGAEj0EAE2ppUtBCvaTA9a7G+gTg6KiNgBIyAEcgQaGtS0Q4vprsGDRoUF+2x80K9274RMAJGoDkItDWpCDJNg+l50aJF8fh7PVs3AkbACBiBxiDQEaTCzq+BAwfGL+m1QC+9MTA6FiNgBIyAEQCBticVpBRIZfDgwbHGIZNXXnklPP/8894B5nfACBgBI9BgBNqaVFJpZI011ogkAsFoOkx6gzF1dEbACBiBjkWgrUlFpAG5jBgxIlayiGbWrFkdW+kuuBEwAkagWQi0NakAGiQCubADTM/Y6Rct/WcEjIARMAINQaCtSUWEAlJjxowpTXvx/NRTT5lYAMLKCBgBI9BABNqaVFKcuPERiUU/1lbS9ZXUr81GwAgYASPQMwQ6hlTWW2+9sOaaa0aUIJO5c+f6I8ietRmHMgJGwAhURKCtSUUL9al0IiS0poJuZQSMgBEwAo1BoK1JJYWIhfrhw4eXrJgOW7p06QrrLCVHG4yAETACRqBHCHQMqSCt8FU9CvNrr70Wli9fHhfre4ScAxkBI2AEjMBKCHQMqVDyIUOGlHZ8cVIxZ4BpimwlZGxhBIyAETACdSPQUaQydOjQkmTCWgqL9VZGwAgYASPQOAQ6hlQgEXZ/aYGeHWBIKlZGwAgYASPQOARWbVxUxY6Jaa611lqr9J0K018c1ZLu/vJUWLHr0LkzAkag+Ah0jKRCVSCpcF+9iGTBggWBXWB6Ln51OYdGwAgYgWIj0DGkAnGwrZjTiqVefvnl0mVdllKEinUjYASMQM8R6BhSAaIBAwYEvqyXQkqZP3++JRUBYt0IGAEj0EsEOopUkFa22GKLEokguTz33HMrnGDcSzwd3AgYASPQ0Qh0DKloekvnf/EMySxZsiQ2AK+rdPR74MIbASPQIAQ6ZveX8Fp//fVLkgnbimfPnh3JRaQDucisMNaNgBEwAkagNgQ6RlIBDsiCDyDT41oWLlwYSQUykbQivTYI7csIGAEjYASEQEeRCmSBdKKPICEZfQCJOf0JIOtGwAgYASNQOwIdRSqQBsTCDjB0PoBkB9i8efMspdTeZuzTCBgBI1ARgY4iFaHAEfgQCurVV18NL7zwgpysGwEjYASMQC8Q6EhSYV1FpII+ffr0kqSCNGNlBIyAETACPUOg40iFaa+111477gDDDInMnDkz3q/Cs5URMAJGwAj0HIGOIxVIhOkvpBXM/PiqnnPArIyAETACRqB3CHQkqay22mph5MiRJVJBQlm8eHHvkHRoI2AEjIARCB1HKpri2mSTTUrVz3EtTIEhtVgZASNgBIxAzxHoOFLRlNcGG2xQIhHsIBURTs/hdEgjYASMQGcj0HGkAnGw42vw4MFh0KBBJSLhYMmlS5eWnju7Wbj0RsAIGIGeIdBxpCJJZfXVVw/rrLNORE1rKs8++6xJpWftyKGMgBEwAhGBjiMV1TvHtWy55ZZxK7HsHnvsMRmtGwEjYASMQA8Q6LhTipFKtCA/bty4eL2wcNP3KhCOlREwAkbACNSPQEf3nsOGDQv8pPhe5aWXXtKjdSNgBIyAEagTgY4lFa2tbLbZZiXIli9fHp544onSsw1GwAgYASNQHwIdRyqa+gImzKNGjSqtq/A8ZcqUeMhkCqO3Gqdo2GwEjIARqIxAx5FKHgq+V1l11VUjwbDV+Pnnn48/HTiJ/5SI8uH9bASMgBEwAm8g0PGkwvcqO+64Y2krMYv0U6dOfQMhm4yAETACRqBmBDqeVEBq5513jqcWQyhIJdOmTYsfSDLt5amvmtuSPRoBI2AEOu/sr3ydQyKcWLzhhhtGAoFE2AV29913R2Lx1FceMT8bASNgBCoj0PGSiqSRCRMmlEgFieW2224LS5YsKV3mVRlCuxgBI2AEjIAQ6HhSQRLhN378+DBkyJAAobBID6HwMSTPVkbACBgBI1AbAu4xM5wgFc4C22233aK0orWVyZMnRxS9rlJbY7IvI2AEjIBJJWkDnAWWnlzM9uI77rgj8WGjETACRsAIVEPApJKgw/biHXbYobS2ghMfQy5btizaJV5tNAJGwAgYgTIIdDypaGpLayt77LFH6f568HrxxRfDrbfeGqfItKivMGXwtJURMAJGoKMR6HhSgUxSkuDr+re+9a2lXV+4sRNs1qxZHd1QXHgjYASMQC0IdDypABLEIgWJsL14+PDhJXsW7m+//fboJe9X4awbASNgBIxA8MePNIJUUoE0kFaOPPLIsMYaa5SmvfjK/q677op+Jd2kBOPGZASMgBEwAiaVFdpASi7cs8Kx+OnBkjfddFOYO3dutDOhrACdH4yAETACEQFPf2UwiCDy+sEHHxw/iAQpCOe1114LEydOjB9EioCkRzT9ZwSMgBHocARMKhUaAOsoTINBLJANP6SWp556KkyaNCmSDIQiIqoQja2NgBEwAh2FgEmlm+oel91jz0eREIh+rK0899xzJpRusLOzETACnYeASaVKnUtCOeyww+INkTwjwbz66qvhkksuCbNnz45Eoyg8FSYkrBsBI9CpCJhUuql5iIRpsIMOOiieDybvTIVdffXV4eWXX16BWORu3QgYASPQiQiYVGqodYhl5MiRcZsxZMIzi/bPPPNMlFiWL1++0o6wdNdYDUnYixEwAkagLRAwqdRRjWPHjg0c4yJigVzmzZsXrr322tJ6i6JjmszKCBgBI9BpCLjnq7HGWS+BKPbee+/wlre8JYaCVLD/61//Gq677rqViAVPXmepEWB7MwJGoC0QMKnUWI0QCD/U/vvvHzbYYINSSMgGYvn9738fpRgRCbrClDzbYASMgBFoYwRMKnVWLiSx2mqrheOPPz6MzabDRCBMiT300EPhhhtuKBGLCaVOcO3dCBiBlkfApNKDKoQsVllllXD00UeHjTfeOEoj2PHjUi8kFhbvKykRUSV32xsBI2AEWhUBk0ovag5iefe73x2JBaLgx1TY1KlT466wRYsWldZZ5I5uCaYXoDuoETAChUbApNLL6hkwYEA45phjwiabbFKSWIiSL+5/9atfxd1hSkJkArFYGQEjYATaEQGTSi9qVVIH0slRRx0VRo8eHddTiBIC4Y77Cy64IMyYMSOmIv/xwX9GwAgYgTZEwKTSi0qV5EEUfHX/3ve+N+y4446lxXvsWVtBYuHYfEjFxAIqVkbACLQrAiaVBtQs5MKPXWGHHHJIvI6YaLGDRNgZNnny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)

식(1)과 같지 않은 다른 형태의 예측 선을 고려할 수 있지만 우리는 지금 식(1)과 같은 형태의 곡선을 예측 선으로 사용해보려고 합니다. 식(1)의 곡선을 결정하는 요소는 $a$와 $b$입니다. 

따라서 $a$값 변화에 따른 오차 생각해 보면 $a$값이 작아지면 오차는 무한대로 커지지만 $a$ 값이 커진다고 해서 오차가 무한대로 커지지는 않습니다. 별첨한 그래프 참고.

$b$값이 너무 크거나 작은 경우 오차는 별첨한 그래프와 같이 2차 함수 그래프와 유사한 형태로 나타납니다.

## 3.오차 공식

 이제 우리에게 주어진 과제는 또다시 $a$값과 $b$값을 구하는 것임을 알았습니다. 시그모이드 함수에서 $a$값과 $b$값을 어떻게 구해야할까요? 역시 경사 하강법입니다.

 경사 하강법은 먼저 오차를 구한 후 오차가 작은 쪽으로 이동시키는 방법이라고 했습니다. 그렇다면 이번에도 예측 값과 실제 값과의 차이, 즉 오차를 구하는 공식이 필요합니다. 그렇다면 이번에도 앞서 배웠던 평균 제곱 오차(MSE)를 사용하면 될까요? 안타깝게도 이번에는 평균 제곱 오차(MSE)를 사용할 수 없습니다. 오차 공식을 도출하기 위해 시그모이드 함수의 특징을 다시 살펴 보겠습니다.  

 시그모이드 함수의 특징은 $y$ 값이 0과 1사이라는 것입니다. 따라서 실제 값이 1 (합격)일때 예측 값이 0(불합격)에 가까워지면 오차는 커집니다. 반대로 실제 값이 0일 때 예측 값이 1에 까가워지는 경우에도 오차가 커집니다. 이를 공식으로 만들 수 있게 하는 함수가 로그 함수입니다. 


## 4. 로그 함수
 오차 식은 다음의 특성이 있기를 기대합니다. 실제 값이 1일 때 예측 값도 1에 가까워지면서 오차도 0 값에 가까워지는 특성이 있어야겠습니다. 반대로 실제 값이 1인데 예측 값이 0에 가까워지면 오차는 커져야합니다. 또 실제 값이 0일 때 예측 값이 0에 가까워지면서 오차도 0에 가까워지는 특성이 있어야하고 예측 값이 1에 가까워진다면 오차는 커져야 합니다. 
 아래 그래프에서 파란색 선은 예측 값이 1에 가까워지면 오차는 0 값에 가까워지고 예측 값이 0에 가까워지면 오차가 커지는 것을 볼 수 있습니다. 실제 값이 1일 때 오차 식은 파란색 선의 특성을 보여야 합니다. 빨간색 선은 예측 값이 0일 때 오차가 0에 가까고 예측 값이 1일 때 오차는 커집니다. 실제 값이 0일 때 오차 식은 빨간색 선의 특성을 보여야 합니다. 


```python
import numpy as np
import matplotlib.pyplot as plt
t = 0.001
y_hat = np.linspace(t, 1-t, 100)
e1 = - np.log(y_hat)                    #실제 값이 1인 경우 적용할 에러 식 ( 파란색 )
e0 = - np.log(1-y_hat)                  #실제 값이 0인 경우 적용할 에러 식 ( 빨간색 )
plt.plot(y_hat, e1, 'b')
plt.plot(y_hat, e0, 'r')
plt.xlabel('$\hat{y}$')
plt.ylabel('e')
plt.grid()
plt.show()
```


    
![png](output_164_0.png)
 
 실제 값이 1인 경우 적용할 에러 식은  식 (6-1)과 같다.  
(식. 6-1)
$$e = -\log(\hat{y})$$
실제 값이 0인 경우 에러 식은  식(6-2)와 같다.  
(식. 6-2)
$$e = -\log(1-\hat{y})$$

여기서 $y$를 실제 값이라고 한다면 식 (6-1)과 식 (6-2)를 통합하여 식 (6-3)과 같이 쓸 수 있다.  
(식.6-3)
$$e = -y\log(\hat{y}) -(1-y)\log(1-\hat{y})=-\{y\log(\hat{y}) +(1-y)\log(1-\hat{y}) \}$$

실제 값 $y$가 1이라면 식 (6-3)은 식 (6-1)이 되고 실제 값 $y$이 0이면 식 (6-3)은 식 (6-2)가 된다.

이렇게 해서 평균 제곱 오차를 대체할 만한 손실 함수를 구했습니다. 이 함수를 러닝 머신에서는 **교차 엔트로피 오차(cross entropy error)**함수라고 합니다. 

선형 회귀에서는 평균 제곱 오차를, 로지스틱 회귀에서는 교차 엔트로피 오차 함수를 사용하게 되는 것입니다. 이 두 개의 함수에서 출발해서 지금은 더 다양한 손실 함수들이 존재합니다. 이와 관련해서는 '10장. 딥러닝 모델 설계하기'에서 다시 다룰  것입니다. 

이제 로지시틱 회귀를 사용해서 어떻게 모델을 만들 수 있는지 텐서플로에서 실행해 보겠습니다.

## 5. 텐서플로에서 실행하는 로지스틱 회귀 모델

텐서플로에서 실행하는 방법은 앞서 선형 회귀 모델을 만들 때와 유사합니다. 다른 점은 오차를 계산하기 위한 손실 함수가 평균 제곡 오차 함수에서 크로스 엔트로피 오차 함수로 바뀐다는 것입니다. 

먼저 공부한 시간 대비 합격/불합격 결과를 위한 데이터를 준비하겠습니다.

|공부한 시간|2|4|6|8|10|12|14
|---|:---:|:---:|:---:|:--:|:---:|:---:|:---:|
|합격 여부|불합격|불합격|불합격|합격|합격|합격|합격| 


```python
# 1. 환경 준비
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 데이터 준비
x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1]) 
```

이제 모델을 준비합니다. 먼저 시그모이드 함수를 사용하게 됨으로 model.add()함수에 activation 파라메타에 'sigmod'를 할당합니다.


``` python
model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))
```

손실 함수로 교차 엔트로피 오차 함수를 이용하기 위해 loss 파라미터에 'binary_crossentropy'를 할당합니다.

```python

# 교차 엔트로피 오차 함수를 이용하기 위하여 'binary_crossentropy'로 설정합니다. 
model.compile(optimizer='sgd' ,loss='binary_crossentropy')
```

생성된 모델을 적용합니다.

```python
model.fit(x, y,epochs = 5000)


