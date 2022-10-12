
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


<img width="375" alt="스크린샷 2022-10-12 오전 10 14 09" src="https://user-images.githubusercontent.com/87309905/195226470-53263046-166a-4d48-8242-9af32365726a.png">
    


**위 두 그래프 해석**  
그래프는 $b$ 값이 커지면 왼쪽으로 이동하고 $b$값이 작아지면 오른쪽으로 이동한다는 것을 확인했습니다. 

<img width="322" alt="스크린샷 2022-10-12 오전 10 13 34" src="https://user-images.githubusercontent.com/87309905/195226402-7c8b530c-4422-4300-9112-6fab19b8bf66.png">

<img width="329" alt="스크린샷 2022-10-12 오전 10 13 50" src="https://user-images.githubusercontent.com/87309905/195226442-1b4f6e4f-636f-4a2e-8846-cc92a17b7cf9.png">

<img width="404" alt="스크린샷 2022-10-12 오전 10 14 00" src="https://user-images.githubusercontent.com/87309905/195226461-d0b1053c-1ff0-445b-ae7d-6e1fa2cd4655.png">



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


<img width="376" alt="스크린샷 2022-10-12 오전 10 14 22" src="https://user-images.githubusercontent.com/87309905/195226484-04860b45-b0cc-49f0-879c-898ff1984339.png">
 
 
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


model.fit()이 종료된 후 model.predict() 함수를 이용해서 학습 시간 $x$가 입력되었을 때 결과를 그래프로 그려 봅니다. model.fit() 함수가 완료되기까지 약 5분의 시간이 걸립니다. (2022년 10월, 4분 25초 거렸음, 하드웨어 가속기 GPU 사용 했을 때 3분대...ㅠ.ㅠ )

```python
plt.scatter(x,y)
plt.plot(x, model.predict(x),'r')
plt.show()
```
<img width="374" alt="스크린샷 2022-10-12 오전 10 16 51" src="https://user-images.githubusercontent.com/87309905/195226761-03664436-917f-43fc-b115-fad0a10ab719.png">



