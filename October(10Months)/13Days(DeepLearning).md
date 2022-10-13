
# 14장 모델의 성능 향상시키기
포도로 만든 와인은 고대 그리스 로마 시대부터 서양 음식의 기본이 된 오랜 양조주입니다. 와인은 빛깔에 따라 맑고 투명한 화이트 와인과 붉은색을 띠는 레드 와인으로 구분됩니다. 이번 실습을 위해 사용되는 데이터는 포르투갈 서북쪽 대서양과 맞닿아 있는 비뉴 베르드(Vinho Verde) 지방에서 만들어진 와인을 평가한 데이터입니다. 

레드 와인 샘플 1,599개를 등급과 맛, 산도를 측정해 분석하고 화이트 와인 샘플 4,898개를 마찬가지로 분석해 데이터를 만들었습니다. 원래는 UCI 저장소에 올라온 각각 분리된 데이터인데 두 데이터를 하나로 합쳐 레드 와인과 화이트 와인을 구분하는 실험을 진행해 보겠습니다.



### 1. 데이터의 확인과 검증셋
먼저 데이터를 불러와 대략적인 구조를 살펴보겠습니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

# 깃허브에 준비된 데이터 가지오기
!git clone https://github.com/taehojo/data.git

# 와인 데이터를 불러오기
df = pd.read_csv('./data/wine.csv', header = None)

# 데이터를 미리 보겠습니다.
df
```

    fatal: destination path 'data' already exists and is not an empty directory.






  <div id="df-5797883b-2a38-4b3f-aec0-e4038159df7e">
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
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6492</th>
      <td>6.2</td>
      <td>0.21</td>
      <td>0.29</td>
      <td>1.6</td>
      <td>0.039</td>
      <td>24.0</td>
      <td>92.0</td>
      <td>0.99114</td>
      <td>3.27</td>
      <td>0.50</td>
      <td>11.2</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6493</th>
      <td>6.6</td>
      <td>0.32</td>
      <td>0.36</td>
      <td>8.0</td>
      <td>0.047</td>
      <td>57.0</td>
      <td>168.0</td>
      <td>0.99490</td>
      <td>3.15</td>
      <td>0.46</td>
      <td>9.6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6494</th>
      <td>6.5</td>
      <td>0.24</td>
      <td>0.19</td>
      <td>1.2</td>
      <td>0.041</td>
      <td>30.0</td>
      <td>111.0</td>
      <td>0.99254</td>
      <td>2.99</td>
      <td>0.46</td>
      <td>9.4</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6495</th>
      <td>5.5</td>
      <td>0.29</td>
      <td>0.30</td>
      <td>1.1</td>
      <td>0.022</td>
      <td>20.0</td>
      <td>110.0</td>
      <td>0.98869</td>
      <td>3.34</td>
      <td>0.38</td>
      <td>12.8</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6496</th>
      <td>6.0</td>
      <td>0.21</td>
      <td>0.38</td>
      <td>0.8</td>
      <td>0.020</td>
      <td>22.0</td>
      <td>98.0</td>
      <td>0.98941</td>
      <td>3.26</td>
      <td>0.32</td>
      <td>11.8</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 13 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5797883b-2a38-4b3f-aec0-e4038159df7e')"
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
          document.querySelector('#df-5797883b-2a38-4b3f-aec0-e4038159df7e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5797883b-2a38-4b3f-aec0-e4038159df7e');
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




샘플이 전체 6,497개 있습니다. 속성이 12개 있으며 13번째 열에 클래스가 기록되어 있습니다. 각 속성에 대한 정보는 다음과 같습니다.

<br><center>

|행 번호|의미|행 번호|의미|
|---|---|---|---|
|0|주석산 농동|7|밀도|
|1|아세트산 농도|8|pH|
|2|구연산 농도|9|황산칼륨 농도|
|3|잔류 당분 농도|10|알코올 도수|
|4|염화나트륨 농도|11|와인의 맛(0~10등급)|
|5|유리 아황산 농도|12|클래스(1:레드 와인, 0:화이트 와인)|
|6|총 아황산 농도|
</center><br>  


0~11번째 열에 해당하는 속성 12개를 ```X```로, 13번째 열을 ```y```로 정했습니다.


```python
# 완인의 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]
```

이제 딥러닝을 실행할 차례입니다. 앞서 우리는 학습(train) 데이터셋과 테스트(test) 데이터셋을 나누는 방법에 대해 알아봤습니다. 이 장에서는 검증셋(validation)을 고려해 보겠습니다.  
<br><center>
(그림. 14-1) 학습셋, 데스트셋, 검증셋<br>
<img src="https://drive.google.com/uc?id=1RUHPeVzw-a7m2_qtUsIHjvswWDrhr6Ce" width=300>
</center><br>  

학습이 끝난 모델을 테스트해 보기 위해 테스트 데이터셋의 마련하는 것이고 최적의 학습 파라미터를 찾기 위해 학습 과정에서 사용하는 데이터셋이 검증 데이터셋입니다. 검증 데이터셋을 사용하겠다고 설정하면 검증 데이터셋에 대해 테스트한 결과를 추적하면서 최적의 모델을 만들 수 있습니다. 검증셋은 ```model.fit()```함수로 전달하는 파라미터 **```validation_split```**의 값을 저정해 주면 만들어집니다. 그림(14-1)과 같이 전체 데이터셋의 80%를 학습 데이터셋으로 만들고 이 중 25%를 검증 데이터셋으로 하면 학습셋:검증셋:테스트셋의 비율이 60:20:20이 됩니다.




```python
# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
# 모델이 3개의 은닉층을 갖도록 구조를 설계(설정)합니다.
model.add(Dense(30,  input_dim=12, activation='relu'))    # 첫 번째 은닉층은 30개 노드
model.add(Dense(12, activation='relu'))                   # 두 번째 은닉층은 12개 노드
model.add(Dense(8, activation='relu'))                    # 세 번째 은닉층은 8개 노드로 구성
model.add(Dense(1, activation='sigmoid'))                 # 출력층
model.summary()

# 모델을 컴파일합니다.
# 이진 분류를 위한 손실 함수(lost function) : binary_crossentropy
# 옵티마이저 : adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 30)                390       
                                                                     
     dense_1 (Dense)             (None, 12)                372       
                                                                     
     dense_2 (Dense)             (None, 8)                 104       
                                                                     
     dense_3 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 875
    Trainable params: 875
    Non-trainable params: 0
    _________________________________________________________________


먼저 세 개의 은닉층을 만들고 각각 30개, 12개, 8개의 노드를 만들었습니다. 아래 코드에서는 model을 생성합니다. 즉 모델을 학습 시킵니다. ```model.fit()``` 함수의 인자 ```validation_split``` 값을 할당했으므로 모델이 학습 데이터셋을 학습을 위한 데이터셋과 검증을 위한 데이터셋을 분리하여 모델 파라메타를 찾습니다. 


```python
# 모델을 실행 (학습 데이터를 학습, 검증 데이터로 구분)

history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25) # 0.8 x 0.25 = 0.8 x 1/4 = 0.2
```

    Epoch 1/50
    8/8 [==============================] - 1s 62ms/step - loss: 3.8098 - accuracy: 0.2558 - val_loss: 0.6234 - val_accuracy: 0.6431
    Epoch 2/50
    8/8 [==============================] - 0s 23ms/step - loss: 0.4822 - accuracy: 0.7629 - val_loss: 0.5408 - val_accuracy: 0.7885
    Epoch 3/50
    8/8 [==============================] - 0s 17ms/step - loss: 0.6265 - accuracy: 0.7824 - val_loss: 0.6039 - val_accuracy: 0.8015
    ...
    Epoch 49/50
    8/8 [==============================] - 0s 17ms/step - loss: 0.1788 - accuracy: 0.9341 - val_loss: 0.1659 - val_accuracy: 0.9431
    Epoch 50/50
    8/8 [==============================] - 0s 13ms/step - loss: 0.1780 - accuracy: 0.9335 - val_loss: 0.1640 - val_accuracy: 0.9431


그리고 50번을 반복했을 때 학습 데이터셋에 대한 정확도(accuracy)는 94.97%로 나왔고 검증 데이터셋에 대한 정확도는 94.85% 나왔습니다. 꽤 높은 정확도군요. 아래에는 테스트 데이터 셋에 대한 정확도가 93.76%가 나왔습니다.


```python
# 테스트 결과를 출력합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```

    41/41 [==============================] - 0s 4ms/step - loss: 0.1685 - accuracy: 0.9354
    Test accuracy: 0.9353846311569214


하지만 테스트 데이터셋에 대한 정확도는 약 93%가 나왔습니다. 이 것이 과연 최적의 결과일까요? 이제 여기에 여러 옵션을 더해 가면서 더 나은 모델을 만들어 가는 방법을 알아보겠습니다.


```python
history = model.fit(X_train, y_train, epochs = 50, batch_size = 500)
# 터스트 결과를 출력
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])
```

    Epoch 1/50
    11/11 [==============================] - 0s 4ms/step - loss: 0.1048 - accuracy: 0.9627
    Epoch 2/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.1034 - accuracy: 0.9615
    Epoch 3/50
    11/11 [==============================] - 0s 4ms/step - loss: 0.1010 - accuracy: 0.9625
    ...
    Epoch 48/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0640 - accuracy: 0.9815
    Epoch 49/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0632 - accuracy: 0.9813
    Epoch 50/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0630 - accuracy: 0.9817
    41/41 [==============================] - 0s 2ms/step - loss: 0.0596 - accuracy: 0.9815
    Test accuracy:  0.9815384745597839


## 2. 모델 업데이트하기

에포크(epochs)는 학습을 몇 번 반복할 것인지 정해 줍니다. 에포크가 50이면 순전파와 역전파를 50번 실시한다는 뜻입니다. 학습을 많이 한다고 해서 모델 성능이 지속적으로 좋아지는 것은 아닙니다. 이를 적절히 정해 주는 것이 중요합니다. 만일 50번의 에포크 중 최적의 학습이 40번재에 이루어졌다면 어떻게 40번째 모델을 불러와 사용할 수 있을까요? 이번에는 매 에포크마다 모델의 정확도를 함께 기록하면서 저장하는 방법을 알아보겠습니다. 

먼저 모델이 어떤 식으로 저장될지 정합니다. 다음 코드는 ./data/model/all/ 폴더에 모델을 지정해 줍니다. 50번째 에포크의 검증 데이터셋에 대한 정확도가 0.9346이라면 50-0.9346.hdf5라는 이름으로 저장됩니다.

```
modelpath = './data/model/all/{epcho:02d}-{val_acuracy:.4f}.hdf5'
```

학습 중인 모델을 저장하는 함수는 케라스 API의 ```ModelCheckpoint()```입니다. 모델이 저장될 곳을 정하고 진행되는 상황을 모니터링할 수 있도록 ```verbos```는 1(```True```)로 설정합니다. 

``` 
from tensorflow.keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1)
```



### 기본 코드 불러오기

다시 필요 데이터셋을 불러오고 속성과 클래스를 분리한 다음 학습 데이터셋과 테스트 데이터셋으로 분리합니다. 모델 구조를 정의하고 컴파일합니다 


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 깃허브에 준비된 데이터를 가져옵니다. 앞에서 이미 데이터를 가져왔으므로 주석 처리합니다. 2번 예제만 별도 실행 시 주석을 해제한 후 실행해주세요.
!git clone https://github.com/taehojo/data.git

# 완인 데이터를 불러옵니다.
df = pd.read_csv('./data/wine.csv', header = None)

# 와인의 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

    fatal: destination path 'data' already exists and is not an empty directory.
    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_4 (Dense)             (None, 30)                390       
                                                                     
     dense_5 (Dense)             (None, 12)                372       
                                                                     
     dense_6 (Dense)             (None, 8)                 104       
                                                                     
     dense_7 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 875
    Trainable params: 875
    Non-trainable params: 0
    _________________________________________________________________


### 모델의 저장 설정 및 실행
매 에포크(epochs) 마다 모델의 정확도를 출력합니다. 


```python
# 모델 저장의 조건을 설정합니다.
modelpath = "./data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath = modelpath, verbose = 1)

# 모델을 실행합니다.
history = model.fit(X_train, y_train, epochs = 50, batch_size = 500, validation_split = 0.25, verbose = 0, callbacks = [checkpointer])

print('---'*24)
# 테스트 결과 출력
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])
```

    
    Epoch 1: saving model to ./data/model/all/01-0.9485.hdf5
    
    Epoch 2: saving model to ./data/model/all/02-0.9492.hdf5
    
    Epoch 3: saving model to ./data/model/all/03-0.9492.hdf5
    ...
    
    Epoch 48: saving model to ./data/model/all/48-0.9585.hdf5
    
    Epoch 49: saving model to ./data/model/all/49-0.9646.hdf5
    
    Epoch 50: saving model to ./data/model/all/50-0.9692.hdf5
    ------------------------------------------------------------------------
    41/41 [==============================] - 0s 2ms/step - loss: 0.0964 - accuracy: 0.9685
    Test accuracy:  0.9684615135192871


위 코드 cell 실행 결과로부터 에포그 수가 늘어나면 학습 데이터셋에 대한 정확도가 어느 정도 높아지다가 멈춘다는 것을 알 수 있었습니다. 에포크 50회까지 실행하는 동안 최고의 정확도가 발생한 에포크는 몇 회인가요?


## 3. 그래프로 과적합 확인하기

역전파를 50번 반복하면서 학습을 진행했습니다. 과연 이 반복 횟수는 적절했을까요? 학습의 반복 횟수가 너무 적으면 데이터셋의 패턴을 충분히 파악하지 못합니다. 하지만 학습을 너무 많이 반복하는 것도 좋지 않습니다. 너무 과한 학습은 13.2절에서 이야기한 바 있는데 과적합 현상을 불러오기 때문입니다. 적절한 학습 회수를 정하기 위해서는 검증 데이터셋과 테스트 데이터셋의 결과(정확도)를 그래프로 보는 것이 가장 좋습니다. 이를 확인하기 위해 학습을 길게 실행해 보고 결과를 알아보겠습니다. 먼저 에포크 수를 2000으로 늘려 학습을 2000회 시켜보겠습니다. (실행 시간이 다소 깁니다.)


```python
# 그래프 확인을 위한 긴 학습 
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25)
```

    Epoch 1/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.1048 - accuracy: 0.9672 - val_loss: 0.0975 - val_accuracy: 0.9669
    Epoch 2/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1020 - accuracy: 0.9692 - val_loss: 0.0971 - val_accuracy: 0.9677
    Epoch 3/2000
    ...
    Epoch 1998/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9897 - val_loss: 0.0680 - val_accuracy: 0.9815
    Epoch 1999/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0286 - accuracy: 0.9913 - val_loss: 0.0545 - val_accuracy: 0.9869
    Epoch 2000/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0215 - accuracy: 0.9938 - val_loss: 0.0524 - val_accuracy: 0.9885


```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25)
```

위 예제를 포함해 ```model.fit()```를 실행할 때 결과를 ```history```에 항상 저장했었습니다. ```history```에 저장된 데이터를 어떻게 활용하는지 보겠습니다. 

```model.fit()```은 학습을 진행하면서 매 에포크마다 결과를 출력합니다. 일반적으로 loss 값이 출려되고, ```model.compile()```에서 ```metrics```를 ```accuracy```로 지정하면 accuracy 값이 함께 출력됩니다.  
<br>

**loss**는 학습을 통해 구한 예측 값과 실제 값의 차이(오차)를 의미합니다. **accuracy**는 고려한 샘플들 중에서 정답을 맞춘 샘플이 몇 개인지의 비율(**정확도**)를 의미합니다. 학습 데이터셋으로 학습하고 해당 학습 데이터를 테스트해서 얻은 정확도입니다. 

이번 예제처럼 검증(validation) 데이터셋을 지정하면 **val_loss**가 함께 출력됩니다. 이때 metrics를 accuracy로 지정해두었다면 **val_accuracy** 값도 출력됩니다. val_loss는 학습한 모델을 검증 데이터셋에 적용해 얻은 오차이고, **val_accuracy**는 검증(validation) 데이터셋에 대한 정확도입니다. 


이러한 값이 저장된 ```history```는 파이썬 객체로, ```history.params```에는 ```model.fit()```의 설정 값들이, ```history.epoch```에는 에포크 정보가 들어 있습니다. 우리에게 필요한 loss, accuracy, val_loss, val_accuracy는 ```history.history```에 저장되어 있습니다. 이를 판다스 라이브러리로 불러와 각각의 값을 살펴보겠습니다. 


```python
# history에 저장된 학습 결과를 확인
hist_df = pd.DataFrame(history.history)
hist_df
```





  <div id="df-c0972a81-3f82-4605-afb6-e9a52aa4b301">
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
      <th>loss</th>
      <th>accuracy</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.104765</td>
      <td>0.967154</td>
      <td>0.097498</td>
      <td>0.966923</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.102035</td>
      <td>0.969207</td>
      <td>0.097085</td>
      <td>0.967692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.100406</td>
      <td>0.968950</td>
      <td>0.096251</td>
      <td>0.968462</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.101627</td>
      <td>0.968437</td>
      <td>0.096819</td>
      <td>0.970769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.100613</td>
      <td>0.968694</td>
      <td>0.094706</td>
      <td>0.968462</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>0.023889</td>
      <td>0.991275</td>
      <td>0.066835</td>
      <td>0.985385</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>0.036196</td>
      <td>0.988196</td>
      <td>0.061795</td>
      <td>0.986154</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>0.032621</td>
      <td>0.989736</td>
      <td>0.068035</td>
      <td>0.981538</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>0.028591</td>
      <td>0.991275</td>
      <td>0.054528</td>
      <td>0.986923</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>0.021481</td>
      <td>0.993841</td>
      <td>0.052385</td>
      <td>0.988462</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c0972a81-3f82-4605-afb6-e9a52aa4b301')"
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
          document.querySelector('#df-c0972a81-3f82-4605-afb6-e9a52aa4b301 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c0972a81-3f82-4605-afb6-e9a52aa4b301');
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




2000번의 학습 결과가 저장되어 있음을 알 수 있습니다. 학습한 모델을 검증(validation) 데이터셋에 적용해 얻은 오차(val_loss)는 ```y_vloss```에 저장하고 학습(train) 데이터셋에서 적용해 얻은 오차(loss)는 ```y_loss```에 저장하겠습니다.


```python
# y_vloss에 검증(validation) 데이터셋에 대한 오차를 지정합니다.
y_vloss = hist_df['val_loss']

# y_loss에 학습(train) 데이터셋의 오차를 저장합니다.
y_loss = hist_df['loss']

# x에 에포크 값을 지정하고
# 검증 데이터셋에 대한 오차를 빨간색으로, 학습셋에 대한 오차를 파란색으로 표시합니다.
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, "o", c = "red", markersize = 2, label = 'Validation loss')
plt.plot(x_len, y_loss, "o", c = "blue", markersize = 2, label ='Trains loss')

plt.legend(loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_24_0.png)
    



<br><center>
(그림. 14-2) 학습셋에서 얻은 오차와 검증셋에서 얻은 오차 비교<br>
<img src="https://drive.google.com/uc?id=1wzW5jYKbO3B4iUL8D7p8_jaJAwHbO9vS" width=500>
</center><br>
그림(14.2)의 범례에서 "Testset_loss"는 검증(validation) 데이터셋에 대한 에러를 의미합니다.  

<br><br>
(강사 - 교재에서는 그림(14-2)와 같은 그래프가 나온다고 하는데 이것은 다소 인위적을로 만든 그래프 같다는 생각이 듬. 위 코드를 실제로 수행해 얻은 위 그래프의 의미를 다시 생각해 보시길 바람. 여기서 이해해야 할 핵심 내용은 에포크가 증가함에 따라 학습 데이터셋에 대한 오차는 줄어들지만 검증 데이터셋에 대한 오차는 오히려 증가 할 수 있다는 것이다. **모델이 학습 데이터셋에 과적합되어감에따라 검증 데이터셋에 대해서는 오히려 에러가 증가한다는 것입니다. 그림(14-2)를 보면 에포크의 증가에 따라 검증 데이터셋에 대한 에러가 감소하가다 다시 서서히 증가하는 것을 볼 수 있다. 따라서 검증 데이터셋에 대한 에러가 증가하기 전에 학습을 멈춰 학습 데이터에 과적합되는 것을 막아야 최고의 성능을 기대할 수 있다.**)



## 4. 학습의 자동 중단

텐서플로에 포함된 케라스 API는 ```EarlyStopping()``` 함수를 제공합니다. 학습이 진행되어도 검증 데이터셋에 대한 오차가 줄어들지 않으면 학습을 자동으로 멈추게 하는 함수이니다. 이를 조금 전에 배운 ```ModelCheckpoint()``` 함수와 함께 사용해 보면서 최적의 모델을 저장해 보겠습니다. 먼저 다음과 같이 ```EarlStopping()``` 함수를 호출합니다.

```
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
```

monitor 옵션은 model.fit() 함수의 실행 결과 중 어떤 것을 이용할지 정합니다. 검증(validation) 데이터셋의 오차(val_loss)로 지정하겠습니다. patience 옵션은 지정된 값이 몇 번 이상 향상되지 않으면 학습을 종료시킬지 정합니다. monitor='val_loss', patience=20이라고 지정하면 검증 데이터셋의 오차가 20번 이상 나자지지 않을 경우 학습을 종료하라는 의미입니다. 

모델 저장에 관한 설명은 앞 절에서 사용한 내용을 그대로 따르겠습니다. 다만 이번에는 최고의 모델 하나만 저장되게끔 해보겠습니다. 이를 위해 저장된 모델 이름에 에포크나 정확도 정보를 포함하지 않고 ModelCheckpoint()의 save_best_only 옵션을 True로 설정합니다. 

```
modelpath = "./data/model/Ch14-4-bestmodel.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, mointor='val_loss', verbose=0, save_best_only=True)
```
모델을 실행합니다.  자동으로 최적의 에포트를 찾아 멈출 예정이므로 epochs는 넉넉하게 설정해 줍니다.
```
history = model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.5, verbos=1, callbacks=[early_stopping_callback, checkpointer])
```

앞서 만든 기본 코드에 다음과 같이 새로운 코드를 붙여서 실행해 보겠습니다.

### 기본 코드 불러오기


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd

# 깃허브에 준비된 데이터를 가져옵니다. 앞에서 이미 데이터를 가져왔으므로 주석 처리합니다. 2번 예제만 별도 실행 시 주석을 해제한 후 실행해주세요.
# !git clone https://github.com/taehojo/data.git

# 와인 데이터를 불러오기
df = pd.read_csv('./data/wine.csv', header = None)

# 와인 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# 학습셋과 테스트셋으로 나눕니다. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

# 모델 구조를 설정

model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# 모델을 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_8 (Dense)             (None, 30)                390       
                                                                     
     dense_9 (Dense)             (None, 12)                372       
                                                                     
     dense_10 (Dense)            (None, 8)                 104       
                                                                     
     dense_11 (Dense)            (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 875
    Trainable params: 875
    Non-trainable params: 0
    _________________________________________________________________


## 학습의 자동 중단 및 최적화 모델 저장


```python
# 학습이 언제 자동 중단될지를 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 20)

# 최적화 모델이 저장될 폴더와 모델의 이름을 정함
modelpath = './data/model/Ch14-4-bestmodel.hdf5'

# 최적화 모델을 업데이트하고 저장
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 0, save_best_only = True)

# 모델을 실행
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1, callbacks=[early_stopping_callback,checkpointer])

```

    Epoch 1/2000
    8/8 [==============================] - 1s 35ms/step - loss: 14.1708 - accuracy: 0.2492 - val_loss: 10.2159 - val_accuracy: 0.2431
    Epoch 2/2000
    8/8 [==============================] - 0s 11ms/step - loss: 6.7169 - accuracy: 0.2492 - val_loss: 2.7806 - val_accuracy: 0.2431
    Epoch 3/2000
    8/8 [==============================] - 0s 9ms/step - loss: 1.1996 - accuracy: 0.5055 - val_loss: 0.3055 - val_accuracy: 0.8808
    ...
    8/8 [==============================] - 0s 14ms/step - loss: 0.0517 - accuracy: 0.9833 - val_loss: 0.0526 - val_accuracy: 0.9869
    Epoch 322/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0504 - accuracy: 0.9833 - val_loss: 0.0483 - val_accuracy: 0.9838
    Epoch 323/2000
    8/8 [==============================] - 0s 26ms/step - loss: 0.0489 - accuracy: 0.9867 - val_loss: 0.0487 - val_accuracy: 0.9831


에포크를 2,000번 설정했지만 323번에서 멈췄습니다. 이때의 모델이 model 폴더에 Ch14-4-bestmodel.hdf5라는 이름으로 저장된 것을 확인하십시요.

이제 지금까지 만든 모델을 테스트(test)해 보겠습니다. 따로 보관하여 학습 과정에서 사용되지 않은 테스트 데이터셋을 이 모델에 적용했을 때의 정확도를 확인하기 위해 아래와 같이 코딩하였습니다. 


```python
# 테스트 결과를 출력
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])
```

    41/41 [==============================] - 0s 5ms/step - loss: 0.0332 - accuracy: 0.9885
    Test accuracy:  0.9884615540504456


#### [과제] 
과제 1 - 피마 인디언 데이터를 k 겹 교차 검증 법을 활용하여 모델을 설계하고 예측 정확도를 평가하는 코드를 작성하세요.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

# 피마 인디언 데이터 불러옵니다.
df = pd.read_csv('./data/pima-indians-diabetes3.csv')
df 

X = df.iloc[:,0:8]
y = df.iloc[:, 8]

```


```python
# k 할당
k = 5

# X 독립변수, y 종속변수 설정

kfold = KFold(n_splits = k, shuffle = True)

acc_score = []

# 모델 구조 생성
def model_fn():
    model = Sequential() # 딥러닝 모델의 구조를 시작합니다.
    model.add(Dense(24, input_dim=8, activation='relu'))
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

    5/5 [==============================] - 0s 3ms/step - loss: 0.5488 - accuracy: 0.7468
    5/5 [==============================] - 0s 3ms/step - loss: 0.7269 - accuracy: 0.6948
    5/5 [==============================] - 0s 3ms/step - loss: 0.4881 - accuracy: 0.7597
    5/5 [==============================] - 0s 4ms/step - loss: 0.6191 - accuracy: 0.6536
    5/5 [==============================] - 0s 3ms/step - loss: 0.6328 - accuracy: 0.7124
    정확도: [0.7467532753944397, 0.6948052048683167, 0.7597402334213257, 0.6535947918891907, 0.7124183177947998]
    정확도 평균: 0.7134623646736145



과제 2 - 피마 인디언 데이터를 활용, 이번 장에서 진행한 내용(모델 업데이트, 그래프로 과적합 확인, 학습의 자동 중단)을 실행 하는 코드를 작성하시오


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

# 피마 인디언 데이터 불러옵니다.
df = pd.read_csv('./data/pima-indians-diabetes3.csv')
df 

X = df.iloc[:,0:8]
y = df.iloc[:, 8]

```


```python
# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
# 모델이 3개의 은닉층을 갖도록 구조를 설계(설정)합니다.
model.add(Dense(30,  input_dim=8, activation='relu'))    # 첫 번째 은닉층은 30개 노드
model.add(Dense(12, activation='relu'))                   # 두 번째 은닉층은 12개 노드
model.add(Dense(8, activation='relu'))                    # 세 번째 은닉층은 8개 노드로 구성
model.add(Dense(1, activation='sigmoid'))                 # 출력층
model.summary()

# 모델을 컴파일합니다.
# 이진 분류를 위한 손실 함수(lost function) : binary_crossentropy
# 옵티마이저 : adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

    Model: "sequential_27"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_83 (Dense)            (None, 30)                270       
                                                                     
     dense_84 (Dense)            (None, 12)                372       
                                                                     
     dense_85 (Dense)            (None, 8)                 104       
                                                                     
     dense_86 (Dense)            (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 755
    Trainable params: 755
    Non-trainable params: 0
    _________________________________________________________________



```python
# 그래프 확인을 위한 긴 학습 
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25)
```

    Epoch 1/2000
    1/1 [==============================] - 1s 750ms/step - loss: 22.4350 - accuracy: 0.6478 - val_loss: 21.0081 - val_accuracy: 0.6039
    Epoch 2/2000
    1/1 [==============================] - 0s 34ms/step - loss: 21.2916 - accuracy: 0.6478 - val_loss: 19.8725 - val_accuracy: 0.6039
    ...
    Epoch 1998/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3492 - accuracy: 0.8500 - val_loss: 0.6134 - val_accuracy: 0.7273
    Epoch 1999/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3490 - accuracy: 0.8543 - val_loss: 0.6121 - val_accuracy: 0.7273
    Epoch 2000/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.3488 - accuracy: 0.8522 - val_loss: 0.6129 - val_accuracy: 0.7273



```python
# history에 저장된 학습 결과를 확인해 보겠습니다. 
hist_df=pd.DataFrame(history.history)
hist_df
```





  <div id="df-446c4ff7-b9c5-48c5-93d7-1287b24f3522">
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
      <th>loss</th>
      <th>accuracy</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.435047</td>
      <td>0.647826</td>
      <td>21.008080</td>
      <td>0.603896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.291559</td>
      <td>0.647826</td>
      <td>19.872471</td>
      <td>0.603896</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.154764</td>
      <td>0.647826</td>
      <td>18.739431</td>
      <td>0.603896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.023623</td>
      <td>0.647826</td>
      <td>17.604630</td>
      <td>0.603896</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.901346</td>
      <td>0.647826</td>
      <td>16.478224</td>
      <td>0.603896</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>0.349483</td>
      <td>0.850000</td>
      <td>0.611888</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>0.349430</td>
      <td>0.850000</td>
      <td>0.610848</td>
      <td>0.720779</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>0.349215</td>
      <td>0.850000</td>
      <td>0.613427</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>0.349040</td>
      <td>0.854348</td>
      <td>0.612054</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>0.348809</td>
      <td>0.852174</td>
      <td>0.612860</td>
      <td>0.727273</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-446c4ff7-b9c5-48c5-93d7-1287b24f3522')"
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
          document.querySelector('#df-446c4ff7-b9c5-48c5-93d7-1287b24f3522 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-446c4ff7-b9c5-48c5-93d7-1287b24f3522');
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
# y_vloss에 검증(validation) 데이터셋에 대한 오차를 저장합니다.
y_vloss=hist_df['val_loss']

# y_loss에 학습(train) 데이터셋의 오차를 저장합니다.
y_loss=hist_df['loss']

# x에 에포크 값을 지정하고
# 검증 데이터셋에 대한 오차를 빨간색으로, 학습셋에 대한 오차를 파란색으로 표시합니다.
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Validation loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trains loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_41_0.png)
    



```python
# 학습이 언제 자동 중단될지를 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 최적화 모델이 저장될 폴더와 모델의 이름을 정합니다.
modelpath="./data/model/Ch14-4-bestmodel.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 모델을 실행합니다.
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1, callbacks=[early_stopping_callback,checkpointer])

```

    Epoch 1/2000
    1/1 [==============================] - 0s 95ms/step - loss: 0.3486 - accuracy: 0.8543 - val_loss: 0.6114 - val_accuracy: 0.7273
    Epoch 2/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.3485 - accuracy: 0.8565 - val_loss: 0.6126 - val_accuracy: 0.7208
    Epoch 3/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3483 - accuracy: 0.8565 - val_loss: 0.6112 - val_accuracy: 0.7273
    ...
    Epoch 21/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.3460 - accuracy: 0.8500 - val_loss: 0.6148 - val_accuracy: 0.7273
    Epoch 22/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.3459 - accuracy: 0.8587 - val_loss: 0.6137 - val_accuracy: 0.7273
    Epoch 23/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3457 - accuracy: 0.8609 - val_loss: 0.6146 - val_accuracy: 0.7273



```python
# 테스트 결과를 출력합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```

    5/5 [==============================] - 0s 3ms/step - loss: 0.6057 - accuracy: 0.7403
    Test accuracy: 0.7402597665786743


***

# 15장 실제 데이터로 만들어 보는 모델

지금까지 한 몇 가지 실습은 참 또는 거짓을 맞히거나 여러 개의 보기 중 하나를 예측하는 분류 문제였습니다. 그런데 이번에는 수치를 예측하는 문제입니다. 준비된 **데이터는 아오와주 에임스 지역에서 2006년부터 2010년까지 거래된 실제 부동산 판매 기록입니다.** 주거 유형, 차고, 자재 및 환경에 대한 80개의 서로 다른 속성을 이용해 집의 가격을 예측해 볼 예정인데 **오랜 시간 사람이 일일이 기록하다 보니 빠진 부분도 많고, 집에 따라 어떤 항목은 범위에서 너무 벗어나 있기도 하며, 또 가격과는 관계가 없는 정보가 포함되어 있기도 합니다.** 실제 현장에서 만나게 되는 이런 류의 데이터를 어떻게 다루어야 하는지 이 장에서 학습해 보겠습니다. 

## 1. 데이터 파악하기
먼저 데이터를 불러와 확인해 보겠습니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# 깃허브에 준비된 데이터를 가져오기
!git clone https://github.com/taehojo/data.git

# 집 값 데이터를 불러오기
df = pd.read_csv("./data/house_train.csv")
df
```

    fatal: destination path 'data' already exists and is not an empty directory.






  <div id="df-eb63a690-9402-4e3b-8c20-49d086cef50b">
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
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
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-eb63a690-9402-4e3b-8c20-49d086cef50b')"
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
          document.querySelector('#df-eb63a690-9402-4e3b-8c20-49d086cef50b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-eb63a690-9402-4e3b-8c20-49d086cef50b');
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
df.shape
```




    (1460, 81)



위 데이터 프레임(df)는 총 1,460개의 행과 81개의 열로 구성되어 있습니다. 즉, 1,460개의 샘플이 있고 (Id열 제외)80개의 속성으로 구성된 것을 알 수 있습니다. 각 열이 어떤 유형의 데이터인지 확인해 보겠습니다.


```python
df.dtypes
```




    Id                 int64
    MSSubClass         int64
    MSZoning          object
    LotFrontage      float64
    LotArea            int64
                      ...   
    MoSold             int64
    YrSold             int64
    SaleType          object
    SaleCondition     object
    SalePrice          int64
    Length: 81, dtype: object



정수형(int64), 실수형(float64) 그리고 오브젝트형(object) 형태와 위에서 출력되지 않고 생략된 열이 다른 타입의 테이터일 수도 있습니다.  

## 2. 결측치, 카테고리 변수 처리하기
앞 장에서 다루었던 데이터와의 차이점은 아직 전처리가 끝나지 않은 상태의 데이터라 값이 없는 결측치가 있다는 것입니다. **결측치가 있는지 알아 보는 함수는 isnull() 함수입니다.** 결측치가 모두 몇 개인지 세어 가장 많은 것부터 순서대로 나열한 후 처음 20개만 출력하는 코드는 다음과 같습니다.


```python
df.isnull().sum().sort_values(ascending = False).head(20)
```




    PoolQC          1453
    MiscFeature     1406
    Alley           1369
    Fence           1179
    FireplaceQu      690
    LotFrontage      259
    GarageYrBlt       81
    GarageCond        81
    GarageType        81
    GarageFinish      81
    GarageQual        81
    BsmtFinType2      38
    BsmtExposure      38
    BsmtQual          37
    BsmtCond          37
    BsmtFinType1      37
    MasVnrArea         8
    MasVnrType         8
    Electrical         1
    Id                 0
    dtype: int64



(강사 - PoolQC는 pool qc로 보면 될 것 같고 pool은 수영장, qc는 quality control로 품질 관리 따라서 PoolQC는 수영장 관리 상태 정도로 해석하면 어떨까 싶습니다.)

결측치가 가장 많은 항목은 PoolQC입니다. 샘플 개수가 총 1,460개인데 PoolQC의 값이 누락된 샘플이 1453면 수영장 관리 상태에 대한 기록은 거의 다 빠져 있다고 봐야할 것 같습니다.

(강사 - 실 데이터셋에서 특정 속성에 결측치가 많다는 것은 해당 속성에 큰 의미를 두지 않았기 때문일 수 있음. 즉, 집값에 영향을 주는 요소가 **아닐 수도 있음**)

모델을 만들기 전에 데이터를 전처리하겠습니다. 먼저 12.3절에서 소개되었던 판다스의 get_dummies() 함수를 이용해 카타고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줍니다.

```
df = pd.get_dummies(df)
```

그리고 누락된 데이터, 결측치를 임의값으로 채워줄 생각입니다. 결측치를 채워주는 함수는 판다스의 fillna()입니다. 해당 항목의 평균 값으로 누락된 데이터를 대체하려고 합니다. (**강사 의견 - 이런 처리가 그러니까 누락된 데이터를 평균 값으로 대체하는 게 합리적인가 하는 생각이 듭니다. 이렇게 하는 것이 올바른 전처리 일까요?**)


```
df = df.fillna(df.mean())
```

[review 참고]






```python
# 카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줌
df = pd.get_dummies(df)

# 결측치를 전체 컬럼의 평균으로 대체
df = df.fillna(df.mean())

df
```





  <div id="df-fd16b207-b932-4b77-a79f-3a3fb5037017">
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>62.0</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>85.0</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>119.0</td>
      <td>790</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>66.0</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>0.0</td>
      <td>275</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>68.0</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>0.0</td>
      <td>49</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>75.0</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>0.0</td>
      <td>830</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 290 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fd16b207-b932-4b77-a79f-3a3fb5037017')"
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
          document.querySelector('#df-fd16b207-b932-4b77-a79f-3a3fb5037017 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fd16b207-b932-4b77-a79f-3a3fb5037017');
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
df.shape
```




    (1460, 290)



데이터 프레임의 행의 수 즉, 샘플 개수에는 변화가 없지만 원-핫 코딩으로 인해 열의 개수는 81개에서 290개로 늘었습니다.

카테고리형 변수를 0과 1로 이루어진 변수로 바꾸기 위해 ```get_dummies()```함수를 적용했습니다. 샘플 수에는 변화가 없지만 전체 열이 81개에서 290개로 늘었습니다. 

## 3. 속성별 관련도 추출하기

이중에서 우리에게 필요한 정보를 추출해 보겠습니다. 먼저 속성들 사이의 상관 관계를 ```df_corr```에 저장합니다. 그리고 집 값(SalePrice)와 상관 관계가 큰 요소를 순서대로 정렬합니다. 집 값과 상관 관계가 큰 20개의 속성을 확인합니다


```python
# 데이터 사이의 상관 관계를 저장
df_corr = df.corr()

# 집 값과 관련이 큰 것부터 순서대로 저장합니다.
df_corr_sort = df_corr.sort_values('SalePrice', ascending = False)

# 집 값과 관련도 가장 큰 10개의 속성들을 출력
df_corr_sort['SalePrice'].head(10)
```




    SalePrice       1.000000
    OverallQual     0.790982
    GrLivArea       0.708624
    GarageCars      0.640409
    GarageArea      0.623431
    TotalBsmtSF     0.613581
    1stFlrSF        0.605852
    FullBath        0.560664
    BsmtQual_Ex     0.553105
    TotRmsAbvGrd    0.533723
    Name: SalePrice, dtype: float64




```python
# 집 값과 관련도가 가장 높은 속성들(여기서는 5개)을 추출해서 상관도 그래프를 그려보자.
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
sns.pairplot(df[cols])
plt.show()
```


    
![png](output_61_0.png)
    


위 그래프에서 맨 왼쪽 열의 그림을 보십시요. 선택된 속성들이 집 값(SalePrice)와 양의 관계가 있음을 확인할 수 있습니다. 
* SalePrice vs. TotalBsmtSF
* SalePrice vs. GarageArea
* SalePrice vs. GarageCars
* SalePrice vs. GrLivArea
* SalePrice vs. OverallQual

## 4. 주택 가격 예측 모델
이제 앞서 고려한 중요 속성 5개 데이터로 학습 데이터 셋과 테스트 데이터셋을 만들어 보겠습니다. 집값을 y로 나머지 열을 X_train_pre로 저장한 후 전체 데이터셋의 80%를 학습 데이터셋으로 20%를 테스트 데이터셋으로 지정합니다.


```python
# 집 값을 제외한 나머지 열을 지정
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
X_train_pre = df[cols_train]

# 집 값을 저장
y = df['SalePrice'].values
```


```python
# 전체 80%를 학습셋으로, 20%를 테스트셋으로 지정
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size = 0.2)
```


```python
# 모델의 구조를 설정
model = Sequential()
model.add(Dense(10, input_dim = X_train.shape[1], activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 20)

modelpath = './data/model/Ch15-house.hdf5'

checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 0, save_best_only = True)

history = model.fit(X_train, y_train, validation_split = 0.25, epochs = 2000, batch_size = 32, callbacks=[early_stopping_callback, checkpointer])
```

    Model: "sequential_32"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_95 (Dense)            (None, 10)                60        
                                                                     
     dense_96 (Dense)            (None, 30)                330       
                                                                     
     dense_97 (Dense)            (None, 40)                1240      
                                                                     
     dense_98 (Dense)            (None, 1)                 41        
                                                                     
    =================================================================
    Total params: 1,671
    Trainable params: 1,671
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/2000
    28/28 [==============================] - 1s 17ms/step - loss: 38179950592.0000 - val_loss: 41170690048.0000
    Epoch 2/2000
    28/28 [==============================] - 0s 4ms/step - loss: 37575254016.0000 - val_loss: 40120983552.0000
    Epoch 3/2000
    28/28 [==============================] - 0s 5ms/step - loss: 35960553472.0000 - val_loss: 37357019136.0000
    Epoch 4/2000
    28/28 [==============================] - 0s 4ms/step - loss: 32036481024.0000 - val_loss: 31298416640.0000
    Epoch 5/2000
    28/28 [==============================] - 0s 5ms/step - loss: 24356808704.0000 - val_loss: 20643385344.0000
    Epoch 6/2000
    28/28 [==============================] - 0s 6ms/step - loss: 13357457408.0000 - val_loss: 8114371072.0000
    Epoch 7/2000
    28/28 [==============================] - 0s 5ms/step - loss: 4367775744.0000 - val_loss: 2130110720.0000
    Epoch 8/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2395318016.0000 - val_loss: 1800683776.0000
    Epoch 9/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2386240512.0000 - val_loss: 1824664192.0000
    Epoch 10/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2370490112.0000 - val_loss: 1858737792.0000
    Epoch 11/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2383420672.0000 - val_loss: 1817844224.0000
    Epoch 12/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2364204800.0000 - val_loss: 1860077312.0000
    Epoch 13/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2381326592.0000 - val_loss: 1851089152.0000
    Epoch 14/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2374814976.0000 - val_loss: 1828601856.0000
    Epoch 15/2000
    28/28 [==============================] - 0s 3ms/step - loss: 2367191296.0000 - val_loss: 1831890432.0000
    Epoch 16/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2366802944.0000 - val_loss: 1835865728.0000
    Epoch 17/2000
    28/28 [==============================] - 0s 3ms/step - loss: 2360563712.0000 - val_loss: 1849528704.0000
    Epoch 18/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2352229376.0000 - val_loss: 1824260096.0000
    Epoch 19/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2357563904.0000 - val_loss: 1818201344.0000
    Epoch 20/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2343276032.0000 - val_loss: 1857299200.0000
    Epoch 21/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2365795840.0000 - val_loss: 1849960192.0000
    Epoch 22/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2347627776.0000 - val_loss: 1838668672.0000
    Epoch 23/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2350676224.0000 - val_loss: 1826939392.0000
    Epoch 24/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2354743552.0000 - val_loss: 1822440320.0000
    Epoch 25/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2344488192.0000 - val_loss: 1838552960.0000
    Epoch 26/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2351053312.0000 - val_loss: 1804858880.0000
    Epoch 27/2000
    28/28 [==============================] - 0s 3ms/step - loss: 2334027520.0000 - val_loss: 1811475584.0000
    Epoch 28/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2347704320.0000 - val_loss: 1811726848.0000


#### [과제]
과제 1 - 실제 값(샘플) 20개를 뽑아 모델이 예측 값을 비교하는 그래프를 그리시오.다시 말해서 실제 값과 모델이 예측한 값을 한 그래프에 표시하시오.



```python
# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듬
real_prices = []
pred_prices = []
X_num = []

# 25개의 샘플을 뽑아 실제 값, 예측 값을 출력
n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(25):
  real = y_test[i]
  prediction = Y_prediction[i]
  print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
  real_prices.append(real)
  pred_prices.append(prediction)
  n_iter = n_iter + 1
  X_num.append(n_iter)

```

    10/10 [==============================] - 0s 2ms/step
    실제가격: 370878.00, 예상가격: 232468.09
    실제가격: 160000.00, 예상가격: 157380.53
    실제가격: 110000.00, 예상가격: 255639.88
    실제가격: 188700.00, 예상가격: 160001.53
    실제가격: 141000.00, 예상가격: 126377.66
    실제가격: 145500.00, 예상가격: 140601.39
    실제가격: 145000.00, 예상가격: 160111.20
    실제가격: 153337.00, 예상가격: 169800.38
    실제가격: 159500.00, 예상가격: 256175.97
    실제가격: 150750.00, 예상가격: 182702.94
    실제가격: 171500.00, 예상가격: 174222.97
    실제가격: 190000.00, 예상가격: 207140.88
    실제가격: 250000.00, 예상가격: 260130.86
    실제가격: 118500.00, 예상가격: 143233.97
    실제가격: 224900.00, 예상가격: 229251.59
    실제가격: 214000.00, 예상가격: 212157.38
    실제가격: 126000.00, 예상가격: 161610.59
    실제가격: 290000.00, 예상가격: 283505.88
    실제가격: 180000.00, 예상가격: 212765.98
    실제가격: 162000.00, 예상가격: 162430.06
    실제가격: 173000.00, 예상가격: 171607.45
    실제가격: 157000.00, 예상가격: 187405.97
    실제가격: 161500.00, 예상가격: 185076.38
    실제가격: 92900.00, 예상가격: 102655.59
    실제가격: 239000.00, 예상가격: 300423.50



```python
# 그래프를 통해 샘플로 뽑은 25개의 값을 비교해 봅니다.

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()
```


    
![png](output_69_0.png)
    



과제 2 - 이 예제에서는 집가격과 상관 관계가 높은 5개 속성을 입력으로 고려했습니다. 집 가격과 상관간계가 폰은 7개를 속성을 입력으로 예측된 집 가격과 실제 집 가격을 비교 하시요.


```python
cols_train = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','BsmtQual_Ex','TotRmsAbvGrd']
X_train_pre = df[cols_train]

y = df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size = 0.2)

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 20회 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델의 이름을 정합니다.
modelpath="./data/model/Ch15-house.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 실행 관련 설정을 하는 부분입니다. 전체의 20%를 검증셋으로 설정합니다. 
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])
```

    Model: "sequential_33"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_99 (Dense)            (None, 10)                100       
                                                                     
     dense_100 (Dense)           (None, 30)                330       
                                                                     
     dense_101 (Dense)           (None, 40)                1240      
                                                                     
     dense_102 (Dense)           (None, 1)                 41        
                                                                     
    =================================================================
    Total params: 1,711
    Trainable params: 1,711
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/2000
    28/28 [==============================] - 1s 9ms/step - loss: 37845663744.0000 - val_loss: 38447656960.0000
    Epoch 2/2000
    28/28 [==============================] - 0s 5ms/step - loss: 37483573248.0000 - val_loss: 37964185600.0000
    Epoch 3/2000
    28/28 [==============================] - 0s 5ms/step - loss: 36722409472.0000 - val_loss: 36741939200.0000
    Epoch 4/2000
    28/28 [==============================] - 0s 5ms/step - loss: 34823081984.0000 - val_loss: 33906014208.0000
    Epoch 5/2000
    28/28 [==============================] - 0s 4ms/step - loss: 30826899456.0000 - val_loss: 28243439616.0000
    Epoch 6/2000
    28/28 [==============================] - 0s 5ms/step - loss: 23802138624.0000 - val_loss: 19359379456.0000
    Epoch 7/2000
    28/28 [==============================] - 0s 5ms/step - loss: 14345934848.0000 - val_loss: 9442210816.0000
    Epoch 8/2000
    28/28 [==============================] - 0s 5ms/step - loss: 5911865344.0000 - val_loss: 3333312768.0000
    Epoch 9/2000
    28/28 [==============================] - 0s 7ms/step - loss: 2561738240.0000 - val_loss: 2316768000.0000
    Epoch 10/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2308529152.0000 - val_loss: 2310966016.0000
    Epoch 11/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2326520576.0000 - val_loss: 2309383936.0000
    Epoch 12/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2322880512.0000 - val_loss: 2312873216.0000
    Epoch 13/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2315247360.0000 - val_loss: 2309719040.0000
    Epoch 14/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2316100352.0000 - val_loss: 2309000192.0000
    Epoch 15/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2318244352.0000 - val_loss: 2306982144.0000
    Epoch 16/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2316640768.0000 - val_loss: 2308433408.0000
    Epoch 17/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2314188800.0000 - val_loss: 2304727040.0000
    Epoch 18/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2314873088.0000 - val_loss: 2302754048.0000
    Epoch 19/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2306924288.0000 - val_loss: 2302102784.0000
    Epoch 20/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2304410368.0000 - val_loss: 2306569216.0000
    Epoch 21/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2298486016.0000 - val_loss: 2299223040.0000
    Epoch 22/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2307127808.0000 - val_loss: 2301829632.0000
    Epoch 23/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2294808320.0000 - val_loss: 2296898304.0000
    Epoch 24/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2297516800.0000 - val_loss: 2297668864.0000
    Epoch 25/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2304985088.0000 - val_loss: 2294855936.0000
    Epoch 26/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2317645056.0000 - val_loss: 2297579520.0000
    Epoch 27/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2296615680.0000 - val_loss: 2294558208.0000
    Epoch 28/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2290450944.0000 - val_loss: 2295836160.0000
    Epoch 29/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2289033216.0000 - val_loss: 2291872768.0000
    Epoch 30/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2284200448.0000 - val_loss: 2297583872.0000
    Epoch 31/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2282879488.0000 - val_loss: 2294027776.0000
    Epoch 32/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2306920960.0000 - val_loss: 2296283904.0000
    Epoch 33/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2278308096.0000 - val_loss: 2289058048.0000
    Epoch 34/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2282315520.0000 - val_loss: 2286997760.0000
    Epoch 35/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2273213184.0000 - val_loss: 2285666560.0000
    Epoch 36/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2278596096.0000 - val_loss: 2285108736.0000
    Epoch 37/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2270151680.0000 - val_loss: 2283345152.0000
    Epoch 38/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2265329152.0000 - val_loss: 2283014912.0000
    Epoch 39/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2266867456.0000 - val_loss: 2281876992.0000
    Epoch 40/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2266694144.0000 - val_loss: 2282274048.0000
    Epoch 41/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2285655808.0000 - val_loss: 2281739520.0000
    Epoch 42/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2251983616.0000 - val_loss: 2282098944.0000
    Epoch 43/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2272666112.0000 - val_loss: 2282031104.0000
    Epoch 44/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2259436032.0000 - val_loss: 2278633216.0000
    Epoch 45/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2258193920.0000 - val_loss: 2279401216.0000
    Epoch 46/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2253610752.0000 - val_loss: 2276612864.0000
    Epoch 47/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2246085888.0000 - val_loss: 2284317184.0000
    Epoch 48/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2245098496.0000 - val_loss: 2274007040.0000
    Epoch 49/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2250433280.0000 - val_loss: 2274428416.0000
    Epoch 50/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2244152064.0000 - val_loss: 2272773888.0000
    Epoch 51/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2245462784.0000 - val_loss: 2275945984.0000
    Epoch 52/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2248175872.0000 - val_loss: 2271528448.0000
    Epoch 53/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2239478784.0000 - val_loss: 2273294848.0000
    Epoch 54/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2230283008.0000 - val_loss: 2270349824.0000
    Epoch 55/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2243379200.0000 - val_loss: 2268635136.0000
    Epoch 56/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2226735104.0000 - val_loss: 2268662272.0000
    Epoch 57/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2230682880.0000 - val_loss: 2268832512.0000
    Epoch 58/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2234137600.0000 - val_loss: 2267739904.0000
    Epoch 59/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2224451584.0000 - val_loss: 2270498304.0000
    Epoch 60/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2221895680.0000 - val_loss: 2266304768.0000
    Epoch 61/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2231958784.0000 - val_loss: 2266374400.0000
    Epoch 62/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2218867456.0000 - val_loss: 2263452672.0000
    Epoch 63/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2225822720.0000 - val_loss: 2263249920.0000
    Epoch 64/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2224522240.0000 - val_loss: 2262064640.0000
    Epoch 65/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2219053568.0000 - val_loss: 2268419072.0000
    Epoch 66/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2221756672.0000 - val_loss: 2264532736.0000
    Epoch 67/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2226964480.0000 - val_loss: 2260934912.0000
    Epoch 68/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2220934144.0000 - val_loss: 2263412992.0000
    Epoch 69/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2211807744.0000 - val_loss: 2259500032.0000
    Epoch 70/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2211983872.0000 - val_loss: 2260715008.0000
    Epoch 71/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2209399040.0000 - val_loss: 2258273280.0000
    Epoch 72/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2223734272.0000 - val_loss: 2261318144.0000
    Epoch 73/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2207872000.0000 - val_loss: 2262282496.0000
    Epoch 74/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2213481216.0000 - val_loss: 2260544768.0000
    Epoch 75/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2219409920.0000 - val_loss: 2256687360.0000
    Epoch 76/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2203919360.0000 - val_loss: 2262050048.0000
    Epoch 77/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2230810368.0000 - val_loss: 2254877952.0000
    Epoch 78/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2220006144.0000 - val_loss: 2257553152.0000
    Epoch 79/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2214296576.0000 - val_loss: 2255091200.0000
    Epoch 80/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2202480384.0000 - val_loss: 2253961728.0000
    Epoch 81/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2230841600.0000 - val_loss: 2261250816.0000
    Epoch 82/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2209461504.0000 - val_loss: 2261744640.0000
    Epoch 83/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2206098944.0000 - val_loss: 2252935680.0000
    Epoch 84/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2222011648.0000 - val_loss: 2251710464.0000
    Epoch 85/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2212869888.0000 - val_loss: 2252968448.0000
    Epoch 86/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2193309696.0000 - val_loss: 2250229248.0000
    Epoch 87/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2203387136.0000 - val_loss: 2250312704.0000
    Epoch 88/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2191696896.0000 - val_loss: 2250498816.0000
    Epoch 89/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2190819584.0000 - val_loss: 2248933120.0000
    Epoch 90/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2193292544.0000 - val_loss: 2248640768.0000
    Epoch 91/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2207744000.0000 - val_loss: 2254172160.0000
    Epoch 92/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2180420096.0000 - val_loss: 2248212480.0000
    Epoch 93/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2192857856.0000 - val_loss: 2255341056.0000
    Epoch 94/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2188899072.0000 - val_loss: 2254518528.0000
    Epoch 95/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2184037120.0000 - val_loss: 2246814720.0000
    Epoch 96/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2187944192.0000 - val_loss: 2249724672.0000
    Epoch 97/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2189427456.0000 - val_loss: 2245220608.0000
    Epoch 98/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2197420544.0000 - val_loss: 2245491712.0000
    Epoch 99/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2194396416.0000 - val_loss: 2244958464.0000
    Epoch 100/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2174223872.0000 - val_loss: 2250363648.0000
    Epoch 101/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2175042304.0000 - val_loss: 2243721472.0000
    Epoch 102/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2197355520.0000 - val_loss: 2245039104.0000
    Epoch 103/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2169475072.0000 - val_loss: 2245056512.0000
    Epoch 104/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2178245376.0000 - val_loss: 2245584384.0000
    Epoch 105/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2176217600.0000 - val_loss: 2244355840.0000
    Epoch 106/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2184546560.0000 - val_loss: 2243003648.0000
    Epoch 107/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2204133120.0000 - val_loss: 2242877440.0000
    Epoch 108/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2185464064.0000 - val_loss: 2242770432.0000
    Epoch 109/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2173322752.0000 - val_loss: 2242523648.0000
    Epoch 110/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2199230208.0000 - val_loss: 2243154688.0000
    Epoch 111/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2178800384.0000 - val_loss: 2249113088.0000
    Epoch 112/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2191299840.0000 - val_loss: 2239236608.0000
    Epoch 113/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2166279680.0000 - val_loss: 2238674176.0000
    Epoch 114/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2191990016.0000 - val_loss: 2238040576.0000
    Epoch 115/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2209887744.0000 - val_loss: 2242237696.0000
    Epoch 116/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2188603648.0000 - val_loss: 2239393280.0000
    Epoch 117/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2183698944.0000 - val_loss: 2247360256.0000
    Epoch 118/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2169538048.0000 - val_loss: 2238104576.0000
    Epoch 119/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2179270656.0000 - val_loss: 2237872896.0000
    Epoch 120/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2190565376.0000 - val_loss: 2237196032.0000
    Epoch 121/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2178722304.0000 - val_loss: 2243064320.0000
    Epoch 122/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2165057536.0000 - val_loss: 2237930496.0000
    Epoch 123/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2174440704.0000 - val_loss: 2236848384.0000
    Epoch 124/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2174325248.0000 - val_loss: 2236984832.0000
    Epoch 125/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2168133632.0000 - val_loss: 2237065472.0000
    Epoch 126/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2178351616.0000 - val_loss: 2239832320.0000
    Epoch 127/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2183344896.0000 - val_loss: 2254785280.0000
    Epoch 128/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2177965568.0000 - val_loss: 2236659200.0000
    Epoch 129/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2179424256.0000 - val_loss: 2249019648.0000
    Epoch 130/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2170596352.0000 - val_loss: 2236239616.0000
    Epoch 131/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2182307328.0000 - val_loss: 2236125184.0000
    Epoch 132/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2165276928.0000 - val_loss: 2236122624.0000
    Epoch 133/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2151797760.0000 - val_loss: 2236796928.0000
    Epoch 134/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2165919232.0000 - val_loss: 2242258944.0000
    Epoch 135/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2165022208.0000 - val_loss: 2234665728.0000
    Epoch 136/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2153760768.0000 - val_loss: 2236739328.0000
    Epoch 137/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2173508352.0000 - val_loss: 2233863168.0000
    Epoch 138/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2190447360.0000 - val_loss: 2233830144.0000
    Epoch 139/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2184851200.0000 - val_loss: 2235082240.0000
    Epoch 140/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2159213312.0000 - val_loss: 2234097408.0000
    Epoch 141/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2156803072.0000 - val_loss: 2233413632.0000
    Epoch 142/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2155487488.0000 - val_loss: 2239189504.0000
    Epoch 143/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2147668992.0000 - val_loss: 2232797952.0000
    Epoch 144/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2161239296.0000 - val_loss: 2234654208.0000
    Epoch 145/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2153033472.0000 - val_loss: 2232399872.0000
    Epoch 146/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2156442368.0000 - val_loss: 2234706944.0000
    Epoch 147/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2164251136.0000 - val_loss: 2232425984.0000
    Epoch 148/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2168699392.0000 - val_loss: 2232245248.0000
    Epoch 149/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2163898880.0000 - val_loss: 2231417856.0000
    Epoch 150/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2173642752.0000 - val_loss: 2230740480.0000
    Epoch 151/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2157713920.0000 - val_loss: 2230425344.0000
    Epoch 152/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2187398400.0000 - val_loss: 2230814976.0000
    Epoch 153/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2151896064.0000 - val_loss: 2230286592.0000
    Epoch 154/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2155236864.0000 - val_loss: 2231985664.0000
    Epoch 155/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2137961216.0000 - val_loss: 2234526720.0000
    Epoch 156/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2157567744.0000 - val_loss: 2229741056.0000
    Epoch 157/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2155572480.0000 - val_loss: 2231094016.0000
    Epoch 158/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2170930944.0000 - val_loss: 2230422272.0000
    Epoch 159/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2181462528.0000 - val_loss: 2229277696.0000
    Epoch 160/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2148773120.0000 - val_loss: 2228741888.0000
    Epoch 161/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2152799232.0000 - val_loss: 2228679936.0000
    Epoch 162/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2145415552.0000 - val_loss: 2228636672.0000
    Epoch 163/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2146456192.0000 - val_loss: 2227895552.0000
    Epoch 164/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2178097408.0000 - val_loss: 2231590656.0000
    Epoch 165/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2151084288.0000 - val_loss: 2227643392.0000
    Epoch 166/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2155024896.0000 - val_loss: 2228728832.0000
    Epoch 167/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2150305024.0000 - val_loss: 2228636928.0000
    Epoch 168/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2148523520.0000 - val_loss: 2227065344.0000
    Epoch 169/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2142421760.0000 - val_loss: 2225970432.0000
    Epoch 170/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2152705792.0000 - val_loss: 2227032832.0000
    Epoch 171/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2188560384.0000 - val_loss: 2229012992.0000
    Epoch 172/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2153514240.0000 - val_loss: 2229939456.0000
    Epoch 173/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2159204096.0000 - val_loss: 2226267392.0000
    Epoch 174/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2136814592.0000 - val_loss: 2227693312.0000
    Epoch 175/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2144459904.0000 - val_loss: 2228215040.0000
    Epoch 176/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2143072640.0000 - val_loss: 2225476096.0000
    Epoch 177/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2138946304.0000 - val_loss: 2225204224.0000
    Epoch 178/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2153453056.0000 - val_loss: 2224261888.0000
    Epoch 179/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2214576640.0000 - val_loss: 2234294528.0000
    Epoch 180/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2136413696.0000 - val_loss: 2227287296.0000
    Epoch 181/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2151048960.0000 - val_loss: 2225907456.0000
    Epoch 182/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140774528.0000 - val_loss: 2227231744.0000
    Epoch 183/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2143988736.0000 - val_loss: 2225812736.0000
    Epoch 184/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2147974144.0000 - val_loss: 2225363456.0000
    Epoch 185/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2148379136.0000 - val_loss: 2224869888.0000
    Epoch 186/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2141981184.0000 - val_loss: 2224933120.0000
    Epoch 187/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2161104128.0000 - val_loss: 2230530304.0000
    Epoch 188/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2148402944.0000 - val_loss: 2223377920.0000
    Epoch 189/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140934144.0000 - val_loss: 2224180992.0000
    Epoch 190/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2138535552.0000 - val_loss: 2222256640.0000
    Epoch 191/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2143516160.0000 - val_loss: 2222787584.0000
    Epoch 192/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2157566720.0000 - val_loss: 2222985216.0000
    Epoch 193/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2154274048.0000 - val_loss: 2221718016.0000
    Epoch 194/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140765312.0000 - val_loss: 2221724160.0000
    Epoch 195/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2145681664.0000 - val_loss: 2229287168.0000
    Epoch 196/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2153263360.0000 - val_loss: 2221650688.0000
    Epoch 197/2000
    28/28 [==============================] - 0s 9ms/step - loss: 2141934720.0000 - val_loss: 2224700416.0000
    Epoch 198/2000
    28/28 [==============================] - 0s 7ms/step - loss: 2155649536.0000 - val_loss: 2221923584.0000
    Epoch 199/2000
    28/28 [==============================] - 0s 8ms/step - loss: 2154720768.0000 - val_loss: 2220759552.0000
    Epoch 200/2000
    28/28 [==============================] - 0s 9ms/step - loss: 2159313152.0000 - val_loss: 2220185856.0000
    Epoch 201/2000
    28/28 [==============================] - 0s 7ms/step - loss: 2141913728.0000 - val_loss: 2220808960.0000
    Epoch 202/2000
    28/28 [==============================] - 0s 9ms/step - loss: 2137385856.0000 - val_loss: 2221761280.0000
    Epoch 203/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2155780864.0000 - val_loss: 2218953472.0000
    Epoch 204/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2139730560.0000 - val_loss: 2220092928.0000
    Epoch 205/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2142677120.0000 - val_loss: 2219136256.0000
    Epoch 206/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2151343360.0000 - val_loss: 2222477568.0000
    Epoch 207/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2127565440.0000 - val_loss: 2220066048.0000
    Epoch 208/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2145870976.0000 - val_loss: 2216585984.0000
    Epoch 209/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2137487360.0000 - val_loss: 2222403072.0000
    Epoch 210/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2153020416.0000 - val_loss: 2223906560.0000
    Epoch 211/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2131663616.0000 - val_loss: 2244442624.0000
    Epoch 212/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2134788480.0000 - val_loss: 2219475712.0000
    Epoch 213/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2136484224.0000 - val_loss: 2217653504.0000
    Epoch 214/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2132104576.0000 - val_loss: 2217772288.0000
    Epoch 215/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132835328.0000 - val_loss: 2218470400.0000
    Epoch 216/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2149467648.0000 - val_loss: 2216638720.0000
    Epoch 217/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2149409536.0000 - val_loss: 2215920128.0000
    Epoch 218/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2145366272.0000 - val_loss: 2217599232.0000
    Epoch 219/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2148069632.0000 - val_loss: 2216150272.0000
    Epoch 220/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2135922048.0000 - val_loss: 2215459840.0000
    Epoch 221/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2149266176.0000 - val_loss: 2215814912.0000
    Epoch 222/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2145762688.0000 - val_loss: 2214626560.0000
    Epoch 223/2000
    28/28 [==============================] - 0s 11ms/step - loss: 2131507328.0000 - val_loss: 2216121856.0000
    Epoch 224/2000
    28/28 [==============================] - 0s 12ms/step - loss: 2140930688.0000 - val_loss: 2215629568.0000
    Epoch 225/2000
    28/28 [==============================] - 0s 10ms/step - loss: 2149328896.0000 - val_loss: 2217724160.0000
    Epoch 226/2000
    28/28 [==============================] - 0s 12ms/step - loss: 2142958976.0000 - val_loss: 2214088960.0000
    Epoch 227/2000
    28/28 [==============================] - 0s 13ms/step - loss: 2140782848.0000 - val_loss: 2214987776.0000
    Epoch 228/2000
    28/28 [==============================] - 0s 15ms/step - loss: 2131587968.0000 - val_loss: 2214240256.0000
    Epoch 229/2000
    28/28 [==============================] - 1s 21ms/step - loss: 2131039232.0000 - val_loss: 2213812992.0000
    Epoch 230/2000
    28/28 [==============================] - 0s 16ms/step - loss: 2130892544.0000 - val_loss: 2212870400.0000
    Epoch 231/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2128942080.0000 - val_loss: 2212329984.0000
    Epoch 232/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2134857344.0000 - val_loss: 2215397888.0000
    Epoch 233/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2140189568.0000 - val_loss: 2213385472.0000
    Epoch 234/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140417792.0000 - val_loss: 2213165056.0000
    Epoch 235/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140208384.0000 - val_loss: 2212902656.0000
    Epoch 236/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132683776.0000 - val_loss: 2212313600.0000
    Epoch 237/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2143713408.0000 - val_loss: 2210609408.0000
    Epoch 238/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2133054208.0000 - val_loss: 2213255424.0000
    Epoch 239/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2130107776.0000 - val_loss: 2210771200.0000
    Epoch 240/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2145172224.0000 - val_loss: 2212392960.0000
    Epoch 241/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2151425536.0000 - val_loss: 2211215872.0000
    Epoch 242/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2139470208.0000 - val_loss: 2210908928.0000
    Epoch 243/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2149855488.0000 - val_loss: 2209306880.0000
    Epoch 244/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2157273600.0000 - val_loss: 2209244928.0000
    Epoch 245/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2137400320.0000 - val_loss: 2215038976.0000
    Epoch 246/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2149803264.0000 - val_loss: 2211070976.0000
    Epoch 247/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2141786240.0000 - val_loss: 2210964736.0000
    Epoch 248/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2142523008.0000 - val_loss: 2208123648.0000
    Epoch 249/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2134781696.0000 - val_loss: 2208436736.0000
    Epoch 250/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127449984.0000 - val_loss: 2207995136.0000
    Epoch 251/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2137942016.0000 - val_loss: 2207383040.0000
    Epoch 252/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2129810048.0000 - val_loss: 2206904832.0000
    Epoch 253/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2141224960.0000 - val_loss: 2206982400.0000
    Epoch 254/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2142199424.0000 - val_loss: 2206914816.0000
    Epoch 255/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2139635840.0000 - val_loss: 2208333824.0000
    Epoch 256/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2140148352.0000 - val_loss: 2206509312.0000
    Epoch 257/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2135231616.0000 - val_loss: 2206258432.0000
    Epoch 258/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2143914496.0000 - val_loss: 2205544960.0000
    Epoch 259/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2139856128.0000 - val_loss: 2205399040.0000
    Epoch 260/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132527232.0000 - val_loss: 2204905216.0000
    Epoch 261/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2190697728.0000 - val_loss: 2223087872.0000
    Epoch 262/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2171185408.0000 - val_loss: 2207088896.0000
    Epoch 263/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125191296.0000 - val_loss: 2208217600.0000
    Epoch 264/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2129158144.0000 - val_loss: 2207113472.0000
    Epoch 265/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2136310784.0000 - val_loss: 2206268672.0000
    Epoch 266/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128118016.0000 - val_loss: 2206550784.0000
    Epoch 267/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2155283456.0000 - val_loss: 2206949376.0000
    Epoch 268/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2127413632.0000 - val_loss: 2207008512.0000
    Epoch 269/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2131979392.0000 - val_loss: 2207461632.0000
    Epoch 270/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2131856640.0000 - val_loss: 2211526144.0000
    Epoch 271/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127935488.0000 - val_loss: 2204849152.0000
    Epoch 272/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2127706880.0000 - val_loss: 2205735424.0000
    Epoch 273/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125577728.0000 - val_loss: 2209182976.0000
    Epoch 274/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2128655360.0000 - val_loss: 2204796416.0000
    Epoch 275/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2146360320.0000 - val_loss: 2204063232.0000
    Epoch 276/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2141009536.0000 - val_loss: 2204763904.0000
    Epoch 277/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2131814016.0000 - val_loss: 2203770880.0000
    Epoch 278/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2147546112.0000 - val_loss: 2204842496.0000
    Epoch 279/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2130462208.0000 - val_loss: 2210213376.0000
    Epoch 280/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2147606528.0000 - val_loss: 2205876736.0000
    Epoch 281/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2124530816.0000 - val_loss: 2204802304.0000
    Epoch 282/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125438976.0000 - val_loss: 2203945472.0000
    Epoch 283/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2135811200.0000 - val_loss: 2214194688.0000
    Epoch 284/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117789696.0000 - val_loss: 2207571456.0000
    Epoch 285/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2126026496.0000 - val_loss: 2202923520.0000
    Epoch 286/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121720320.0000 - val_loss: 2203966976.0000
    Epoch 287/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2138292736.0000 - val_loss: 2206273536.0000
    Epoch 288/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2179080960.0000 - val_loss: 2201875456.0000
    Epoch 289/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128554880.0000 - val_loss: 2204315392.0000
    Epoch 290/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2135451008.0000 - val_loss: 2202242304.0000
    Epoch 291/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2129979264.0000 - val_loss: 2202898944.0000
    Epoch 292/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2124500992.0000 - val_loss: 2201405696.0000
    Epoch 293/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2127214208.0000 - val_loss: 2201594368.0000
    Epoch 294/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2130809088.0000 - val_loss: 2205769728.0000
    Epoch 295/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2141263744.0000 - val_loss: 2201753600.0000
    Epoch 296/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132848640.0000 - val_loss: 2200438784.0000
    Epoch 297/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127102848.0000 - val_loss: 2199906304.0000
    Epoch 298/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121310336.0000 - val_loss: 2201019136.0000
    Epoch 299/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118845184.0000 - val_loss: 2199556608.0000
    Epoch 300/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122544512.0000 - val_loss: 2199135744.0000
    Epoch 301/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128717056.0000 - val_loss: 2199250688.0000
    Epoch 302/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127200256.0000 - val_loss: 2198667264.0000
    Epoch 303/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2156046336.0000 - val_loss: 2198492928.0000
    Epoch 304/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2129587200.0000 - val_loss: 2197553664.0000
    Epoch 305/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2129564160.0000 - val_loss: 2200962304.0000
    Epoch 306/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2138460800.0000 - val_loss: 2200583168.0000
    Epoch 307/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2114732288.0000 - val_loss: 2201169920.0000
    Epoch 308/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2132846848.0000 - val_loss: 2197938432.0000
    Epoch 309/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2135919232.0000 - val_loss: 2197734400.0000
    Epoch 310/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2144680448.0000 - val_loss: 2197081856.0000
    Epoch 311/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2121990400.0000 - val_loss: 2197278976.0000
    Epoch 312/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2133314304.0000 - val_loss: 2197618688.0000
    Epoch 313/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118989440.0000 - val_loss: 2196603136.0000
    Epoch 314/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121615872.0000 - val_loss: 2198566656.0000
    Epoch 315/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132604928.0000 - val_loss: 2197964800.0000
    Epoch 316/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118674688.0000 - val_loss: 2195865088.0000
    Epoch 317/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2118627456.0000 - val_loss: 2197684480.0000
    Epoch 318/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2121182208.0000 - val_loss: 2195613952.0000
    Epoch 319/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2124405760.0000 - val_loss: 2194697728.0000
    Epoch 320/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2124278272.0000 - val_loss: 2195559168.0000
    Epoch 321/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2150552064.0000 - val_loss: 2197407744.0000
    Epoch 322/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121694336.0000 - val_loss: 2197280000.0000
    Epoch 323/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2132936064.0000 - val_loss: 2195389440.0000
    Epoch 324/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2116339712.0000 - val_loss: 2194468352.0000
    Epoch 325/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2123691392.0000 - val_loss: 2194474752.0000
    Epoch 326/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2144128896.0000 - val_loss: 2193816576.0000
    Epoch 327/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2147904512.0000 - val_loss: 2194220800.0000
    Epoch 328/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2115162624.0000 - val_loss: 2192937984.0000
    Epoch 329/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2130037888.0000 - val_loss: 2192954112.0000
    Epoch 330/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140826368.0000 - val_loss: 2193746176.0000
    Epoch 331/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2140223104.0000 - val_loss: 2195061760.0000
    Epoch 332/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2116172672.0000 - val_loss: 2199096576.0000
    Epoch 333/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2120029952.0000 - val_loss: 2194006784.0000
    Epoch 334/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121294336.0000 - val_loss: 2193224704.0000
    Epoch 335/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2136419712.0000 - val_loss: 2194705920.0000
    Epoch 336/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117234944.0000 - val_loss: 2193194752.0000
    Epoch 337/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2116291328.0000 - val_loss: 2192560128.0000
    Epoch 338/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2119075328.0000 - val_loss: 2197600256.0000
    Epoch 339/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2136302464.0000 - val_loss: 2192919040.0000
    Epoch 340/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2119018752.0000 - val_loss: 2193873408.0000
    Epoch 341/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2129562112.0000 - val_loss: 2191169792.0000
    Epoch 342/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2131630976.0000 - val_loss: 2192382720.0000
    Epoch 343/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2113391104.0000 - val_loss: 2194508032.0000
    Epoch 344/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117088128.0000 - val_loss: 2191693568.0000
    Epoch 345/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2119058560.0000 - val_loss: 2191765504.0000
    Epoch 346/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2121225216.0000 - val_loss: 2190715392.0000
    Epoch 347/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122637568.0000 - val_loss: 2190540032.0000
    Epoch 348/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2125532032.0000 - val_loss: 2190455808.0000
    Epoch 349/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2124892032.0000 - val_loss: 2190386688.0000
    Epoch 350/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2128584832.0000 - val_loss: 2190055680.0000
    Epoch 351/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2124913408.0000 - val_loss: 2189873408.0000
    Epoch 352/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117187200.0000 - val_loss: 2189161472.0000
    Epoch 353/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2120524800.0000 - val_loss: 2188965376.0000
    Epoch 354/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2120496640.0000 - val_loss: 2189466112.0000
    Epoch 355/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127145344.0000 - val_loss: 2188263680.0000
    Epoch 356/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2130125696.0000 - val_loss: 2191098368.0000
    Epoch 357/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2113219072.0000 - val_loss: 2194230272.0000
    Epoch 358/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128198528.0000 - val_loss: 2189905152.0000
    Epoch 359/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2124012160.0000 - val_loss: 2199602688.0000
    Epoch 360/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2136206080.0000 - val_loss: 2191606272.0000
    Epoch 361/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2122288256.0000 - val_loss: 2189756928.0000
    Epoch 362/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2137994112.0000 - val_loss: 2189487104.0000
    Epoch 363/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2134557312.0000 - val_loss: 2188325120.0000
    Epoch 364/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113570688.0000 - val_loss: 2188934144.0000
    Epoch 365/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2123584640.0000 - val_loss: 2191894272.0000
    Epoch 366/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2114863744.0000 - val_loss: 2189398016.0000
    Epoch 367/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2119340928.0000 - val_loss: 2188455680.0000
    Epoch 368/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117313280.0000 - val_loss: 2188275456.0000
    Epoch 369/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2110776448.0000 - val_loss: 2188817152.0000
    Epoch 370/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2123463936.0000 - val_loss: 2189999104.0000
    Epoch 371/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2132876544.0000 - val_loss: 2194463744.0000
    Epoch 372/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122601600.0000 - val_loss: 2187991296.0000
    Epoch 373/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2130301440.0000 - val_loss: 2187996416.0000
    Epoch 374/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118819840.0000 - val_loss: 2191314944.0000
    Epoch 375/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2146600576.0000 - val_loss: 2187983104.0000
    Epoch 376/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2139711104.0000 - val_loss: 2194724608.0000
    Epoch 377/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117210496.0000 - val_loss: 2188952576.0000
    Epoch 378/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118936576.0000 - val_loss: 2186050560.0000
    Epoch 379/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2121716736.0000 - val_loss: 2185976576.0000
    Epoch 380/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2133572480.0000 - val_loss: 2186239232.0000
    Epoch 381/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106174848.0000 - val_loss: 2189659648.0000
    Epoch 382/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117766784.0000 - val_loss: 2189017600.0000
    Epoch 383/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2113358592.0000 - val_loss: 2183478272.0000
    Epoch 384/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2120620032.0000 - val_loss: 2183954176.0000
    Epoch 385/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2119515776.0000 - val_loss: 2186656512.0000
    Epoch 386/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2129382528.0000 - val_loss: 2183656192.0000
    Epoch 387/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2109954688.0000 - val_loss: 2182378752.0000
    Epoch 388/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2109409280.0000 - val_loss: 2183750656.0000
    Epoch 389/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2114935296.0000 - val_loss: 2197650688.0000
    Epoch 390/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113468800.0000 - val_loss: 2183538432.0000
    Epoch 391/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2123155328.0000 - val_loss: 2185300224.0000
    Epoch 392/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2110407424.0000 - val_loss: 2191802624.0000
    Epoch 393/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2121428992.0000 - val_loss: 2182193920.0000
    Epoch 394/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112399488.0000 - val_loss: 2187314176.0000
    Epoch 395/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112567680.0000 - val_loss: 2183067904.0000
    Epoch 396/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2115770752.0000 - val_loss: 2180913664.0000
    Epoch 397/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2134287360.0000 - val_loss: 2182638080.0000
    Epoch 398/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2137363328.0000 - val_loss: 2180591872.0000
    Epoch 399/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128218624.0000 - val_loss: 2183840768.0000
    Epoch 400/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2124019200.0000 - val_loss: 2186969856.0000
    Epoch 401/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2108477312.0000 - val_loss: 2187952896.0000
    Epoch 402/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125790592.0000 - val_loss: 2182024960.0000
    Epoch 403/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2115114240.0000 - val_loss: 2180776960.0000
    Epoch 404/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2110901376.0000 - val_loss: 2180188928.0000
    Epoch 405/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2116677120.0000 - val_loss: 2180806912.0000
    Epoch 406/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2112874624.0000 - val_loss: 2180590080.0000
    Epoch 407/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117804928.0000 - val_loss: 2180199936.0000
    Epoch 408/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2129337216.0000 - val_loss: 2179287040.0000
    Epoch 409/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2113944960.0000 - val_loss: 2178447872.0000
    Epoch 410/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2123063552.0000 - val_loss: 2179575552.0000
    Epoch 411/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2132253568.0000 - val_loss: 2179503104.0000
    Epoch 412/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113134080.0000 - val_loss: 2183119872.0000
    Epoch 413/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101379840.0000 - val_loss: 2184235776.0000
    Epoch 414/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2116434432.0000 - val_loss: 2178809088.0000
    Epoch 415/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2115245568.0000 - val_loss: 2178852352.0000
    Epoch 416/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2124019840.0000 - val_loss: 2179876352.0000
    Epoch 417/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2108865152.0000 - val_loss: 2178346752.0000
    Epoch 418/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2126160256.0000 - val_loss: 2180173312.0000
    Epoch 419/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118135680.0000 - val_loss: 2178033408.0000
    Epoch 420/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2115679488.0000 - val_loss: 2178705664.0000
    Epoch 421/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106175744.0000 - val_loss: 2180805376.0000
    Epoch 422/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2109603200.0000 - val_loss: 2177920000.0000
    Epoch 423/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2109628928.0000 - val_loss: 2177698560.0000
    Epoch 424/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2127343360.0000 - val_loss: 2178376960.0000
    Epoch 425/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2113095040.0000 - val_loss: 2176301824.0000
    Epoch 426/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113735296.0000 - val_loss: 2178636544.0000
    Epoch 427/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2112608384.0000 - val_loss: 2175312896.0000
    Epoch 428/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125258240.0000 - val_loss: 2177139712.0000
    Epoch 429/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2116542592.0000 - val_loss: 2175302144.0000
    Epoch 430/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111914112.0000 - val_loss: 2178171648.0000
    Epoch 431/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112667904.0000 - val_loss: 2178240256.0000
    Epoch 432/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2103817344.0000 - val_loss: 2175999232.0000
    Epoch 433/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113581824.0000 - val_loss: 2176037888.0000
    Epoch 434/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117134976.0000 - val_loss: 2180289024.0000
    Epoch 435/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2125214592.0000 - val_loss: 2175721472.0000
    Epoch 436/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113817472.0000 - val_loss: 2175737344.0000
    Epoch 437/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2128224384.0000 - val_loss: 2176979200.0000
    Epoch 438/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2109376128.0000 - val_loss: 2175207168.0000
    Epoch 439/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2118068480.0000 - val_loss: 2175526656.0000
    Epoch 440/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121730944.0000 - val_loss: 2175520768.0000
    Epoch 441/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111404288.0000 - val_loss: 2176327424.0000
    Epoch 442/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2108777472.0000 - val_loss: 2179269888.0000
    Epoch 443/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2116902272.0000 - val_loss: 2175255552.0000
    Epoch 444/2000
    28/28 [==============================] - 1s 23ms/step - loss: 2112065920.0000 - val_loss: 2175040768.0000
    Epoch 445/2000
    28/28 [==============================] - 1s 22ms/step - loss: 2106667264.0000 - val_loss: 2174454016.0000
    Epoch 446/2000
    28/28 [==============================] - 0s 10ms/step - loss: 2107842560.0000 - val_loss: 2175403776.0000
    Epoch 447/2000
    28/28 [==============================] - 0s 13ms/step - loss: 2110251008.0000 - val_loss: 2173357568.0000
    Epoch 448/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2118846976.0000 - val_loss: 2174160384.0000
    Epoch 449/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2120317312.0000 - val_loss: 2173829376.0000
    Epoch 450/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2133235840.0000 - val_loss: 2176242944.0000
    Epoch 451/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113872256.0000 - val_loss: 2173809920.0000
    Epoch 452/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117594752.0000 - val_loss: 2175737600.0000
    Epoch 453/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2102917504.0000 - val_loss: 2176284672.0000
    Epoch 454/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2119819648.0000 - val_loss: 2171618816.0000
    Epoch 455/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122428416.0000 - val_loss: 2174689536.0000
    Epoch 456/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2125794176.0000 - val_loss: 2177163008.0000
    Epoch 457/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117245696.0000 - val_loss: 2172165120.0000
    Epoch 458/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2104342272.0000 - val_loss: 2172426496.0000
    Epoch 459/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2110307200.0000 - val_loss: 2171448832.0000
    Epoch 460/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122707584.0000 - val_loss: 2170940160.0000
    Epoch 461/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2113531392.0000 - val_loss: 2174480896.0000
    Epoch 462/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2121379072.0000 - val_loss: 2173893632.0000
    Epoch 463/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111386752.0000 - val_loss: 2173580800.0000
    Epoch 464/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2118227968.0000 - val_loss: 2174856960.0000
    Epoch 465/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2111862016.0000 - val_loss: 2173320192.0000
    Epoch 466/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2117619072.0000 - val_loss: 2174916608.0000
    Epoch 467/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2105263488.0000 - val_loss: 2171997184.0000
    Epoch 468/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2110847872.0000 - val_loss: 2175868672.0000
    Epoch 469/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2107080576.0000 - val_loss: 2171381760.0000
    Epoch 470/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106776960.0000 - val_loss: 2174725376.0000
    Epoch 471/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2104035712.0000 - val_loss: 2174796032.0000
    Epoch 472/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2106085504.0000 - val_loss: 2169643008.0000
    Epoch 473/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101086976.0000 - val_loss: 2171874048.0000
    Epoch 474/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2107977984.0000 - val_loss: 2170299904.0000
    Epoch 475/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111518848.0000 - val_loss: 2169895680.0000
    Epoch 476/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2105996160.0000 - val_loss: 2168997632.0000
    Epoch 477/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112912384.0000 - val_loss: 2171628544.0000
    Epoch 478/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2104594048.0000 - val_loss: 2169315328.0000
    Epoch 479/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101941120.0000 - val_loss: 2172137728.0000
    Epoch 480/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106506880.0000 - val_loss: 2169423616.0000
    Epoch 481/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2101137920.0000 - val_loss: 2169018624.0000
    Epoch 482/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2097708544.0000 - val_loss: 2166763776.0000
    Epoch 483/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112108288.0000 - val_loss: 2169336320.0000
    Epoch 484/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2116253440.0000 - val_loss: 2166671104.0000
    Epoch 485/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2103262720.0000 - val_loss: 2169062656.0000
    Epoch 486/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2108843136.0000 - val_loss: 2172632064.0000
    Epoch 487/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2102068736.0000 - val_loss: 2167906304.0000
    Epoch 488/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111307136.0000 - val_loss: 2170341888.0000
    Epoch 489/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2105465984.0000 - val_loss: 2166379008.0000
    Epoch 490/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2110867072.0000 - val_loss: 2169890560.0000
    Epoch 491/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2115949312.0000 - val_loss: 2170488832.0000
    Epoch 492/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2120020480.0000 - val_loss: 2169437696.0000
    Epoch 493/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117211648.0000 - val_loss: 2166360832.0000
    Epoch 494/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2112401536.0000 - val_loss: 2171715840.0000
    Epoch 495/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2115296640.0000 - val_loss: 2167698688.0000
    Epoch 496/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2102913152.0000 - val_loss: 2170283008.0000
    Epoch 497/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2111764864.0000 - val_loss: 2168305408.0000
    Epoch 498/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2103479424.0000 - val_loss: 2167367424.0000
    Epoch 499/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2103724416.0000 - val_loss: 2166628352.0000
    Epoch 500/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2100662272.0000 - val_loss: 2166351872.0000
    Epoch 501/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2104525312.0000 - val_loss: 2167793408.0000
    Epoch 502/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2109827072.0000 - val_loss: 2165678592.0000
    Epoch 503/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2102986240.0000 - val_loss: 2165488640.0000
    Epoch 504/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2106998272.0000 - val_loss: 2165435904.0000
    Epoch 505/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2110341632.0000 - val_loss: 2169648384.0000
    Epoch 506/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2106084352.0000 - val_loss: 2164451840.0000
    Epoch 507/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2117384576.0000 - val_loss: 2165406464.0000
    Epoch 508/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106226688.0000 - val_loss: 2166173696.0000
    Epoch 509/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2114363904.0000 - val_loss: 2165145344.0000
    Epoch 510/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2101701632.0000 - val_loss: 2164876544.0000
    Epoch 511/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2110454144.0000 - val_loss: 2165064192.0000
    Epoch 512/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2095195904.0000 - val_loss: 2168208640.0000
    Epoch 513/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106943104.0000 - val_loss: 2166064384.0000
    Epoch 514/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2099887104.0000 - val_loss: 2163208704.0000
    Epoch 515/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2099794048.0000 - val_loss: 2162255872.0000
    Epoch 516/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2107731072.0000 - val_loss: 2165193472.0000
    Epoch 517/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2098805632.0000 - val_loss: 2168450304.0000
    Epoch 518/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2124669824.0000 - val_loss: 2163759616.0000
    Epoch 519/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101619328.0000 - val_loss: 2166675712.0000
    Epoch 520/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2122882560.0000 - val_loss: 2163211264.0000
    Epoch 521/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106741760.0000 - val_loss: 2164664064.0000
    Epoch 522/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2148046336.0000 - val_loss: 2166937856.0000
    Epoch 523/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2108119680.0000 - val_loss: 2166033152.0000
    Epoch 524/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2105936256.0000 - val_loss: 2165838080.0000
    Epoch 525/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2098386304.0000 - val_loss: 2164783872.0000
    Epoch 526/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2092430592.0000 - val_loss: 2163632896.0000
    Epoch 527/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2101611008.0000 - val_loss: 2165749760.0000
    Epoch 528/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2096711168.0000 - val_loss: 2163559424.0000
    Epoch 529/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2104152832.0000 - val_loss: 2161003008.0000
    Epoch 530/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2093313536.0000 - val_loss: 2162051328.0000
    Epoch 531/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2136669952.0000 - val_loss: 2176240640.0000
    Epoch 532/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2098102016.0000 - val_loss: 2161162240.0000
    Epoch 533/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2098244480.0000 - val_loss: 2164216832.0000
    Epoch 534/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2106407936.0000 - val_loss: 2160128768.0000
    Epoch 535/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2122308864.0000 - val_loss: 2164048128.0000
    Epoch 536/2000
    28/28 [==============================] - 0s 6ms/step - loss: 2112574464.0000 - val_loss: 2162985728.0000
    Epoch 537/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2104293504.0000 - val_loss: 2159522304.0000
    Epoch 538/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2102955520.0000 - val_loss: 2165669888.0000
    Epoch 539/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2090696448.0000 - val_loss: 2161045760.0000
    Epoch 540/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2092584576.0000 - val_loss: 2162687744.0000
    Epoch 541/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2093084928.0000 - val_loss: 2161252096.0000
    Epoch 542/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2099528192.0000 - val_loss: 2159023616.0000
    Epoch 543/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2096416000.0000 - val_loss: 2161722368.0000
    Epoch 544/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106028544.0000 - val_loss: 2164084224.0000
    Epoch 545/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2108577280.0000 - val_loss: 2156680704.0000
    Epoch 546/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2209112064.0000 - val_loss: 2168574976.0000
    Epoch 547/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101807488.0000 - val_loss: 2170158848.0000
    Epoch 548/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2093976448.0000 - val_loss: 2162755328.0000
    Epoch 549/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2102244352.0000 - val_loss: 2161693184.0000
    Epoch 550/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2096095232.0000 - val_loss: 2162303488.0000
    Epoch 551/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2096047616.0000 - val_loss: 2159550464.0000
    Epoch 552/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2108486144.0000 - val_loss: 2162570496.0000
    Epoch 553/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2095207296.0000 - val_loss: 2158736896.0000
    Epoch 554/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2098064256.0000 - val_loss: 2158376960.0000
    Epoch 555/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2101844096.0000 - val_loss: 2163914240.0000
    Epoch 556/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2102469248.0000 - val_loss: 2160344832.0000
    Epoch 557/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2095377792.0000 - val_loss: 2160948736.0000
    Epoch 558/2000
    28/28 [==============================] - 0s 5ms/step - loss: 2096589056.0000 - val_loss: 2158798848.0000
    Epoch 559/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2100995712.0000 - val_loss: 2160999168.0000
    Epoch 560/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2097477248.0000 - val_loss: 2163399168.0000
    Epoch 561/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2118508800.0000 - val_loss: 2159020800.0000
    Epoch 562/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2102112000.0000 - val_loss: 2166404352.0000
    Epoch 563/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2103068544.0000 - val_loss: 2162768640.0000
    Epoch 564/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2106238720.0000 - val_loss: 2157551616.0000
    Epoch 565/2000
    28/28 [==============================] - 0s 4ms/step - loss: 2098352128.0000 - val_loss: 2161246464.0000



```python
# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
real_prices =[]
pred_prices = []
X_num = []

# 25개의 샘플을 뽑아 실제 값, 예측 값을 출력해 봅니다. 
n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)
```

    10/10 [==============================] - 0s 2ms/step
    실제가격: 135000.00, 예상가격: 128777.05
    실제가격: 213000.00, 예상가격: 180473.20
    실제가격: 207500.00, 예상가격: 235001.75
    실제가격: 270000.00, 예상가격: 193644.30
    실제가격: 140000.00, 예상가격: 133359.56
    실제가격: 230000.00, 예상가격: 186328.53
    실제가격: 426000.00, 예상가격: 293966.72
    실제가격: 140000.00, 예상가격: 200886.77
    실제가격: 119500.00, 예상가격: 120022.17
    실제가격: 144000.00, 예상가격: 151178.81
    실제가격: 101000.00, 예상가격: 119192.53
    실제가격: 173500.00, 예상가격: 177343.78
    실제가격: 165000.00, 예상가격: 165637.11
    실제가격: 172500.00, 예상가격: 195765.12
    실제가격: 163500.00, 예상가격: 133594.06
    실제가격: 175000.00, 예상가격: 186796.91
    실제가격: 120000.00, 예상가격: 128624.90
    실제가격: 125500.00, 예상가격: 145157.00
    실제가격: 145000.00, 예상가격: 174973.05
    실제가격: 152000.00, 예상가격: 159674.50
    실제가격: 120000.00, 예상가격: 123029.12
    실제가격: 147000.00, 예상가격: 160102.83
    실제가격: 181134.00, 예상가격: 170064.94
    실제가격: 423000.00, 예상가격: 318571.34
    실제가격: 162900.00, 예상가격: 172703.23



```python
# 그래프를 통해 샘플로 뽑은 25개의 값을 비교해 봅니다.

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()
```


    
![png](output_73_0.png)
    


***

# 16장 이미지 인식의 꽃, 컨볼루션 신경망(CNN)

<br><center>
<img src ="https://drive.google.com/uc?id=1XnLRGBH8hQRhTl-UBrNJr-Odn285dHhX" width=400>
</center><br>
급히 전달 받은 노트에 숫자가 적혀 있습니다. 뭐라고 쓰여있는지 읽기에 어렵지 않습니다. 일반적인 사람에게 이 사진에 나온 숫자를 읽어 보라고 하면 대부분 '504192'라고 일겠지요. 그런데 컴퓨터에 이 글씨를 읽게 하고 이 글씨가 어떤 의미인지 알게 하는 과정은 쉽지 않습니다. 사람이 볼 때는 쉽게 알수 있는 글씨라고 하더라도 숫자 5는 어떤 특징을 가졌고 숫자 9는 6과 어떻게 다른지 기계가 스스로 파악해 정확하게 읽고 판단하게 만드는 것은 머신 러닝의 오랜 진입 과제였습니다. 

**MNIST 데이터셋은 미국 국립표준기술원(NIST)이 고등학생과 인구조사국 직원 등이 쓴 손 글씨를 이용해 만든 데이터로 구성되어 있습니다.** 7만 개의 글자 이미지에 각각 0부터 9까지 이름표를 붙인 데이터셋으로 머신 러닝을 배우는 사람이라면 자신의 알고리즘과 다른 알고리즘의 성과를 비교해 보고자 한 번씩 도전해 보는 유명한 데이터 중 하나 입니다. 

<br><center>
(그림. 16-1) MNIST 손글씨 데이터 이미지<br>
<img src="https://drive.google.com/uc?id=1Qk2TBWCeUWTwoKjgj741m12W5akADIKJ">
</center><br>  

지금까지 배운 딥러닝을 이용해서 과연 이 손글씨 이미지를 몇 %나 정확히 예측할 수 있을까요?

## 1. 이미지를 인식하는 원리
MNIST 데이터는 텐서플로의 케라스 API를 이용해 간단히 불러 올 수 있습니다.  

```from tensorflow.keras.datasets import mnist```

이 때 불러온 이미지 데이터를 X로 이 이미지에 0\~9를 붙인 이름표를 y로 구분해 표시하겠습니다.  

```(X_train, y_train), (X_test, y_test) = mnist.load_data()```  

케라스의 MNIST 데이터는 총 7만개 이미 중 6만개를 학습용으로 1만 개는 테스트용으로 미리 구분해 놓고 있습니다. 이런 내용은 다음과 같이 확인할 수 있습니다.  

```
print('학습 데이터셋 이미지 수 :', X_train.shape[0])
print('테스트 데이터셋 이미지 수 :', X_test.shaep[0])
```






```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 0s 0us/step



```python
X_train.shape
```




    (60000, 28, 28)



불러온 이미지들 중 한 개만 출력해 보겠습니다.


```python
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap = 'Greys')
plt.show()
```


    
![png](output_80_0.png)
    


이 이미지를 컴퓨터는 어떻게 인식할까요?

이 이미지는 가로 28, 세로 28 픽셀 크기입니다. 28 x 28 = 784개의 픽셀로 구성되어 있습니다. 각 픽셀은 밝기 정도에 따라 0부터 255까지 등급을 매깁니다. 흰색 픽셀이의 값이 0이라면 1부터 255까지 옅은 회색에서 점점 더 어두워져 최종적으로 픽셀값이 255는 완전한 검은색이 됩니다. 따라서 위 이미지는 0~255까지의 값을 갖는 행렬로 해석할 수 있습니다. 다시 말해 집합 또는 배열로 생각할 수 있습니다.


```python
import sys

for pixel_line in X_train[0]:
  for pixel in pixel_line:
    sys.stdout.write("%-4s" % pixel)
  sys.stdout.write('\n')
```

    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   3   18  18  18  126 136 175 26  166 255 247 127 0   0   0   0   
    0   0   0   0   0   0   0   0   30  36  94  154 170 253 253 253 253 253 225 172 253 242 195 64  0   0   0   0   
    0   0   0   0   0   0   0   49  238 253 253 253 253 253 253 253 253 251 93  82  82  56  39  0   0   0   0   0   
    0   0   0   0   0   0   0   18  219 253 253 253 253 253 198 182 247 241 0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   80  156 107 253 253 205 11  0   43  154 0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   14  1   154 253 90  0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   139 253 190 2   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   11  190 253 70  0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   35  241 225 160 108 1   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   81  240 253 253 119 25  0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   45  186 253 253 150 27  0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   16  93  252 253 187 0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   249 253 249 64  0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   46  130 183 253 253 207 2   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   39  148 229 253 253 253 250 182 0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   24  114 221 253 253 253 253 201 78  0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   23  66  213 253 253 253 253 198 81  2   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   18  171 219 253 253 253 253 195 80  9   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   55  172 226 253 253 253 253 244 133 11  0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   136 253 253 253 212 135 132 16  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   


이렇게 이미지는 숫자의 집합으로 바뀌어 학습 데이터셋으로 사용됩니다. 우리가 앞서 배운 여러 예제와 마찬가지로 속성을 담은 데이터 모델에 집어 넣고 클래스를 예측하는 문제로 귀결되는 것입니다. 28x28=784개의 속성을 이용해 0\~9의 클래스 열 개중 하나를 맞추는 문제가 됩니다. 

이제 주어진 가로 28 세로 28의 2차월 배열을 1차원 배열로 바꾸어주어야 합니다. 이를 위해 ```reshape()```함수를 사용합니다. 

reshape(총 샘플 수, 1차원 속성의 개수) 형식으로 지정합니다. 총 샘플 수는 앞서 사용한 X_train.shape[0]을 이용하고 1차원 속성의 개수는 이미 살펴 본 대로 784입니다.
```
X_train=X_train.reshape(X_train.shape[0], 784)
```



케라스는 데이터를 0에서 1 사이의 값으로 변환 후 구동할 때 최적의 성능을 보입니다. 따라서 현재 0\~255 사이의 수로 표현되는 픽셀의 값을 0\~1 사이의 값으로 바꿔야 합니다. 바꾸는 방법은 각 픽셀의 값을 255로 나누면 됩니다. 이러한 과정, 0~1사이의 값으로 바꾸는 과정을 '정규화(normalization)'라고 합니다. 정규화에 앞서 데이터형을 실수로 빠꾼 후 정규화 합니다.   

```
X_train = X_train.astype('float64')
X_train = X_train / 255
```

X_test에 대해서도 위와 같은 이유에서 동일한 작업을 수행합니다.
```
X_test = X_text.reshape(X_test.shape[0], 784).astype('flat64')/255
```  
이제 숫자 이미지에 대해 붙여진 이름(클래스, 레이블)을 확인해 보겠습니다. 우리는 앞서 불러온 숫자 이미지가 5라는 것을 눈으로 보아 알 수 있었습니다. 실제로 이 숫자의 레이블이 어떤지 불어오고자 합니다.




```python
print('class(label) :', y_train[0])
```

    class(label) : 5


그런데 12장에서 아이리스 품종을 예측할 때 딥러닝의 분류 문제를 해결하려면 원-핫 인코딩 방식을 적용해야 한다고 배웠습니다. 즉, 0\~9의 정수형 값을 갖는 현재 형태에서 0 또는 1로만 이루어진 값으로 수정해야 합니다.   
지금 우리가 본 이미지의 클래스(레이블)은 5였습니다. 이를 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]으로 바꿔야 합ㄴ니다. 이를 가능하게 해주는 함수가 ```np_utils.to_categorical()```함수입닌다. ```to_categorical(클래스, 그래스의 개수)```형식으로 지정합니다.

```
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```


```python
from tensorflow.keras.utils import to_categorical

print(y_train[0])
y_train = to_categorical(y_train, 10)
print(y_train[0])
```

    [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]


지금까지 이야기한 내용과 관련된 코드를 아래에 모음


```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import sys

# MNIST 데이터셋을 불러와 학습셋과 테스트셋으로 저장합니다. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 학습셋과 테스트셋이 각각 몇 개의 이미지로 되어 있는지 확인합니다. 
print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))
```

    학습셋 이미지 수 : 60000 개
    테스트셋 이미지 수 : 10000 개



```python
# 첫 번째 이미지를 확인해 봅시다.
plt.imshow(X_train[0], cmap='Greys')
plt.show()
```


    
![png](output_90_0.png)
    



```python
# 이미지가 인식되는 원리를 알아봅시다.
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%-3s" % i)
    sys.stdout.write('\n')
```

    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  3  18 18 18 12613617526 1662552471270  0  0  0  
    0  0  0  0  0  0  0  0  30 36 94 15417025325325325325322517225324219564 0  0  0  0  
    0  0  0  0  0  0  0  49 23825325325325325325325325325193 82 82 56 39 0  0  0  0  0  
    0  0  0  0  0  0  0  18 2192532532532532531981822472410  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  80 15610725325320511 0  43 1540  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  14 1  15425390 0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  1392531902  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  11 19025370 0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  35 2412251601081  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  81 24025325311925 0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  45 18625325315027 0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  16 93 2522531870  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  24925324964 0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  46 1301832532532072  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  39 1482292532532532501820  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  24 11422125325325325320178 0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  23 66 21325325325325319881 2  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  18 17121925325325325319580 9  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  55 17222625325325325324413311 0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  13625325325321213513216 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  



```python
# 차원 변환 과정을 실습해 봅니다.
X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

# 클래스 값을 확인해 봅니다.
print("class : %d " % (y_train[0]))

# 바이너리화 과정을 실습해 봅니다.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(y_train[0])
```

    class : 5 
    [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


## 2. 딥러닝 기본 프레임 만들기 
모델 구조를 정하고 모델을 커파일 하는 단계로 넘어가겠습니다. 

총 784개의 속성이 있고 열 개의 클래스(레이블)가 있습니다. 그래서 다음과 같이 모델을 생성합니다.


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os

# MNIST 데이터를 불러옵니다. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 차원 변환 후, 테스트셋과 학습셋으로 나누어 줍니다.
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

#### [과제]
과제 - 본 실습을 위해 모델 구조를 다음과 같이 제안합니다.
* 은닉층 1개로 구성
* 은닉층의 노드 개수는 512개
* 은닉층의 활성화 함수는 relu
* 출력측의 활성화 함수는 softmax
가 되도록 코딩하십시요.


```python
# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
```

    Model: "sequential_42"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_120 (Dense)           (None, 512)               401920    
                                                                     
     dense_121 (Dense)           (None, 10)                5130      
                                                                     
    =================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    _________________________________________________________________


#### [과제] 
과제 - 모델 실행 환경을 위한 설정을 다음과 같이 제안합니다.
* 다중 분류 상황이기 때문에 손실 함수(loss function)는 categorical_crossentropy
* 옵티마이저는 adam


```python
# 모델 실행 환경을 설정합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
# 모델 최적화를 위한 설정 구간입니다.
modelpath="./MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델을 실행합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도를 출력합니다.
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

    
    Epoch 1: val_loss improved from inf to 0.19017, saving model to ./MNIST_MLP.hdf5
    
    Epoch 2: val_loss improved from 0.19017 to 0.13124, saving model to ./MNIST_MLP.hdf5
    
    Epoch 3: val_loss improved from 0.13124 to 0.11152, saving model to ./MNIST_MLP.hdf5
    
    Epoch 4: val_loss improved from 0.11152 to 0.09714, saving model to ./MNIST_MLP.hdf5
    
    Epoch 5: val_loss improved from 0.09714 to 0.08911, saving model to ./MNIST_MLP.hdf5
    
    Epoch 6: val_loss improved from 0.08911 to 0.08867, saving model to ./MNIST_MLP.hdf5
    
    Epoch 7: val_loss did not improve from 0.08867
    
    Epoch 8: val_loss improved from 0.08867 to 0.08145, saving model to ./MNIST_MLP.hdf5
    
    Epoch 9: val_loss did not improve from 0.08145
    
    Epoch 10: val_loss did not improve from 0.08145
    
    Epoch 11: val_loss improved from 0.08145 to 0.08076, saving model to ./MNIST_MLP.hdf5
    
    Epoch 12: val_loss did not improve from 0.08076
    
    Epoch 13: val_loss did not improve from 0.08076
    
    Epoch 14: val_loss did not improve from 0.08076
    
    Epoch 15: val_loss did not improve from 0.08076
    
    Epoch 16: val_loss did not improve from 0.08076
    
    Epoch 17: val_loss did not improve from 0.08076
    
    Epoch 18: val_loss did not improve from 0.08076
    
    Epoch 19: val_loss did not improve from 0.08076
    
    Epoch 20: val_loss did not improve from 0.08076
    
    Epoch 21: val_loss did not improve from 0.08076
    313/313 [==============================] - 1s 3ms/step - loss: 0.0721 - accuracy: 0.9815
    
     Test Accuracy: 0.9815


**실행 결과를 그래프로 표현**해보려고 합니다. 역시 14장에서 실습한 내용과 크게 다르지 않습니다. 다만 이번에는 학습 데이터셋의 정확도 대신 학습 데이터셋의 오차를 그래프로 표현하겠습니다. 학습 데이터셋의 오차는 1에서 학습 데이터셋에 대한 정확도를 뺀 값입니다. 좀 더 세밀한 변화를 볼 수 있도록 학습 데이터셋 오차와 테스테 데이터셋의 오차를 그래프 하나에 나타내겠습니다.


```python
# 검증(validation) 데이터셋과 학습 데이터셋의 오차를 저장합니다. 
# 실제 값이 검증 데이터셋과 예측 값 사이의 차이
# 실제 값인 학습 데이터셋과 예측 값 사이의 차이

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프로 표현해 봅니다.
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Validset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시해 보겠습니다.
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


    
![png](output_101_0.png)
    


epoch가 19(?)에서 갱신(모델업데이트 과정)을 멈췄습니다. 갱신 과정(다음 에포크)을 더 진행했다면 학습 데이터셋에 대한 오차는 줄어 들 수 있으나 검증 데이터셋의 더 이상 작아지지 않고 학습 데이터셋에 대한 과적합(overfit)으로 오히려 검증 데이터에 대한 에러가 커질 수 있습니다.  

앞서 높은 정확도를 보였던 딥러닝 모델은 하나의 은닉층을 두 아주 단순한 모델입니다. 이를 도식화 하면 그림 16-5와 같습니다.  
<br><center>
<img src="https://drive.google.com/uc?id=12E2tAvpsXXYEb__X9LYuKkS1iGZV3j-n" width=200>.................
<img src="https://drive.google.com/uc?id=11x6By8NmA3K-B_D6N5Vev1tqYKPpy7LC" width=400>
</center><br>

딥러닝은 이러한 기본 모델을 바탕으로 프로젝트에 맞추어 어떤 옵션을 더하고 어떤 층을 추가하느냐에 따라 성능이 좋아질 수 있습니다. 지금부터 기본 딥러닝 모델에 이미지 인식 분야에 강력한 성능을 보이는 컨볼류션 신경망(CNN)을 추가해 보겠습니다.

# 아래 내용은 조금 더 익숙해진 이후에 학습하는 것으로...






## 3. 컨볼루션 신경망

**컨볼루션 신경망(Convolutional Neural Network)은 입력된 이미지에서 다시 한 번 특징을 추출하기 위해 커널(슬라이딩 원도우)를 도입하는 기법입니다.** 예를 들어 입력된 이미지가 다음과 같은 값을 가지고 있다고 가정하겠습니다.
<br><center>
<img src="https://drive.google.com/uc?id=1kmkb3v9yT9zhjpNDExlEiU5KodZ9uPYg" width=200>
</center><br>

여기에 2x2 커널을 준비합니다. 각 칸에는 가중치가 들어 있습니다. 샘플 가중치를 다음과 같이 x1, x0이라고 하겠습니다. 
<br><center>
<img src="https://drive.google.com/uc?id=1MjjrQVgD3FOb0tDn4uuCnum6GGaFp8R1" width=100>
</center><br>  
이제 이 커널을 이미지 맨 왼쪽 윗칸에 적용시켜 보겠습니다. 

<br><center>
<img src="https://drive.google.com/uc?id=1RDiYx3pu4cll4uoh57ptH12bBx8lJrbC" width=200>
</center><br>  
적용된 부분은 원래 있던 이미지 값에 가중치의 값을 곱해서 각각의 값을 합산해 새로운 값을 생성합니다.  
$$(1\times1) + (0\times0) +(0\times0) + (1\times1)=2$$  

이 커널을 한 칸씩 옮겨 모두 적용해 보겠습니다. 
<br><center>
<img src="https://drive.google.com/uc?id=1ukY_jnDI-Isg0q7mW3mIf2SokQQs3mB0" width=400><br>
<img src="https://drive.google.com/uc?id=1NrxBhRRLIc3dW1ZYr2kL8zPIjIwp2uf9" width=400><br>
<img src="https://drive.google.com/uc?id=1tibqQCTOeRD7xyG66YmzVYBvAK8VMcnK" width=400>
</center><br>
위 결과를 정리하면 다음과 같습니다.

<br><center>
<img src="https://drive.google.com/uc?id=1O1cBeGghvfegnNCacKAN8IK1Vf9KL57V" width=200>
</center><br>  

이렇게 해서 새롭게 만들어진 층을 컨볼루션(합성곱) 층이라고 합니다. 컨볼루션 층을 만들려면 입력 데이터가 가진 특성을 대략적으로 추출해서 학습을 진행할 수 있습니다. 이러한 커널을 여러 개 만들 경우 여러 개의 컨볼루션 층이 만들어집니다.   

[그림](https://thebook.io/080324/part05/ch16/03-03/)

케라스에서 컨볼루션 층을 추가하는 함수는 ```Conv2D()```입니다. 다음과 같이 컨볼루션 층을 적용해 MNIST 손글씨 인식률을 높여 봅시다. 

```
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
```

```Conv2D()```에 전달되는 네 가지 인자는 다음과 같습니다. 
- 첫번재 인자: 커널을 몇 개 적용할지 결정. 여기서는 32개의 커널을 적용했음.
- kernel_size: 커널의 크기. ```kernel_size(행, 열)``` 형식으로 지정. 여기서는 3 $\times$ 3  
- input_shape: Dense 층과 마찬가지로 맨 처음 층에는 입력되는 값의 형태 등을 알려주야 함. ```input_shape=(행, 열, 색상타입)``` 형식으로 정합니다. 컬러이미지면 색상타입은 3이고 흑백이면 1입니다. 여기서는 28 $\times$28 크기의 흑백 이미지를 사용한다고 지정했음.
- activation: 사용할 활성화 함수.

이어서 컨벌루션 층을 하나 더 추가해 보겠습니다. 다음과 같이 커널 수는 64개, 커널의 크기는 3 $\times$ 3으로 지정하고 활성화 함수로 렐루를 사용하는 컨볼루션 층을 추가합니다. 

```
model.add(Conv2D(64, (3, 3), activation='relu'))
```
<br><center>
<img src="https://drive.google.com/uc?id=11x6By8NmA3K-B_D6N5Vev1tqYKPpy7LC" width=500>
</center><br>

