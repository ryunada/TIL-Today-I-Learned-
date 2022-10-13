
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
    Epoch 4/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.6395 - accuracy: 0.7968 - val_loss: 0.5542 - val_accuracy: 0.8208
    Epoch 5/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.5658 - accuracy: 0.8160 - val_loss: 0.4592 - val_accuracy: 0.8431
    Epoch 6/50
    8/8 [==============================] - 0s 13ms/step - loss: 0.4583 - accuracy: 0.8468 - val_loss: 0.3497 - val_accuracy: 0.8715
    Epoch 7/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.3409 - accuracy: 0.8794 - val_loss: 0.2490 - val_accuracy: 0.9154
    Epoch 8/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.2491 - accuracy: 0.9189 - val_loss: 0.2308 - val_accuracy: 0.9369
    Epoch 9/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.2389 - accuracy: 0.9271 - val_loss: 0.2211 - val_accuracy: 0.9362
    Epoch 10/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.2272 - accuracy: 0.9276 - val_loss: 0.2052 - val_accuracy: 0.9400
    Epoch 11/50
    8/8 [==============================] - 0s 24ms/step - loss: 0.2207 - accuracy: 0.9266 - val_loss: 0.1995 - val_accuracy: 0.9408
    Epoch 12/50
    8/8 [==============================] - 0s 15ms/step - loss: 0.2162 - accuracy: 0.9274 - val_loss: 0.1967 - val_accuracy: 0.9385
    Epoch 13/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.2121 - accuracy: 0.9271 - val_loss: 0.1944 - val_accuracy: 0.9392
    Epoch 14/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.2101 - accuracy: 0.9264 - val_loss: 0.1908 - val_accuracy: 0.9392
    Epoch 15/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.2080 - accuracy: 0.9274 - val_loss: 0.1893 - val_accuracy: 0.9408
    Epoch 16/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.2064 - accuracy: 0.9271 - val_loss: 0.1891 - val_accuracy: 0.9392
    Epoch 17/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.2050 - accuracy: 0.9294 - val_loss: 0.1871 - val_accuracy: 0.9392
    Epoch 18/50
    8/8 [==============================] - 0s 16ms/step - loss: 0.2037 - accuracy: 0.9292 - val_loss: 0.1857 - val_accuracy: 0.9392
    Epoch 19/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.2028 - accuracy: 0.9302 - val_loss: 0.1853 - val_accuracy: 0.9385
    Epoch 20/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.2013 - accuracy: 0.9302 - val_loss: 0.1836 - val_accuracy: 0.9377
    Epoch 21/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.2004 - accuracy: 0.9307 - val_loss: 0.1824 - val_accuracy: 0.9377
    Epoch 22/50
    8/8 [==============================] - 0s 10ms/step - loss: 0.1992 - accuracy: 0.9307 - val_loss: 0.1827 - val_accuracy: 0.9408
    Epoch 23/50
    8/8 [==============================] - 0s 10ms/step - loss: 0.1981 - accuracy: 0.9312 - val_loss: 0.1812 - val_accuracy: 0.9400
    Epoch 24/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.1971 - accuracy: 0.9310 - val_loss: 0.1805 - val_accuracy: 0.9408
    Epoch 25/50
    8/8 [==============================] - 0s 16ms/step - loss: 0.1962 - accuracy: 0.9320 - val_loss: 0.1799 - val_accuracy: 0.9400
    Epoch 26/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.1954 - accuracy: 0.9320 - val_loss: 0.1785 - val_accuracy: 0.9408
    Epoch 27/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.1945 - accuracy: 0.9325 - val_loss: 0.1783 - val_accuracy: 0.9392
    Epoch 28/50
    8/8 [==============================] - 0s 20ms/step - loss: 0.1937 - accuracy: 0.9320 - val_loss: 0.1775 - val_accuracy: 0.9392
    Epoch 29/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.1932 - accuracy: 0.9317 - val_loss: 0.1776 - val_accuracy: 0.9400
    Epoch 30/50
    8/8 [==============================] - 0s 9ms/step - loss: 0.1919 - accuracy: 0.9328 - val_loss: 0.1757 - val_accuracy: 0.9408
    Epoch 31/50
    8/8 [==============================] - 0s 10ms/step - loss: 0.1914 - accuracy: 0.9325 - val_loss: 0.1754 - val_accuracy: 0.9415
    Epoch 32/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.1905 - accuracy: 0.9328 - val_loss: 0.1750 - val_accuracy: 0.9415
    Epoch 33/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.1899 - accuracy: 0.9328 - val_loss: 0.1747 - val_accuracy: 0.9400
    Epoch 34/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.1894 - accuracy: 0.9317 - val_loss: 0.1744 - val_accuracy: 0.9408
    Epoch 35/50
    8/8 [==============================] - 0s 10ms/step - loss: 0.1884 - accuracy: 0.9320 - val_loss: 0.1727 - val_accuracy: 0.9415
    Epoch 36/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.1879 - accuracy: 0.9320 - val_loss: 0.1727 - val_accuracy: 0.9408
    Epoch 37/50
    8/8 [==============================] - 0s 12ms/step - loss: 0.1872 - accuracy: 0.9323 - val_loss: 0.1725 - val_accuracy: 0.9408
    Epoch 38/50
    8/8 [==============================] - 0s 13ms/step - loss: 0.1864 - accuracy: 0.9320 - val_loss: 0.1708 - val_accuracy: 0.9423
    Epoch 39/50
    8/8 [==============================] - 0s 10ms/step - loss: 0.1857 - accuracy: 0.9315 - val_loss: 0.1710 - val_accuracy: 0.9415
    Epoch 40/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.1851 - accuracy: 0.9328 - val_loss: 0.1708 - val_accuracy: 0.9415
    Epoch 41/50
    8/8 [==============================] - 0s 18ms/step - loss: 0.1844 - accuracy: 0.9323 - val_loss: 0.1694 - val_accuracy: 0.9415
    Epoch 42/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.1837 - accuracy: 0.9325 - val_loss: 0.1691 - val_accuracy: 0.9423
    Epoch 43/50
    8/8 [==============================] - 0s 19ms/step - loss: 0.1831 - accuracy: 0.9330 - val_loss: 0.1692 - val_accuracy: 0.9431
    Epoch 44/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.1821 - accuracy: 0.9338 - val_loss: 0.1673 - val_accuracy: 0.9415
    Epoch 45/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.1817 - accuracy: 0.9338 - val_loss: 0.1675 - val_accuracy: 0.9431
    Epoch 46/50
    8/8 [==============================] - 0s 11ms/step - loss: 0.1809 - accuracy: 0.9338 - val_loss: 0.1665 - val_accuracy: 0.9431
    Epoch 47/50
    8/8 [==============================] - 0s 14ms/step - loss: 0.1801 - accuracy: 0.9335 - val_loss: 0.1664 - val_accuracy: 0.9438
    Epoch 48/50
    8/8 [==============================] - 0s 27ms/step - loss: 0.1796 - accuracy: 0.9341 - val_loss: 0.1655 - val_accuracy: 0.9431
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
    Epoch 4/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0995 - accuracy: 0.9646
    Epoch 5/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0978 - accuracy: 0.9673
    Epoch 6/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0973 - accuracy: 0.9663
    Epoch 7/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0953 - accuracy: 0.9684
    Epoch 8/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0945 - accuracy: 0.9690
    Epoch 9/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0922 - accuracy: 0.9713
    Epoch 10/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0902 - accuracy: 0.9717
    Epoch 11/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0884 - accuracy: 0.9744
    Epoch 12/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0886 - accuracy: 0.9731
    Epoch 13/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0869 - accuracy: 0.9729
    Epoch 14/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0844 - accuracy: 0.9736
    Epoch 15/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0826 - accuracy: 0.9740
    Epoch 16/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0851 - accuracy: 0.9738
    Epoch 17/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0866 - accuracy: 0.9734
    Epoch 18/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0849 - accuracy: 0.9723
    Epoch 19/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0819 - accuracy: 0.9754
    Epoch 20/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0811 - accuracy: 0.9746
    Epoch 21/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0785 - accuracy: 0.9758
    Epoch 22/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0777 - accuracy: 0.9767
    Epoch 23/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0771 - accuracy: 0.9765
    Epoch 24/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0754 - accuracy: 0.9786
    Epoch 25/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0750 - accuracy: 0.9769
    Epoch 26/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0744 - accuracy: 0.9771
    Epoch 27/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0728 - accuracy: 0.9786
    Epoch 28/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0743 - accuracy: 0.9769
    Epoch 29/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0738 - accuracy: 0.9777
    Epoch 30/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0730 - accuracy: 0.9771
    Epoch 31/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0711 - accuracy: 0.9786
    Epoch 32/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0707 - accuracy: 0.9792
    Epoch 33/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0693 - accuracy: 0.9788
    Epoch 34/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0687 - accuracy: 0.9796
    Epoch 35/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0690 - accuracy: 0.9790
    Epoch 36/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0683 - accuracy: 0.9806
    Epoch 37/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0704 - accuracy: 0.9769
    Epoch 38/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0669 - accuracy: 0.9792
    Epoch 39/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0672 - accuracy: 0.9796
    Epoch 40/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0666 - accuracy: 0.9796
    Epoch 41/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0666 - accuracy: 0.9800
    Epoch 42/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0656 - accuracy: 0.9794
    Epoch 43/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0650 - accuracy: 0.9813
    Epoch 44/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0646 - accuracy: 0.9804
    Epoch 45/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0641 - accuracy: 0.9811
    Epoch 46/50
    11/11 [==============================] - 0s 3ms/step - loss: 0.0640 - accuracy: 0.9811
    Epoch 47/50
    11/11 [==============================] - 0s 2ms/step - loss: 0.0649 - accuracy: 0.9802
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
    
    Epoch 4: saving model to ./data/model/all/04-0.9477.hdf5
    
    Epoch 5: saving model to ./data/model/all/05-0.9469.hdf5
    
    Epoch 6: saving model to ./data/model/all/06-0.9492.hdf5
    
    Epoch 7: saving model to ./data/model/all/07-0.9492.hdf5
    
    Epoch 8: saving model to ./data/model/all/08-0.9515.hdf5
    
    Epoch 9: saving model to ./data/model/all/09-0.9508.hdf5
    
    Epoch 10: saving model to ./data/model/all/10-0.9508.hdf5
    
    Epoch 11: saving model to ./data/model/all/11-0.9523.hdf5
    
    Epoch 12: saving model to ./data/model/all/12-0.9523.hdf5
    
    Epoch 13: saving model to ./data/model/all/13-0.9523.hdf5
    
    Epoch 14: saving model to ./data/model/all/14-0.9538.hdf5
    
    Epoch 15: saving model to ./data/model/all/15-0.9523.hdf5
    
    Epoch 16: saving model to ./data/model/all/16-0.9554.hdf5
    
    Epoch 17: saving model to ./data/model/all/17-0.9508.hdf5
    
    Epoch 18: saving model to ./data/model/all/18-0.9500.hdf5
    
    Epoch 19: saving model to ./data/model/all/19-0.9569.hdf5
    
    Epoch 20: saving model to ./data/model/all/20-0.9562.hdf5
    
    Epoch 21: saving model to ./data/model/all/21-0.9569.hdf5
    
    Epoch 22: saving model to ./data/model/all/22-0.9569.hdf5
    
    Epoch 23: saving model to ./data/model/all/23-0.9508.hdf5
    
    Epoch 24: saving model to ./data/model/all/24-0.9592.hdf5
    
    Epoch 25: saving model to ./data/model/all/25-0.9592.hdf5
    
    Epoch 26: saving model to ./data/model/all/26-0.9600.hdf5
    
    Epoch 27: saving model to ./data/model/all/27-0.9538.hdf5
    
    Epoch 28: saving model to ./data/model/all/28-0.9623.hdf5
    
    Epoch 29: saving model to ./data/model/all/29-0.9615.hdf5
    
    Epoch 30: saving model to ./data/model/all/30-0.9623.hdf5
    
    Epoch 31: saving model to ./data/model/all/31-0.9562.hdf5
    
    Epoch 32: saving model to ./data/model/all/32-0.9546.hdf5
    
    Epoch 33: saving model to ./data/model/all/33-0.9638.hdf5
    
    Epoch 34: saving model to ./data/model/all/34-0.9615.hdf5
    
    Epoch 35: saving model to ./data/model/all/35-0.9546.hdf5
    
    Epoch 36: saving model to ./data/model/all/36-0.9615.hdf5
    
    Epoch 37: saving model to ./data/model/all/37-0.9646.hdf5
    
    Epoch 38: saving model to ./data/model/all/38-0.9615.hdf5
    
    Epoch 39: saving model to ./data/model/all/39-0.9638.hdf5
    
    Epoch 40: saving model to ./data/model/all/40-0.9669.hdf5
    
    Epoch 41: saving model to ./data/model/all/41-0.9638.hdf5
    
    Epoch 42: saving model to ./data/model/all/42-0.9631.hdf5
    
    Epoch 43: saving model to ./data/model/all/43-0.9669.hdf5
    
    Epoch 44: saving model to ./data/model/all/44-0.9623.hdf5
    
    Epoch 45: saving model to ./data/model/all/45-0.9638.hdf5
    
    Epoch 46: saving model to ./data/model/all/46-0.9669.hdf5
    
    Epoch 47: saving model to ./data/model/all/47-0.9669.hdf5
    
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
    8/8 [==============================] - 0s 6ms/step - loss: 0.1004 - accuracy: 0.9690 - val_loss: 0.0963 - val_accuracy: 0.9685
    Epoch 4/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1016 - accuracy: 0.9684 - val_loss: 0.0968 - val_accuracy: 0.9708
    Epoch 5/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.1006 - accuracy: 0.9687 - val_loss: 0.0947 - val_accuracy: 0.9685
    Epoch 6/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0992 - accuracy: 0.9715 - val_loss: 0.0972 - val_accuracy: 0.9662
    Epoch 7/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0987 - accuracy: 0.9700 - val_loss: 0.0935 - val_accuracy: 0.9692
    Epoch 8/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0977 - accuracy: 0.9684 - val_loss: 0.0939 - val_accuracy: 0.9715
    Epoch 9/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0979 - accuracy: 0.9684 - val_loss: 0.0925 - val_accuracy: 0.9692
    Epoch 10/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0962 - accuracy: 0.9725 - val_loss: 0.0951 - val_accuracy: 0.9677
    Epoch 11/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0958 - accuracy: 0.9733 - val_loss: 0.0927 - val_accuracy: 0.9692
    Epoch 12/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0952 - accuracy: 0.9695 - val_loss: 0.0978 - val_accuracy: 0.9723
    Epoch 13/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1008 - accuracy: 0.9695 - val_loss: 0.0915 - val_accuracy: 0.9685
    Epoch 14/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0962 - accuracy: 0.9707 - val_loss: 0.0982 - val_accuracy: 0.9646
    Epoch 15/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0962 - accuracy: 0.9713 - val_loss: 0.0893 - val_accuracy: 0.9700
    Epoch 16/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0929 - accuracy: 0.9713 - val_loss: 0.0893 - val_accuracy: 0.9731
    Epoch 17/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0919 - accuracy: 0.9733 - val_loss: 0.0878 - val_accuracy: 0.9731
    Epoch 18/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0902 - accuracy: 0.9754 - val_loss: 0.0895 - val_accuracy: 0.9685
    Epoch 19/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0921 - accuracy: 0.9710 - val_loss: 0.0865 - val_accuracy: 0.9738
    Epoch 20/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0909 - accuracy: 0.9736 - val_loss: 0.0869 - val_accuracy: 0.9746
    Epoch 21/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0887 - accuracy: 0.9746 - val_loss: 0.0862 - val_accuracy: 0.9700
    Epoch 22/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0882 - accuracy: 0.9751 - val_loss: 0.0863 - val_accuracy: 0.9708
    Epoch 23/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0870 - accuracy: 0.9751 - val_loss: 0.0844 - val_accuracy: 0.9738
    Epoch 24/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0862 - accuracy: 0.9754 - val_loss: 0.0840 - val_accuracy: 0.9738
    Epoch 25/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0860 - accuracy: 0.9759 - val_loss: 0.0840 - val_accuracy: 0.9715
    Epoch 26/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0848 - accuracy: 0.9761 - val_loss: 0.0832 - val_accuracy: 0.9754
    Epoch 27/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0866 - accuracy: 0.9738 - val_loss: 0.0824 - val_accuracy: 0.9738
    Epoch 28/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0845 - accuracy: 0.9777 - val_loss: 0.0852 - val_accuracy: 0.9715
    Epoch 29/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0838 - accuracy: 0.9761 - val_loss: 0.0819 - val_accuracy: 0.9762
    Epoch 30/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0836 - accuracy: 0.9764 - val_loss: 0.0813 - val_accuracy: 0.9746
    Epoch 31/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0822 - accuracy: 0.9769 - val_loss: 0.0807 - val_accuracy: 0.9754
    Epoch 32/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0830 - accuracy: 0.9769 - val_loss: 0.0862 - val_accuracy: 0.9723
    Epoch 33/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0827 - accuracy: 0.9764 - val_loss: 0.0796 - val_accuracy: 0.9754
    Epoch 34/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0812 - accuracy: 0.9777 - val_loss: 0.0798 - val_accuracy: 0.9754
    Epoch 35/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0800 - accuracy: 0.9766 - val_loss: 0.0801 - val_accuracy: 0.9769
    Epoch 36/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0808 - accuracy: 0.9772 - val_loss: 0.0779 - val_accuracy: 0.9754
    Epoch 37/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0798 - accuracy: 0.9759 - val_loss: 0.0804 - val_accuracy: 0.9738
    Epoch 38/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0795 - accuracy: 0.9772 - val_loss: 0.0771 - val_accuracy: 0.9754
    Epoch 39/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0779 - accuracy: 0.9777 - val_loss: 0.0764 - val_accuracy: 0.9777
    Epoch 40/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0776 - accuracy: 0.9782 - val_loss: 0.0758 - val_accuracy: 0.9777
    Epoch 41/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0774 - accuracy: 0.9782 - val_loss: 0.0759 - val_accuracy: 0.9785
    Epoch 42/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0770 - accuracy: 0.9779 - val_loss: 0.0750 - val_accuracy: 0.9777
    Epoch 43/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0762 - accuracy: 0.9787 - val_loss: 0.0755 - val_accuracy: 0.9785
    Epoch 44/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0757 - accuracy: 0.9782 - val_loss: 0.0749 - val_accuracy: 0.9777
    Epoch 45/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0747 - accuracy: 0.9784 - val_loss: 0.0759 - val_accuracy: 0.9792
    Epoch 46/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0765 - accuracy: 0.9769 - val_loss: 0.0732 - val_accuracy: 0.9785
    Epoch 47/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0747 - accuracy: 0.9790 - val_loss: 0.0728 - val_accuracy: 0.9785
    Epoch 48/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0730 - accuracy: 0.9790 - val_loss: 0.0723 - val_accuracy: 0.9792
    Epoch 49/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0728 - accuracy: 0.9792 - val_loss: 0.0719 - val_accuracy: 0.9785
    Epoch 50/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0724 - accuracy: 0.9797 - val_loss: 0.0738 - val_accuracy: 0.9785
    Epoch 51/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0733 - accuracy: 0.9790 - val_loss: 0.0717 - val_accuracy: 0.9800
    Epoch 52/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0722 - accuracy: 0.9784 - val_loss: 0.0716 - val_accuracy: 0.9800
    Epoch 53/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0712 - accuracy: 0.9795 - val_loss: 0.0722 - val_accuracy: 0.9800
    Epoch 54/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0718 - accuracy: 0.9800 - val_loss: 0.0732 - val_accuracy: 0.9785
    Epoch 55/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0708 - accuracy: 0.9795 - val_loss: 0.0711 - val_accuracy: 0.9808
    Epoch 56/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0698 - accuracy: 0.9797 - val_loss: 0.0697 - val_accuracy: 0.9808
    Epoch 57/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0690 - accuracy: 0.9797 - val_loss: 0.0700 - val_accuracy: 0.9823
    Epoch 58/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0691 - accuracy: 0.9787 - val_loss: 0.0722 - val_accuracy: 0.9792
    Epoch 59/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0703 - accuracy: 0.9787 - val_loss: 0.0714 - val_accuracy: 0.9815
    Epoch 60/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0698 - accuracy: 0.9795 - val_loss: 0.0689 - val_accuracy: 0.9823
    Epoch 61/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0676 - accuracy: 0.9810 - val_loss: 0.0689 - val_accuracy: 0.9808
    Epoch 62/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0691 - accuracy: 0.9805 - val_loss: 0.0675 - val_accuracy: 0.9808
    Epoch 63/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0682 - accuracy: 0.9813 - val_loss: 0.0686 - val_accuracy: 0.9815
    Epoch 64/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0705 - accuracy: 0.9800 - val_loss: 0.0687 - val_accuracy: 0.9815
    Epoch 65/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0677 - accuracy: 0.9813 - val_loss: 0.0667 - val_accuracy: 0.9815
    Epoch 66/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0659 - accuracy: 0.9805 - val_loss: 0.0665 - val_accuracy: 0.9831
    Epoch 67/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0655 - accuracy: 0.9810 - val_loss: 0.0661 - val_accuracy: 0.9815
    Epoch 68/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0659 - accuracy: 0.9815 - val_loss: 0.0722 - val_accuracy: 0.9815
    Epoch 69/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0671 - accuracy: 0.9823 - val_loss: 0.0700 - val_accuracy: 0.9831
    Epoch 70/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0668 - accuracy: 0.9818 - val_loss: 0.0665 - val_accuracy: 0.9831
    Epoch 71/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0647 - accuracy: 0.9833 - val_loss: 0.0704 - val_accuracy: 0.9838
    Epoch 72/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0658 - accuracy: 0.9820 - val_loss: 0.0655 - val_accuracy: 0.9831
    Epoch 73/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0643 - accuracy: 0.9818 - val_loss: 0.0649 - val_accuracy: 0.9823
    Epoch 74/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0641 - accuracy: 0.9818 - val_loss: 0.0648 - val_accuracy: 0.9815
    Epoch 75/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0641 - accuracy: 0.9820 - val_loss: 0.0644 - val_accuracy: 0.9815
    Epoch 76/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0671 - accuracy: 0.9810 - val_loss: 0.0638 - val_accuracy: 0.9815
    Epoch 77/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0668 - accuracy: 0.9831 - val_loss: 0.0834 - val_accuracy: 0.9731
    Epoch 78/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0697 - accuracy: 0.9823 - val_loss: 0.0728 - val_accuracy: 0.9808
    Epoch 79/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0639 - accuracy: 0.9820 - val_loss: 0.0700 - val_accuracy: 0.9838
    Epoch 80/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0656 - accuracy: 0.9810 - val_loss: 0.0649 - val_accuracy: 0.9846
    Epoch 81/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0661 - accuracy: 0.9815 - val_loss: 0.0628 - val_accuracy: 0.9823
    Epoch 82/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0656 - accuracy: 0.9823 - val_loss: 0.0626 - val_accuracy: 0.9823
    Epoch 83/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0661 - accuracy: 0.9831 - val_loss: 0.0688 - val_accuracy: 0.9831
    Epoch 84/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0625 - accuracy: 0.9838 - val_loss: 0.0650 - val_accuracy: 0.9838
    Epoch 85/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0618 - accuracy: 0.9828 - val_loss: 0.0625 - val_accuracy: 0.9823
    Epoch 86/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0627 - accuracy: 0.9828 - val_loss: 0.0681 - val_accuracy: 0.9838
    Epoch 87/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0658 - accuracy: 0.9808 - val_loss: 0.0633 - val_accuracy: 0.9854
    Epoch 88/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0650 - accuracy: 0.9813 - val_loss: 0.0711 - val_accuracy: 0.9800
    Epoch 89/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0638 - accuracy: 0.9831 - val_loss: 0.0683 - val_accuracy: 0.9823
    Epoch 90/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0617 - accuracy: 0.9833 - val_loss: 0.0612 - val_accuracy: 0.9831
    Epoch 91/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0592 - accuracy: 0.9838 - val_loss: 0.0625 - val_accuracy: 0.9846
    Epoch 92/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0589 - accuracy: 0.9841 - val_loss: 0.0611 - val_accuracy: 0.9831
    Epoch 93/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0595 - accuracy: 0.9831 - val_loss: 0.0608 - val_accuracy: 0.9831
    Epoch 94/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0596 - accuracy: 0.9831 - val_loss: 0.0642 - val_accuracy: 0.9838
    Epoch 95/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0595 - accuracy: 0.9838 - val_loss: 0.0621 - val_accuracy: 0.9846
    Epoch 96/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0593 - accuracy: 0.9846 - val_loss: 0.0606 - val_accuracy: 0.9854
    Epoch 97/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0588 - accuracy: 0.9841 - val_loss: 0.0594 - val_accuracy: 0.9838
    Epoch 98/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0585 - accuracy: 0.9836 - val_loss: 0.0604 - val_accuracy: 0.9838
    Epoch 99/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0583 - accuracy: 0.9846 - val_loss: 0.0606 - val_accuracy: 0.9838
    Epoch 100/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0575 - accuracy: 0.9849 - val_loss: 0.0597 - val_accuracy: 0.9854
    Epoch 101/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0579 - accuracy: 0.9841 - val_loss: 0.0587 - val_accuracy: 0.9862
    Epoch 102/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0593 - accuracy: 0.9838 - val_loss: 0.0583 - val_accuracy: 0.9854
    Epoch 103/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0576 - accuracy: 0.9851 - val_loss: 0.0591 - val_accuracy: 0.9854
    Epoch 104/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0565 - accuracy: 0.9856 - val_loss: 0.0589 - val_accuracy: 0.9854
    Epoch 105/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0559 - accuracy: 0.9859 - val_loss: 0.0589 - val_accuracy: 0.9869
    Epoch 106/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0583 - accuracy: 0.9838 - val_loss: 0.0586 - val_accuracy: 0.9869
    Epoch 107/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0559 - accuracy: 0.9854 - val_loss: 0.0584 - val_accuracy: 0.9869
    Epoch 108/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0585 - accuracy: 0.9833 - val_loss: 0.0573 - val_accuracy: 0.9869
    Epoch 109/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0581 - accuracy: 0.9851 - val_loss: 0.0638 - val_accuracy: 0.9831
    Epoch 110/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0621 - accuracy: 0.9841 - val_loss: 0.0722 - val_accuracy: 0.9785
    Epoch 111/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0614 - accuracy: 0.9838 - val_loss: 0.0699 - val_accuracy: 0.9808
    Epoch 112/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0620 - accuracy: 0.9851 - val_loss: 0.0597 - val_accuracy: 0.9862
    Epoch 113/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0582 - accuracy: 0.9838 - val_loss: 0.0563 - val_accuracy: 0.9846
    Epoch 114/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0553 - accuracy: 0.9864 - val_loss: 0.0583 - val_accuracy: 0.9854
    Epoch 115/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0542 - accuracy: 0.9864 - val_loss: 0.0558 - val_accuracy: 0.9854
    Epoch 116/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0541 - accuracy: 0.9869 - val_loss: 0.0561 - val_accuracy: 0.9846
    Epoch 117/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0555 - accuracy: 0.9849 - val_loss: 0.0637 - val_accuracy: 0.9831
    Epoch 118/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0540 - accuracy: 0.9859 - val_loss: 0.0556 - val_accuracy: 0.9854
    Epoch 119/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0536 - accuracy: 0.9864 - val_loss: 0.0575 - val_accuracy: 0.9862
    Epoch 120/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0533 - accuracy: 0.9859 - val_loss: 0.0556 - val_accuracy: 0.9854
    Epoch 121/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0533 - accuracy: 0.9861 - val_loss: 0.0560 - val_accuracy: 0.9854
    Epoch 122/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0537 - accuracy: 0.9872 - val_loss: 0.0557 - val_accuracy: 0.9862
    Epoch 123/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0527 - accuracy: 0.9872 - val_loss: 0.0555 - val_accuracy: 0.9854
    Epoch 124/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0520 - accuracy: 0.9872 - val_loss: 0.0552 - val_accuracy: 0.9854
    Epoch 125/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0518 - accuracy: 0.9864 - val_loss: 0.0550 - val_accuracy: 0.9846
    Epoch 126/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0531 - accuracy: 0.9877 - val_loss: 0.0617 - val_accuracy: 0.9854
    Epoch 127/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0541 - accuracy: 0.9869 - val_loss: 0.0568 - val_accuracy: 0.9869
    Epoch 128/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0517 - accuracy: 0.9874 - val_loss: 0.0553 - val_accuracy: 0.9862
    Epoch 129/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0551 - accuracy: 0.9849 - val_loss: 0.0547 - val_accuracy: 0.9838
    Epoch 130/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0518 - accuracy: 0.9879 - val_loss: 0.0546 - val_accuracy: 0.9846
    Epoch 131/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0509 - accuracy: 0.9874 - val_loss: 0.0548 - val_accuracy: 0.9854
    Epoch 132/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0516 - accuracy: 0.9867 - val_loss: 0.0546 - val_accuracy: 0.9854
    Epoch 133/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0518 - accuracy: 0.9872 - val_loss: 0.0572 - val_accuracy: 0.9854
    Epoch 134/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0541 - accuracy: 0.9872 - val_loss: 0.0585 - val_accuracy: 0.9846
    Epoch 135/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0536 - accuracy: 0.9849 - val_loss: 0.0639 - val_accuracy: 0.9831
    Epoch 136/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0540 - accuracy: 0.9854 - val_loss: 0.0569 - val_accuracy: 0.9846
    Epoch 137/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0547 - accuracy: 0.9867 - val_loss: 0.0544 - val_accuracy: 0.9854
    Epoch 138/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0516 - accuracy: 0.9867 - val_loss: 0.0557 - val_accuracy: 0.9862
    Epoch 139/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0509 - accuracy: 0.9872 - val_loss: 0.0540 - val_accuracy: 0.9854
    Epoch 140/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0497 - accuracy: 0.9872 - val_loss: 0.0548 - val_accuracy: 0.9869
    Epoch 141/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0498 - accuracy: 0.9877 - val_loss: 0.0542 - val_accuracy: 0.9854
    Epoch 142/2000
    8/8 [==============================] - 0s 22ms/step - loss: 0.0499 - accuracy: 0.9872 - val_loss: 0.0541 - val_accuracy: 0.9854
    Epoch 143/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0494 - accuracy: 0.9872 - val_loss: 0.0536 - val_accuracy: 0.9846
    Epoch 144/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0527 - accuracy: 0.9856 - val_loss: 0.0593 - val_accuracy: 0.9854
    Epoch 145/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0493 - accuracy: 0.9872 - val_loss: 0.0535 - val_accuracy: 0.9838
    Epoch 146/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0512 - accuracy: 0.9872 - val_loss: 0.0535 - val_accuracy: 0.9846
    Epoch 147/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0499 - accuracy: 0.9869 - val_loss: 0.0539 - val_accuracy: 0.9877
    Epoch 148/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0490 - accuracy: 0.9869 - val_loss: 0.0540 - val_accuracy: 0.9869
    Epoch 149/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0492 - accuracy: 0.9872 - val_loss: 0.0532 - val_accuracy: 0.9846
    Epoch 150/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0488 - accuracy: 0.9882 - val_loss: 0.0545 - val_accuracy: 0.9862
    Epoch 151/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0492 - accuracy: 0.9869 - val_loss: 0.0536 - val_accuracy: 0.9869
    Epoch 152/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0499 - accuracy: 0.9864 - val_loss: 0.0547 - val_accuracy: 0.9869
    Epoch 153/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0487 - accuracy: 0.9887 - val_loss: 0.0530 - val_accuracy: 0.9831
    Epoch 154/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0477 - accuracy: 0.9877 - val_loss: 0.0534 - val_accuracy: 0.9869
    Epoch 155/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0491 - accuracy: 0.9864 - val_loss: 0.0558 - val_accuracy: 0.9862
    Epoch 156/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0492 - accuracy: 0.9869 - val_loss: 0.0523 - val_accuracy: 0.9854
    Epoch 157/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0479 - accuracy: 0.9874 - val_loss: 0.0551 - val_accuracy: 0.9869
    Epoch 158/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0500 - accuracy: 0.9864 - val_loss: 0.0557 - val_accuracy: 0.9862
    Epoch 159/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0517 - accuracy: 0.9872 - val_loss: 0.0560 - val_accuracy: 0.9862
    Epoch 160/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0498 - accuracy: 0.9864 - val_loss: 0.0533 - val_accuracy: 0.9877
    Epoch 161/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0477 - accuracy: 0.9869 - val_loss: 0.0521 - val_accuracy: 0.9862
    Epoch 162/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0483 - accuracy: 0.9874 - val_loss: 0.0525 - val_accuracy: 0.9846
    Epoch 163/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0472 - accuracy: 0.9874 - val_loss: 0.0518 - val_accuracy: 0.9846
    Epoch 164/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0465 - accuracy: 0.9887 - val_loss: 0.0521 - val_accuracy: 0.9862
    Epoch 165/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0490 - accuracy: 0.9864 - val_loss: 0.0537 - val_accuracy: 0.9869
    Epoch 166/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0484 - accuracy: 0.9872 - val_loss: 0.0576 - val_accuracy: 0.9838
    Epoch 167/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0514 - accuracy: 0.9872 - val_loss: 0.0542 - val_accuracy: 0.9869
    Epoch 168/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0488 - accuracy: 0.9879 - val_loss: 0.0521 - val_accuracy: 0.9869
    Epoch 169/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0483 - accuracy: 0.9872 - val_loss: 0.0546 - val_accuracy: 0.9838
    Epoch 170/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0463 - accuracy: 0.9872 - val_loss: 0.0517 - val_accuracy: 0.9869
    Epoch 171/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0480 - accuracy: 0.9874 - val_loss: 0.0605 - val_accuracy: 0.9846
    Epoch 172/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0471 - accuracy: 0.9879 - val_loss: 0.0518 - val_accuracy: 0.9846
    Epoch 173/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0462 - accuracy: 0.9877 - val_loss: 0.0515 - val_accuracy: 0.9869
    Epoch 174/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0466 - accuracy: 0.9879 - val_loss: 0.0519 - val_accuracy: 0.9846
    Epoch 175/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0459 - accuracy: 0.9861 - val_loss: 0.0636 - val_accuracy: 0.9823
    Epoch 176/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0511 - accuracy: 0.9874 - val_loss: 0.0602 - val_accuracy: 0.9846
    Epoch 177/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0513 - accuracy: 0.9861 - val_loss: 0.0573 - val_accuracy: 0.9846
    Epoch 178/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0474 - accuracy: 0.9877 - val_loss: 0.0533 - val_accuracy: 0.9869
    Epoch 179/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0478 - accuracy: 0.9874 - val_loss: 0.0509 - val_accuracy: 0.9877
    Epoch 180/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0459 - accuracy: 0.9877 - val_loss: 0.0509 - val_accuracy: 0.9877
    Epoch 181/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0457 - accuracy: 0.9885 - val_loss: 0.0510 - val_accuracy: 0.9869
    Epoch 182/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0451 - accuracy: 0.9890 - val_loss: 0.0509 - val_accuracy: 0.9862
    Epoch 183/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0457 - accuracy: 0.9887 - val_loss: 0.0512 - val_accuracy: 0.9846
    Epoch 184/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0445 - accuracy: 0.9879 - val_loss: 0.0528 - val_accuracy: 0.9846
    Epoch 185/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0490 - accuracy: 0.9869 - val_loss: 0.0578 - val_accuracy: 0.9846
    Epoch 186/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0565 - accuracy: 0.9854 - val_loss: 0.0539 - val_accuracy: 0.9838
    Epoch 187/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0541 - accuracy: 0.9854 - val_loss: 0.0511 - val_accuracy: 0.9877
    Epoch 188/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0474 - accuracy: 0.9869 - val_loss: 0.0543 - val_accuracy: 0.9854
    Epoch 189/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0471 - accuracy: 0.9864 - val_loss: 0.0525 - val_accuracy: 0.9869
    Epoch 190/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0486 - accuracy: 0.9869 - val_loss: 0.0520 - val_accuracy: 0.9846
    Epoch 191/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0456 - accuracy: 0.9887 - val_loss: 0.0521 - val_accuracy: 0.9854
    Epoch 192/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0441 - accuracy: 0.9885 - val_loss: 0.0508 - val_accuracy: 0.9854
    Epoch 193/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0440 - accuracy: 0.9877 - val_loss: 0.0503 - val_accuracy: 0.9862
    Epoch 194/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0450 - accuracy: 0.9897 - val_loss: 0.0511 - val_accuracy: 0.9846
    Epoch 195/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0464 - accuracy: 0.9877 - val_loss: 0.0502 - val_accuracy: 0.9862
    Epoch 196/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0449 - accuracy: 0.9885 - val_loss: 0.0513 - val_accuracy: 0.9877
    Epoch 197/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0427 - accuracy: 0.9905 - val_loss: 0.0551 - val_accuracy: 0.9838
    Epoch 198/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0464 - accuracy: 0.9874 - val_loss: 0.0522 - val_accuracy: 0.9838
    Epoch 199/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0444 - accuracy: 0.9877 - val_loss: 0.0499 - val_accuracy: 0.9869
    Epoch 200/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0437 - accuracy: 0.9887 - val_loss: 0.0538 - val_accuracy: 0.9862
    Epoch 201/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0466 - accuracy: 0.9867 - val_loss: 0.0563 - val_accuracy: 0.9854
    Epoch 202/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0481 - accuracy: 0.9859 - val_loss: 0.0516 - val_accuracy: 0.9869
    Epoch 203/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0488 - accuracy: 0.9867 - val_loss: 0.0524 - val_accuracy: 0.9846
    Epoch 204/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0435 - accuracy: 0.9895 - val_loss: 0.0614 - val_accuracy: 0.9831
    Epoch 205/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0522 - accuracy: 0.9838 - val_loss: 0.0612 - val_accuracy: 0.9831
    Epoch 206/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0538 - accuracy: 0.9854 - val_loss: 0.0500 - val_accuracy: 0.9877
    Epoch 207/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0502 - accuracy: 0.9849 - val_loss: 0.0584 - val_accuracy: 0.9831
    Epoch 208/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0479 - accuracy: 0.9872 - val_loss: 0.0552 - val_accuracy: 0.9862
    Epoch 209/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0511 - accuracy: 0.9859 - val_loss: 0.0492 - val_accuracy: 0.9869
    Epoch 210/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0454 - accuracy: 0.9882 - val_loss: 0.0509 - val_accuracy: 0.9846
    Epoch 211/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0468 - accuracy: 0.9877 - val_loss: 0.0554 - val_accuracy: 0.9854
    Epoch 212/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0443 - accuracy: 0.9882 - val_loss: 0.0501 - val_accuracy: 0.9854
    Epoch 213/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0446 - accuracy: 0.9882 - val_loss: 0.0485 - val_accuracy: 0.9877
    Epoch 214/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0429 - accuracy: 0.9890 - val_loss: 0.0489 - val_accuracy: 0.9877
    Epoch 215/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0427 - accuracy: 0.9885 - val_loss: 0.0576 - val_accuracy: 0.9846
    Epoch 216/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0433 - accuracy: 0.9874 - val_loss: 0.0502 - val_accuracy: 0.9869
    Epoch 217/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0444 - accuracy: 0.9882 - val_loss: 0.0493 - val_accuracy: 0.9869
    Epoch 218/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0446 - accuracy: 0.9887 - val_loss: 0.0498 - val_accuracy: 0.9877
    Epoch 219/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0447 - accuracy: 0.9872 - val_loss: 0.0516 - val_accuracy: 0.9854
    Epoch 220/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0448 - accuracy: 0.9874 - val_loss: 0.0515 - val_accuracy: 0.9854
    Epoch 221/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0468 - accuracy: 0.9859 - val_loss: 0.0495 - val_accuracy: 0.9869
    Epoch 222/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0427 - accuracy: 0.9874 - val_loss: 0.0505 - val_accuracy: 0.9854
    Epoch 223/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0431 - accuracy: 0.9885 - val_loss: 0.0494 - val_accuracy: 0.9869
    Epoch 224/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0417 - accuracy: 0.9897 - val_loss: 0.0494 - val_accuracy: 0.9862
    Epoch 225/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0432 - accuracy: 0.9882 - val_loss: 0.0493 - val_accuracy: 0.9854
    Epoch 226/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0450 - accuracy: 0.9877 - val_loss: 0.0490 - val_accuracy: 0.9877
    Epoch 227/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0421 - accuracy: 0.9885 - val_loss: 0.0490 - val_accuracy: 0.9885
    Epoch 228/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0419 - accuracy: 0.9882 - val_loss: 0.0487 - val_accuracy: 0.9877
    Epoch 229/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0432 - accuracy: 0.9885 - val_loss: 0.0491 - val_accuracy: 0.9885
    Epoch 230/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0428 - accuracy: 0.9879 - val_loss: 0.0501 - val_accuracy: 0.9869
    Epoch 231/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0436 - accuracy: 0.9872 - val_loss: 0.0518 - val_accuracy: 0.9869
    Epoch 232/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0455 - accuracy: 0.9861 - val_loss: 0.0490 - val_accuracy: 0.9869
    Epoch 233/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0444 - accuracy: 0.9885 - val_loss: 0.0529 - val_accuracy: 0.9838
    Epoch 234/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0426 - accuracy: 0.9885 - val_loss: 0.0517 - val_accuracy: 0.9846
    Epoch 235/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0448 - accuracy: 0.9885 - val_loss: 0.0503 - val_accuracy: 0.9854
    Epoch 236/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0469 - accuracy: 0.9879 - val_loss: 0.0488 - val_accuracy: 0.9877
    Epoch 237/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0466 - accuracy: 0.9864 - val_loss: 0.0517 - val_accuracy: 0.9869
    Epoch 238/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0430 - accuracy: 0.9900 - val_loss: 0.0498 - val_accuracy: 0.9869
    Epoch 239/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0448 - accuracy: 0.9879 - val_loss: 0.0484 - val_accuracy: 0.9885
    Epoch 240/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0426 - accuracy: 0.9890 - val_loss: 0.0493 - val_accuracy: 0.9846
    Epoch 241/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0423 - accuracy: 0.9882 - val_loss: 0.0522 - val_accuracy: 0.9846
    Epoch 242/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0420 - accuracy: 0.9897 - val_loss: 0.0489 - val_accuracy: 0.9869
    Epoch 243/2000
    8/8 [==============================] - 0s 21ms/step - loss: 0.0420 - accuracy: 0.9882 - val_loss: 0.0481 - val_accuracy: 0.9885
    Epoch 244/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0417 - accuracy: 0.9892 - val_loss: 0.0490 - val_accuracy: 0.9877
    Epoch 245/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0410 - accuracy: 0.9892 - val_loss: 0.0485 - val_accuracy: 0.9862
    Epoch 246/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0416 - accuracy: 0.9885 - val_loss: 0.0494 - val_accuracy: 0.9869
    Epoch 247/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0451 - accuracy: 0.9879 - val_loss: 0.0494 - val_accuracy: 0.9854
    Epoch 248/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0471 - accuracy: 0.9882 - val_loss: 0.0523 - val_accuracy: 0.9846
    Epoch 249/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0437 - accuracy: 0.9900 - val_loss: 0.0570 - val_accuracy: 0.9846
    Epoch 250/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0443 - accuracy: 0.9887 - val_loss: 0.0492 - val_accuracy: 0.9854
    Epoch 251/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0440 - accuracy: 0.9877 - val_loss: 0.0493 - val_accuracy: 0.9885
    Epoch 252/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0434 - accuracy: 0.9874 - val_loss: 0.0524 - val_accuracy: 0.9862
    Epoch 253/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0457 - accuracy: 0.9882 - val_loss: 0.0491 - val_accuracy: 0.9885
    Epoch 254/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0492 - accuracy: 0.9867 - val_loss: 0.0500 - val_accuracy: 0.9854
    Epoch 255/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0529 - accuracy: 0.9833 - val_loss: 0.0597 - val_accuracy: 0.9831
    Epoch 256/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0465 - accuracy: 0.9882 - val_loss: 0.0574 - val_accuracy: 0.9838
    Epoch 257/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0439 - accuracy: 0.9885 - val_loss: 0.0481 - val_accuracy: 0.9869
    Epoch 258/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0415 - accuracy: 0.9879 - val_loss: 0.0487 - val_accuracy: 0.9862
    Epoch 259/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0408 - accuracy: 0.9890 - val_loss: 0.0480 - val_accuracy: 0.9869
    Epoch 260/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0405 - accuracy: 0.9897 - val_loss: 0.0477 - val_accuracy: 0.9869
    Epoch 261/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0408 - accuracy: 0.9887 - val_loss: 0.0478 - val_accuracy: 0.9885
    Epoch 262/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0420 - accuracy: 0.9887 - val_loss: 0.0481 - val_accuracy: 0.9869
    Epoch 263/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0433 - accuracy: 0.9882 - val_loss: 0.0507 - val_accuracy: 0.9869
    Epoch 264/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0415 - accuracy: 0.9879 - val_loss: 0.0484 - val_accuracy: 0.9877
    Epoch 265/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0402 - accuracy: 0.9908 - val_loss: 0.0516 - val_accuracy: 0.9846
    Epoch 266/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0401 - accuracy: 0.9910 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 267/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0405 - accuracy: 0.9887 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 268/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0400 - accuracy: 0.9900 - val_loss: 0.0529 - val_accuracy: 0.9838
    Epoch 269/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0423 - accuracy: 0.9890 - val_loss: 0.0480 - val_accuracy: 0.9869
    Epoch 270/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0426 - accuracy: 0.9887 - val_loss: 0.0481 - val_accuracy: 0.9869
    Epoch 271/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0414 - accuracy: 0.9882 - val_loss: 0.0477 - val_accuracy: 0.9869
    Epoch 272/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0418 - accuracy: 0.9890 - val_loss: 0.0509 - val_accuracy: 0.9869
    Epoch 273/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0420 - accuracy: 0.9895 - val_loss: 0.0486 - val_accuracy: 0.9885
    Epoch 274/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0433 - accuracy: 0.9892 - val_loss: 0.0538 - val_accuracy: 0.9838
    Epoch 275/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0431 - accuracy: 0.9882 - val_loss: 0.0547 - val_accuracy: 0.9838
    Epoch 276/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0435 - accuracy: 0.9882 - val_loss: 0.0511 - val_accuracy: 0.9846
    Epoch 277/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0468 - accuracy: 0.9882 - val_loss: 0.0501 - val_accuracy: 0.9877
    Epoch 278/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0441 - accuracy: 0.9882 - val_loss: 0.0624 - val_accuracy: 0.9823
    Epoch 279/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0471 - accuracy: 0.9851 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 280/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0430 - accuracy: 0.9887 - val_loss: 0.0467 - val_accuracy: 0.9869
    Epoch 281/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0419 - accuracy: 0.9902 - val_loss: 0.0517 - val_accuracy: 0.9846
    Epoch 282/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0401 - accuracy: 0.9887 - val_loss: 0.0470 - val_accuracy: 0.9869
    Epoch 283/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0400 - accuracy: 0.9887 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 284/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0412 - accuracy: 0.9892 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 285/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0398 - accuracy: 0.9895 - val_loss: 0.0470 - val_accuracy: 0.9885
    Epoch 286/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0405 - accuracy: 0.9885 - val_loss: 0.0469 - val_accuracy: 0.9862
    Epoch 287/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0392 - accuracy: 0.9908 - val_loss: 0.0471 - val_accuracy: 0.9862
    Epoch 288/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0393 - accuracy: 0.9892 - val_loss: 0.0504 - val_accuracy: 0.9854
    Epoch 289/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0403 - accuracy: 0.9895 - val_loss: 0.0495 - val_accuracy: 0.9862
    Epoch 290/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0413 - accuracy: 0.9895 - val_loss: 0.0479 - val_accuracy: 0.9862
    Epoch 291/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0416 - accuracy: 0.9892 - val_loss: 0.0473 - val_accuracy: 0.9885
    Epoch 292/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0462 - accuracy: 0.9874 - val_loss: 0.0509 - val_accuracy: 0.9869
    Epoch 293/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0421 - accuracy: 0.9890 - val_loss: 0.0505 - val_accuracy: 0.9869
    Epoch 294/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0419 - accuracy: 0.9890 - val_loss: 0.0470 - val_accuracy: 0.9869
    Epoch 295/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0430 - accuracy: 0.9879 - val_loss: 0.0501 - val_accuracy: 0.9862
    Epoch 296/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0410 - accuracy: 0.9885 - val_loss: 0.0547 - val_accuracy: 0.9846
    Epoch 297/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0424 - accuracy: 0.9892 - val_loss: 0.0509 - val_accuracy: 0.9854
    Epoch 298/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0415 - accuracy: 0.9882 - val_loss: 0.0470 - val_accuracy: 0.9869
    Epoch 299/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0402 - accuracy: 0.9908 - val_loss: 0.0470 - val_accuracy: 0.9885
    Epoch 300/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0408 - accuracy: 0.9905 - val_loss: 0.0464 - val_accuracy: 0.9862
    Epoch 301/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0391 - accuracy: 0.9908 - val_loss: 0.0466 - val_accuracy: 0.9869
    Epoch 302/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0391 - accuracy: 0.9902 - val_loss: 0.0463 - val_accuracy: 0.9869
    Epoch 303/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0391 - accuracy: 0.9908 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 304/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0403 - accuracy: 0.9895 - val_loss: 0.0466 - val_accuracy: 0.9885
    Epoch 305/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0397 - accuracy: 0.9897 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 306/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0393 - accuracy: 0.9895 - val_loss: 0.0470 - val_accuracy: 0.9869
    Epoch 307/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0393 - accuracy: 0.9895 - val_loss: 0.0502 - val_accuracy: 0.9854
    Epoch 308/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0394 - accuracy: 0.9908 - val_loss: 0.0465 - val_accuracy: 0.9862
    Epoch 309/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0397 - accuracy: 0.9900 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 310/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0402 - accuracy: 0.9892 - val_loss: 0.0481 - val_accuracy: 0.9885
    Epoch 311/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0401 - accuracy: 0.9882 - val_loss: 0.0471 - val_accuracy: 0.9885
    Epoch 312/2000
    8/8 [==============================] - 0s 25ms/step - loss: 0.0399 - accuracy: 0.9895 - val_loss: 0.0478 - val_accuracy: 0.9869
    Epoch 313/2000
    8/8 [==============================] - 0s 36ms/step - loss: 0.0387 - accuracy: 0.9895 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 314/2000
    8/8 [==============================] - 0s 35ms/step - loss: 0.0391 - accuracy: 0.9895 - val_loss: 0.0465 - val_accuracy: 0.9892
    Epoch 315/2000
    8/8 [==============================] - 0s 34ms/step - loss: 0.0387 - accuracy: 0.9900 - val_loss: 0.0465 - val_accuracy: 0.9862
    Epoch 316/2000
    8/8 [==============================] - 0s 30ms/step - loss: 0.0391 - accuracy: 0.9890 - val_loss: 0.0516 - val_accuracy: 0.9846
    Epoch 317/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0392 - accuracy: 0.9902 - val_loss: 0.0481 - val_accuracy: 0.9862
    Epoch 318/2000
    8/8 [==============================] - 0s 27ms/step - loss: 0.0391 - accuracy: 0.9902 - val_loss: 0.0498 - val_accuracy: 0.9862
    Epoch 319/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0417 - accuracy: 0.9874 - val_loss: 0.0507 - val_accuracy: 0.9854
    Epoch 320/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0414 - accuracy: 0.9890 - val_loss: 0.0481 - val_accuracy: 0.9862
    Epoch 321/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0387 - accuracy: 0.9900 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 322/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0391 - accuracy: 0.9895 - val_loss: 0.0459 - val_accuracy: 0.9892
    Epoch 323/2000
    8/8 [==============================] - 0s 39ms/step - loss: 0.0417 - accuracy: 0.9882 - val_loss: 0.0493 - val_accuracy: 0.9877
    Epoch 324/2000
    8/8 [==============================] - 0s 25ms/step - loss: 0.0413 - accuracy: 0.9897 - val_loss: 0.0466 - val_accuracy: 0.9900
    Epoch 325/2000
    8/8 [==============================] - 0s 33ms/step - loss: 0.0391 - accuracy: 0.9913 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 326/2000
    8/8 [==============================] - 0s 35ms/step - loss: 0.0412 - accuracy: 0.9887 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 327/2000
    8/8 [==============================] - 0s 34ms/step - loss: 0.0401 - accuracy: 0.9897 - val_loss: 0.0462 - val_accuracy: 0.9900
    Epoch 328/2000
    8/8 [==============================] - 0s 21ms/step - loss: 0.0389 - accuracy: 0.9908 - val_loss: 0.0460 - val_accuracy: 0.9900
    Epoch 329/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0386 - accuracy: 0.9905 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 330/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0392 - accuracy: 0.9913 - val_loss: 0.0468 - val_accuracy: 0.9869
    Epoch 331/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0387 - accuracy: 0.9905 - val_loss: 0.0458 - val_accuracy: 0.9862
    Epoch 332/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0384 - accuracy: 0.9897 - val_loss: 0.0469 - val_accuracy: 0.9892
    Epoch 333/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0393 - accuracy: 0.9892 - val_loss: 0.0468 - val_accuracy: 0.9900
    Epoch 334/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0379 - accuracy: 0.9908 - val_loss: 0.0464 - val_accuracy: 0.9869
    Epoch 335/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0381 - accuracy: 0.9908 - val_loss: 0.0458 - val_accuracy: 0.9900
    Epoch 336/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0395 - accuracy: 0.9885 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 337/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0384 - accuracy: 0.9900 - val_loss: 0.0481 - val_accuracy: 0.9869
    Epoch 338/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0381 - accuracy: 0.9902 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 339/2000
    8/8 [==============================] - 0s 23ms/step - loss: 0.0377 - accuracy: 0.9900 - val_loss: 0.0461 - val_accuracy: 0.9892
    Epoch 340/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0381 - accuracy: 0.9905 - val_loss: 0.0457 - val_accuracy: 0.9885
    Epoch 341/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0386 - accuracy: 0.9897 - val_loss: 0.0464 - val_accuracy: 0.9869
    Epoch 342/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0374 - accuracy: 0.9918 - val_loss: 0.0458 - val_accuracy: 0.9900
    Epoch 343/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0382 - accuracy: 0.9897 - val_loss: 0.0455 - val_accuracy: 0.9869
    Epoch 344/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0376 - accuracy: 0.9908 - val_loss: 0.0481 - val_accuracy: 0.9862
    Epoch 345/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0386 - accuracy: 0.9905 - val_loss: 0.0457 - val_accuracy: 0.9885
    Epoch 346/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0383 - accuracy: 0.9900 - val_loss: 0.0480 - val_accuracy: 0.9885
    Epoch 347/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0403 - accuracy: 0.9895 - val_loss: 0.0473 - val_accuracy: 0.9869
    Epoch 348/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0417 - accuracy: 0.9877 - val_loss: 0.0549 - val_accuracy: 0.9846
    Epoch 349/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0442 - accuracy: 0.9877 - val_loss: 0.0597 - val_accuracy: 0.9831
    Epoch 350/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0450 - accuracy: 0.9890 - val_loss: 0.0456 - val_accuracy: 0.9877
    Epoch 351/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0432 - accuracy: 0.9879 - val_loss: 0.0493 - val_accuracy: 0.9862
    Epoch 352/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0385 - accuracy: 0.9908 - val_loss: 0.0459 - val_accuracy: 0.9869
    Epoch 353/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0375 - accuracy: 0.9910 - val_loss: 0.0454 - val_accuracy: 0.9885
    Epoch 354/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0376 - accuracy: 0.9905 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 355/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0376 - accuracy: 0.9915 - val_loss: 0.0455 - val_accuracy: 0.9877
    Epoch 356/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0384 - accuracy: 0.9905 - val_loss: 0.0459 - val_accuracy: 0.9900
    Epoch 357/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0378 - accuracy: 0.9900 - val_loss: 0.0469 - val_accuracy: 0.9869
    Epoch 358/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0376 - accuracy: 0.9910 - val_loss: 0.0468 - val_accuracy: 0.9869
    Epoch 359/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0382 - accuracy: 0.9900 - val_loss: 0.0460 - val_accuracy: 0.9892
    Epoch 360/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0382 - accuracy: 0.9902 - val_loss: 0.0457 - val_accuracy: 0.9892
    Epoch 361/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0376 - accuracy: 0.9918 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 362/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0378 - accuracy: 0.9905 - val_loss: 0.0456 - val_accuracy: 0.9892
    Epoch 363/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0375 - accuracy: 0.9913 - val_loss: 0.0463 - val_accuracy: 0.9869
    Epoch 364/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0381 - accuracy: 0.9908 - val_loss: 0.0455 - val_accuracy: 0.9877
    Epoch 365/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0402 - accuracy: 0.9892 - val_loss: 0.0480 - val_accuracy: 0.9854
    Epoch 366/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0378 - accuracy: 0.9913 - val_loss: 0.0489 - val_accuracy: 0.9869
    Epoch 367/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0384 - accuracy: 0.9900 - val_loss: 0.0454 - val_accuracy: 0.9877
    Epoch 368/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0373 - accuracy: 0.9908 - val_loss: 0.0459 - val_accuracy: 0.9900
    Epoch 369/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0373 - accuracy: 0.9915 - val_loss: 0.0457 - val_accuracy: 0.9877
    Epoch 370/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0378 - accuracy: 0.9908 - val_loss: 0.0454 - val_accuracy: 0.9885
    Epoch 371/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0373 - accuracy: 0.9910 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 372/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0369 - accuracy: 0.9902 - val_loss: 0.0478 - val_accuracy: 0.9885
    Epoch 373/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0400 - accuracy: 0.9897 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 374/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0400 - accuracy: 0.9902 - val_loss: 0.0450 - val_accuracy: 0.9877
    Epoch 375/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0407 - accuracy: 0.9905 - val_loss: 0.0541 - val_accuracy: 0.9838
    Epoch 376/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0388 - accuracy: 0.9897 - val_loss: 0.0557 - val_accuracy: 0.9831
    Epoch 377/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0413 - accuracy: 0.9887 - val_loss: 0.0563 - val_accuracy: 0.9846
    Epoch 378/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0402 - accuracy: 0.9872 - val_loss: 0.0452 - val_accuracy: 0.9900
    Epoch 379/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0397 - accuracy: 0.9879 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 380/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0375 - accuracy: 0.9905 - val_loss: 0.0451 - val_accuracy: 0.9892
    Epoch 381/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0371 - accuracy: 0.9910 - val_loss: 0.0491 - val_accuracy: 0.9862
    Epoch 382/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0365 - accuracy: 0.9895 - val_loss: 0.0482 - val_accuracy: 0.9885
    Epoch 383/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0402 - accuracy: 0.9887 - val_loss: 0.0504 - val_accuracy: 0.9877
    Epoch 384/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0392 - accuracy: 0.9895 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 385/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0385 - accuracy: 0.9902 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 386/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0423 - accuracy: 0.9885 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 387/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0367 - accuracy: 0.9915 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 388/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0372 - accuracy: 0.9905 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 389/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0379 - accuracy: 0.9902 - val_loss: 0.0449 - val_accuracy: 0.9900
    Epoch 390/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0375 - accuracy: 0.9902 - val_loss: 0.0451 - val_accuracy: 0.9900
    Epoch 391/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0375 - accuracy: 0.9913 - val_loss: 0.0455 - val_accuracy: 0.9900
    Epoch 392/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0373 - accuracy: 0.9902 - val_loss: 0.0473 - val_accuracy: 0.9885
    Epoch 393/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0389 - accuracy: 0.9895 - val_loss: 0.0456 - val_accuracy: 0.9900
    Epoch 394/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0376 - accuracy: 0.9913 - val_loss: 0.0473 - val_accuracy: 0.9869
    Epoch 395/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0374 - accuracy: 0.9913 - val_loss: 0.0453 - val_accuracy: 0.9877
    Epoch 396/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0377 - accuracy: 0.9900 - val_loss: 0.0489 - val_accuracy: 0.9862
    Epoch 397/2000
    8/8 [==============================] - 0s 23ms/step - loss: 0.0372 - accuracy: 0.9910 - val_loss: 0.0473 - val_accuracy: 0.9846
    Epoch 398/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0368 - accuracy: 0.9923 - val_loss: 0.0453 - val_accuracy: 0.9877
    Epoch 399/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0366 - accuracy: 0.9913 - val_loss: 0.0448 - val_accuracy: 0.9885
    Epoch 400/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0365 - accuracy: 0.9918 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 401/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0370 - accuracy: 0.9908 - val_loss: 0.0451 - val_accuracy: 0.9892
    Epoch 402/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0382 - accuracy: 0.9910 - val_loss: 0.0485 - val_accuracy: 0.9877
    Epoch 403/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0397 - accuracy: 0.9895 - val_loss: 0.0452 - val_accuracy: 0.9908
    Epoch 404/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0360 - accuracy: 0.9918 - val_loss: 0.0466 - val_accuracy: 0.9862
    Epoch 405/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0372 - accuracy: 0.9908 - val_loss: 0.0459 - val_accuracy: 0.9869
    Epoch 406/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0368 - accuracy: 0.9910 - val_loss: 0.0448 - val_accuracy: 0.9892
    Epoch 407/2000
    8/8 [==============================] - 0s 22ms/step - loss: 0.0371 - accuracy: 0.9908 - val_loss: 0.0452 - val_accuracy: 0.9900
    Epoch 408/2000
    8/8 [==============================] - 0s 26ms/step - loss: 0.0388 - accuracy: 0.9885 - val_loss: 0.0449 - val_accuracy: 0.9885
    Epoch 409/2000
    8/8 [==============================] - 0s 21ms/step - loss: 0.0371 - accuracy: 0.9908 - val_loss: 0.0467 - val_accuracy: 0.9892
    Epoch 410/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0378 - accuracy: 0.9910 - val_loss: 0.0460 - val_accuracy: 0.9900
    Epoch 411/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0382 - accuracy: 0.9902 - val_loss: 0.0448 - val_accuracy: 0.9900
    Epoch 412/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0393 - accuracy: 0.9902 - val_loss: 0.0449 - val_accuracy: 0.9885
    Epoch 413/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0363 - accuracy: 0.9910 - val_loss: 0.0461 - val_accuracy: 0.9877
    Epoch 414/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0368 - accuracy: 0.9918 - val_loss: 0.0494 - val_accuracy: 0.9854
    Epoch 415/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0381 - accuracy: 0.9905 - val_loss: 0.0461 - val_accuracy: 0.9862
    Epoch 416/2000
    8/8 [==============================] - 0s 28ms/step - loss: 0.0371 - accuracy: 0.9908 - val_loss: 0.0455 - val_accuracy: 0.9869
    Epoch 417/2000
    8/8 [==============================] - 0s 30ms/step - loss: 0.0379 - accuracy: 0.9908 - val_loss: 0.0447 - val_accuracy: 0.9885
    Epoch 418/2000
    8/8 [==============================] - 0s 21ms/step - loss: 0.0361 - accuracy: 0.9902 - val_loss: 0.0447 - val_accuracy: 0.9900
    Epoch 419/2000
    8/8 [==============================] - 0s 27ms/step - loss: 0.0364 - accuracy: 0.9910 - val_loss: 0.0462 - val_accuracy: 0.9892
    Epoch 420/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0412 - accuracy: 0.9897 - val_loss: 0.0454 - val_accuracy: 0.9877
    Epoch 421/2000
    8/8 [==============================] - 0s 31ms/step - loss: 0.0415 - accuracy: 0.9872 - val_loss: 0.0452 - val_accuracy: 0.9885
    Epoch 422/2000
    8/8 [==============================] - 0s 26ms/step - loss: 0.0378 - accuracy: 0.9908 - val_loss: 0.0453 - val_accuracy: 0.9877
    Epoch 423/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0360 - accuracy: 0.9918 - val_loss: 0.0449 - val_accuracy: 0.9908
    Epoch 424/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0382 - accuracy: 0.9900 - val_loss: 0.0480 - val_accuracy: 0.9892
    Epoch 425/2000
    8/8 [==============================] - 0s 36ms/step - loss: 0.0362 - accuracy: 0.9923 - val_loss: 0.0459 - val_accuracy: 0.9877
    Epoch 426/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0367 - accuracy: 0.9910 - val_loss: 0.0451 - val_accuracy: 0.9877
    Epoch 427/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0357 - accuracy: 0.9920 - val_loss: 0.0449 - val_accuracy: 0.9877
    Epoch 428/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0357 - accuracy: 0.9920 - val_loss: 0.0499 - val_accuracy: 0.9854
    Epoch 429/2000
    8/8 [==============================] - 0s 29ms/step - loss: 0.0374 - accuracy: 0.9908 - val_loss: 0.0462 - val_accuracy: 0.9869
    Epoch 430/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0365 - accuracy: 0.9923 - val_loss: 0.0451 - val_accuracy: 0.9877
    Epoch 431/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0358 - accuracy: 0.9918 - val_loss: 0.0462 - val_accuracy: 0.9869
    Epoch 432/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0365 - accuracy: 0.9902 - val_loss: 0.0451 - val_accuracy: 0.9885
    Epoch 433/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0374 - accuracy: 0.9918 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 434/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0375 - accuracy: 0.9900 - val_loss: 0.0448 - val_accuracy: 0.9885
    Epoch 435/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0369 - accuracy: 0.9918 - val_loss: 0.0449 - val_accuracy: 0.9877
    Epoch 436/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0382 - accuracy: 0.9897 - val_loss: 0.0535 - val_accuracy: 0.9838
    Epoch 437/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0375 - accuracy: 0.9915 - val_loss: 0.0454 - val_accuracy: 0.9877
    Epoch 438/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0391 - accuracy: 0.9897 - val_loss: 0.0488 - val_accuracy: 0.9862
    Epoch 439/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0436 - accuracy: 0.9877 - val_loss: 0.0457 - val_accuracy: 0.9908
    Epoch 440/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0384 - accuracy: 0.9900 - val_loss: 0.0444 - val_accuracy: 0.9900
    Epoch 441/2000
    8/8 [==============================] - 0s 30ms/step - loss: 0.0410 - accuracy: 0.9887 - val_loss: 0.0465 - val_accuracy: 0.9862
    Epoch 442/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0396 - accuracy: 0.9897 - val_loss: 0.0447 - val_accuracy: 0.9892
    Epoch 443/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0395 - accuracy: 0.9905 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 444/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0360 - accuracy: 0.9902 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 445/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0375 - accuracy: 0.9897 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 446/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0363 - accuracy: 0.9895 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 447/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0368 - accuracy: 0.9902 - val_loss: 0.0512 - val_accuracy: 0.9854
    Epoch 448/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0389 - accuracy: 0.9910 - val_loss: 0.0466 - val_accuracy: 0.9869
    Epoch 449/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0365 - accuracy: 0.9910 - val_loss: 0.0451 - val_accuracy: 0.9885
    Epoch 450/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0376 - accuracy: 0.9915 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 451/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0388 - accuracy: 0.9890 - val_loss: 0.0472 - val_accuracy: 0.9862
    Epoch 452/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0368 - accuracy: 0.9915 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 453/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0370 - accuracy: 0.9908 - val_loss: 0.0482 - val_accuracy: 0.9869
    Epoch 454/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0374 - accuracy: 0.9905 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 455/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0356 - accuracy: 0.9915 - val_loss: 0.0448 - val_accuracy: 0.9877
    Epoch 456/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0357 - accuracy: 0.9915 - val_loss: 0.0445 - val_accuracy: 0.9908
    Epoch 457/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0364 - accuracy: 0.9918 - val_loss: 0.0446 - val_accuracy: 0.9908
    Epoch 458/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0355 - accuracy: 0.9920 - val_loss: 0.0445 - val_accuracy: 0.9900
    Epoch 459/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0370 - accuracy: 0.9913 - val_loss: 0.0444 - val_accuracy: 0.9908
    Epoch 460/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0371 - accuracy: 0.9897 - val_loss: 0.0472 - val_accuracy: 0.9885
    Epoch 461/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0362 - accuracy: 0.9910 - val_loss: 0.0455 - val_accuracy: 0.9877
    Epoch 462/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0382 - accuracy: 0.9895 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 463/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0451 - accuracy: 0.9874 - val_loss: 0.0452 - val_accuracy: 0.9885
    Epoch 464/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0409 - accuracy: 0.9882 - val_loss: 0.0488 - val_accuracy: 0.9869
    Epoch 465/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0371 - accuracy: 0.9920 - val_loss: 0.0576 - val_accuracy: 0.9838
    Epoch 466/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0421 - accuracy: 0.9877 - val_loss: 0.0457 - val_accuracy: 0.9877
    Epoch 467/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0418 - accuracy: 0.9882 - val_loss: 0.0452 - val_accuracy: 0.9885
    Epoch 468/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0426 - accuracy: 0.9897 - val_loss: 0.0466 - val_accuracy: 0.9892
    Epoch 469/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0373 - accuracy: 0.9918 - val_loss: 0.0461 - val_accuracy: 0.9892
    Epoch 470/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0362 - accuracy: 0.9897 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 471/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0358 - accuracy: 0.9918 - val_loss: 0.0471 - val_accuracy: 0.9862
    Epoch 472/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0370 - accuracy: 0.9913 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 473/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0367 - accuracy: 0.9897 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 474/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0382 - accuracy: 0.9902 - val_loss: 0.0521 - val_accuracy: 0.9862
    Epoch 475/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0379 - accuracy: 0.9920 - val_loss: 0.0447 - val_accuracy: 0.9877
    Epoch 476/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0368 - accuracy: 0.9918 - val_loss: 0.0462 - val_accuracy: 0.9862
    Epoch 477/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0366 - accuracy: 0.9900 - val_loss: 0.0494 - val_accuracy: 0.9854
    Epoch 478/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0364 - accuracy: 0.9915 - val_loss: 0.0462 - val_accuracy: 0.9862
    Epoch 479/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0389 - accuracy: 0.9890 - val_loss: 0.0499 - val_accuracy: 0.9854
    Epoch 480/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0414 - accuracy: 0.9892 - val_loss: 0.0459 - val_accuracy: 0.9869
    Epoch 481/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0381 - accuracy: 0.9897 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 482/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0357 - accuracy: 0.9913 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 483/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0357 - accuracy: 0.9918 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 484/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0358 - accuracy: 0.9915 - val_loss: 0.0453 - val_accuracy: 0.9877
    Epoch 485/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0351 - accuracy: 0.9910 - val_loss: 0.0447 - val_accuracy: 0.9908
    Epoch 486/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0374 - accuracy: 0.9910 - val_loss: 0.0491 - val_accuracy: 0.9892
    Epoch 487/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0382 - accuracy: 0.9902 - val_loss: 0.0446 - val_accuracy: 0.9908
    Epoch 488/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0362 - accuracy: 0.9918 - val_loss: 0.0453 - val_accuracy: 0.9877
    Epoch 489/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0356 - accuracy: 0.9915 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 490/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0374 - accuracy: 0.9913 - val_loss: 0.0459 - val_accuracy: 0.9877
    Epoch 491/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0373 - accuracy: 0.9908 - val_loss: 0.0482 - val_accuracy: 0.9869
    Epoch 492/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0354 - accuracy: 0.9915 - val_loss: 0.0449 - val_accuracy: 0.9885
    Epoch 493/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0355 - accuracy: 0.9918 - val_loss: 0.0446 - val_accuracy: 0.9900
    Epoch 494/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0379 - accuracy: 0.9915 - val_loss: 0.0443 - val_accuracy: 0.9900
    Epoch 495/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0392 - accuracy: 0.9897 - val_loss: 0.0469 - val_accuracy: 0.9862
    Epoch 496/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0392 - accuracy: 0.9915 - val_loss: 0.0468 - val_accuracy: 0.9869
    Epoch 497/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0350 - accuracy: 0.9918 - val_loss: 0.0454 - val_accuracy: 0.9877
    Epoch 498/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0351 - accuracy: 0.9920 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 499/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0351 - accuracy: 0.9913 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 500/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0351 - accuracy: 0.9913 - val_loss: 0.0444 - val_accuracy: 0.9908
    Epoch 501/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0350 - accuracy: 0.9918 - val_loss: 0.0461 - val_accuracy: 0.9869
    Epoch 502/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0347 - accuracy: 0.9920 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 503/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0351 - accuracy: 0.9918 - val_loss: 0.0445 - val_accuracy: 0.9885
    Epoch 504/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0350 - accuracy: 0.9918 - val_loss: 0.0489 - val_accuracy: 0.9869
    Epoch 505/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0343 - accuracy: 0.9923 - val_loss: 0.0454 - val_accuracy: 0.9900
    Epoch 506/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0368 - accuracy: 0.9908 - val_loss: 0.0465 - val_accuracy: 0.9892
    Epoch 507/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0370 - accuracy: 0.9905 - val_loss: 0.0450 - val_accuracy: 0.9908
    Epoch 508/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0361 - accuracy: 0.9910 - val_loss: 0.0445 - val_accuracy: 0.9885
    Epoch 509/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0350 - accuracy: 0.9926 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 510/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0356 - accuracy: 0.9918 - val_loss: 0.0516 - val_accuracy: 0.9854
    Epoch 511/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0375 - accuracy: 0.9905 - val_loss: 0.0531 - val_accuracy: 0.9838
    Epoch 512/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0389 - accuracy: 0.9902 - val_loss: 0.0520 - val_accuracy: 0.9846
    Epoch 513/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0369 - accuracy: 0.9918 - val_loss: 0.0443 - val_accuracy: 0.9900
    Epoch 514/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0357 - accuracy: 0.9915 - val_loss: 0.0450 - val_accuracy: 0.9877
    Epoch 515/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0370 - accuracy: 0.9905 - val_loss: 0.0498 - val_accuracy: 0.9854
    Epoch 516/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0356 - accuracy: 0.9923 - val_loss: 0.0471 - val_accuracy: 0.9862
    Epoch 517/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0349 - accuracy: 0.9913 - val_loss: 0.0460 - val_accuracy: 0.9900
    Epoch 518/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0349 - accuracy: 0.9908 - val_loss: 0.0449 - val_accuracy: 0.9908
    Epoch 519/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0386 - accuracy: 0.9890 - val_loss: 0.0444 - val_accuracy: 0.9892
    Epoch 520/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0425 - accuracy: 0.9887 - val_loss: 0.0456 - val_accuracy: 0.9877
    Epoch 521/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0392 - accuracy: 0.9879 - val_loss: 0.0511 - val_accuracy: 0.9869
    Epoch 522/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0383 - accuracy: 0.9913 - val_loss: 0.0479 - val_accuracy: 0.9862
    Epoch 523/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0378 - accuracy: 0.9918 - val_loss: 0.0472 - val_accuracy: 0.9862
    Epoch 524/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0372 - accuracy: 0.9920 - val_loss: 0.0498 - val_accuracy: 0.9869
    Epoch 525/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0384 - accuracy: 0.9910 - val_loss: 0.0483 - val_accuracy: 0.9862
    Epoch 526/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0348 - accuracy: 0.9931 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 527/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0351 - accuracy: 0.9926 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 528/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0358 - accuracy: 0.9908 - val_loss: 0.0457 - val_accuracy: 0.9869
    Epoch 529/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0351 - accuracy: 0.9926 - val_loss: 0.0475 - val_accuracy: 0.9869
    Epoch 530/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0360 - accuracy: 0.9926 - val_loss: 0.0501 - val_accuracy: 0.9862
    Epoch 531/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0393 - accuracy: 0.9892 - val_loss: 0.0574 - val_accuracy: 0.9831
    Epoch 532/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0384 - accuracy: 0.9908 - val_loss: 0.0496 - val_accuracy: 0.9862
    Epoch 533/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0370 - accuracy: 0.9910 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 534/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0379 - accuracy: 0.9908 - val_loss: 0.0472 - val_accuracy: 0.9892
    Epoch 535/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0374 - accuracy: 0.9915 - val_loss: 0.0485 - val_accuracy: 0.9892
    Epoch 536/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0371 - accuracy: 0.9908 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 537/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0349 - accuracy: 0.9918 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 538/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0345 - accuracy: 0.9928 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 539/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0377 - accuracy: 0.9913 - val_loss: 0.0469 - val_accuracy: 0.9869
    Epoch 540/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0361 - accuracy: 0.9913 - val_loss: 0.0445 - val_accuracy: 0.9908
    Epoch 541/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0358 - accuracy: 0.9918 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 542/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0378 - accuracy: 0.9908 - val_loss: 0.0462 - val_accuracy: 0.9862
    Epoch 543/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0364 - accuracy: 0.9910 - val_loss: 0.0536 - val_accuracy: 0.9846
    Epoch 544/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0382 - accuracy: 0.9902 - val_loss: 0.0454 - val_accuracy: 0.9885
    Epoch 545/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0347 - accuracy: 0.9928 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 546/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0362 - accuracy: 0.9897 - val_loss: 0.0450 - val_accuracy: 0.9877
    Epoch 547/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0338 - accuracy: 0.9920 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 548/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0339 - accuracy: 0.9931 - val_loss: 0.0442 - val_accuracy: 0.9900
    Epoch 549/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0350 - accuracy: 0.9915 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 550/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0341 - accuracy: 0.9926 - val_loss: 0.0471 - val_accuracy: 0.9862
    Epoch 551/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0339 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 552/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0345 - accuracy: 0.9913 - val_loss: 0.0441 - val_accuracy: 0.9900
    Epoch 553/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0350 - accuracy: 0.9928 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 554/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0347 - accuracy: 0.9920 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 555/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0346 - accuracy: 0.9931 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 556/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0343 - accuracy: 0.9926 - val_loss: 0.0449 - val_accuracy: 0.9908
    Epoch 557/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0362 - accuracy: 0.9915 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 558/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0364 - accuracy: 0.9905 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 559/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9920 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 560/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9923 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 561/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0356 - accuracy: 0.9913 - val_loss: 0.0444 - val_accuracy: 0.9908
    Epoch 562/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0392 - accuracy: 0.9908 - val_loss: 0.0463 - val_accuracy: 0.9862
    Epoch 563/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0376 - accuracy: 0.9908 - val_loss: 0.0458 - val_accuracy: 0.9862
    Epoch 564/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0354 - accuracy: 0.9915 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 565/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0342 - accuracy: 0.9920 - val_loss: 0.0476 - val_accuracy: 0.9869
    Epoch 566/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0348 - accuracy: 0.9918 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 567/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0339 - accuracy: 0.9928 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 568/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0341 - accuracy: 0.9923 - val_loss: 0.0444 - val_accuracy: 0.9885
    Epoch 569/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0337 - accuracy: 0.9923 - val_loss: 0.0488 - val_accuracy: 0.9892
    Epoch 570/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0352 - accuracy: 0.9913 - val_loss: 0.0445 - val_accuracy: 0.9900
    Epoch 571/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0360 - accuracy: 0.9908 - val_loss: 0.0455 - val_accuracy: 0.9900
    Epoch 572/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0404 - accuracy: 0.9895 - val_loss: 0.0445 - val_accuracy: 0.9908
    Epoch 573/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0381 - accuracy: 0.9895 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 574/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0361 - accuracy: 0.9902 - val_loss: 0.0480 - val_accuracy: 0.9862
    Epoch 575/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0373 - accuracy: 0.9908 - val_loss: 0.0537 - val_accuracy: 0.9854
    Epoch 576/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0367 - accuracy: 0.9900 - val_loss: 0.0505 - val_accuracy: 0.9862
    Epoch 577/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0352 - accuracy: 0.9918 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 578/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0341 - accuracy: 0.9923 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 579/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0345 - accuracy: 0.9926 - val_loss: 0.0447 - val_accuracy: 0.9885
    Epoch 580/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0347 - accuracy: 0.9920 - val_loss: 0.0447 - val_accuracy: 0.9885
    Epoch 581/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0343 - accuracy: 0.9928 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 582/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0342 - accuracy: 0.9910 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 583/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0364 - accuracy: 0.9931 - val_loss: 0.0439 - val_accuracy: 0.9892
    Epoch 584/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0342 - accuracy: 0.9920 - val_loss: 0.0477 - val_accuracy: 0.9862
    Epoch 585/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0337 - accuracy: 0.9931 - val_loss: 0.0442 - val_accuracy: 0.9892
    Epoch 586/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0356 - accuracy: 0.9915 - val_loss: 0.0490 - val_accuracy: 0.9862
    Epoch 587/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0353 - accuracy: 0.9913 - val_loss: 0.0468 - val_accuracy: 0.9862
    Epoch 588/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9931 - val_loss: 0.0442 - val_accuracy: 0.9892
    Epoch 589/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0335 - accuracy: 0.9926 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 590/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 591/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9926 - val_loss: 0.0458 - val_accuracy: 0.9900
    Epoch 592/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0358 - accuracy: 0.9905 - val_loss: 0.0457 - val_accuracy: 0.9877
    Epoch 593/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0347 - accuracy: 0.9926 - val_loss: 0.0476 - val_accuracy: 0.9869
    Epoch 594/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0336 - accuracy: 0.9920 - val_loss: 0.0450 - val_accuracy: 0.9908
    Epoch 595/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0356 - accuracy: 0.9910 - val_loss: 0.0496 - val_accuracy: 0.9877
    Epoch 596/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0353 - accuracy: 0.9900 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 597/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0339 - accuracy: 0.9928 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 598/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0340 - accuracy: 0.9923 - val_loss: 0.0480 - val_accuracy: 0.9862
    Epoch 599/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0347 - accuracy: 0.9905 - val_loss: 0.0493 - val_accuracy: 0.9862
    Epoch 600/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0371 - accuracy: 0.9910 - val_loss: 0.0500 - val_accuracy: 0.9862
    Epoch 601/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0357 - accuracy: 0.9915 - val_loss: 0.0451 - val_accuracy: 0.9885
    Epoch 602/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0349 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 603/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0331 - accuracy: 0.9926 - val_loss: 0.0469 - val_accuracy: 0.9900
    Epoch 604/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0353 - accuracy: 0.9913 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 605/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0339 - accuracy: 0.9931 - val_loss: 0.0497 - val_accuracy: 0.9885
    Epoch 606/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0342 - accuracy: 0.9913 - val_loss: 0.0444 - val_accuracy: 0.9892
    Epoch 607/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0335 - accuracy: 0.9923 - val_loss: 0.0465 - val_accuracy: 0.9892
    Epoch 608/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0365 - accuracy: 0.9900 - val_loss: 0.0471 - val_accuracy: 0.9892
    Epoch 609/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0358 - accuracy: 0.9905 - val_loss: 0.0493 - val_accuracy: 0.9885
    Epoch 610/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0370 - accuracy: 0.9900 - val_loss: 0.0485 - val_accuracy: 0.9885
    Epoch 611/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0370 - accuracy: 0.9902 - val_loss: 0.0460 - val_accuracy: 0.9900
    Epoch 612/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0362 - accuracy: 0.9908 - val_loss: 0.0494 - val_accuracy: 0.9862
    Epoch 613/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0342 - accuracy: 0.9920 - val_loss: 0.0457 - val_accuracy: 0.9877
    Epoch 614/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0332 - accuracy: 0.9928 - val_loss: 0.0437 - val_accuracy: 0.9892
    Epoch 615/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0333 - accuracy: 0.9928 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 616/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0335 - accuracy: 0.9926 - val_loss: 0.0471 - val_accuracy: 0.9892
    Epoch 617/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0346 - accuracy: 0.9926 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 618/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0356 - accuracy: 0.9908 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 619/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9931 - val_loss: 0.0443 - val_accuracy: 0.9900
    Epoch 620/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9933 - val_loss: 0.0468 - val_accuracy: 0.9892
    Epoch 621/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0362 - accuracy: 0.9905 - val_loss: 0.0467 - val_accuracy: 0.9892
    Epoch 622/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0336 - accuracy: 0.9923 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 623/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9928 - val_loss: 0.0465 - val_accuracy: 0.9862
    Epoch 624/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0342 - accuracy: 0.9926 - val_loss: 0.0447 - val_accuracy: 0.9885
    Epoch 625/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0401 - accuracy: 0.9895 - val_loss: 0.0515 - val_accuracy: 0.9862
    Epoch 626/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0426 - accuracy: 0.9872 - val_loss: 0.0519 - val_accuracy: 0.9854
    Epoch 627/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0385 - accuracy: 0.9905 - val_loss: 0.0542 - val_accuracy: 0.9854
    Epoch 628/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0369 - accuracy: 0.9887 - val_loss: 0.0555 - val_accuracy: 0.9854
    Epoch 629/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0365 - accuracy: 0.9905 - val_loss: 0.0468 - val_accuracy: 0.9869
    Epoch 630/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0345 - accuracy: 0.9920 - val_loss: 0.0459 - val_accuracy: 0.9862
    Epoch 631/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0369 - accuracy: 0.9900 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 632/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0346 - accuracy: 0.9920 - val_loss: 0.0456 - val_accuracy: 0.9900
    Epoch 633/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0335 - accuracy: 0.9931 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 634/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0334 - accuracy: 0.9926 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 635/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0333 - accuracy: 0.9915 - val_loss: 0.0444 - val_accuracy: 0.9892
    Epoch 636/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0336 - accuracy: 0.9920 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 637/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0333 - accuracy: 0.9931 - val_loss: 0.0478 - val_accuracy: 0.9862
    Epoch 638/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0345 - accuracy: 0.9923 - val_loss: 0.0475 - val_accuracy: 0.9869
    Epoch 639/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0387 - accuracy: 0.9905 - val_loss: 0.0469 - val_accuracy: 0.9869
    Epoch 640/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0350 - accuracy: 0.9915 - val_loss: 0.0455 - val_accuracy: 0.9900
    Epoch 641/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9928 - val_loss: 0.0449 - val_accuracy: 0.9908
    Epoch 642/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0332 - accuracy: 0.9941 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 643/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0331 - accuracy: 0.9928 - val_loss: 0.0442 - val_accuracy: 0.9892
    Epoch 644/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0329 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 645/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0333 - accuracy: 0.9926 - val_loss: 0.0435 - val_accuracy: 0.9908
    Epoch 646/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0346 - accuracy: 0.9920 - val_loss: 0.0449 - val_accuracy: 0.9877
    Epoch 647/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0384 - accuracy: 0.9915 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 648/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0376 - accuracy: 0.9920 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 649/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0350 - accuracy: 0.9926 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 650/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0345 - accuracy: 0.9926 - val_loss: 0.0459 - val_accuracy: 0.9869
    Epoch 651/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0331 - accuracy: 0.9926 - val_loss: 0.0454 - val_accuracy: 0.9900
    Epoch 652/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0339 - accuracy: 0.9926 - val_loss: 0.0452 - val_accuracy: 0.9908
    Epoch 653/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0341 - accuracy: 0.9923 - val_loss: 0.0442 - val_accuracy: 0.9908
    Epoch 654/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0339 - accuracy: 0.9913 - val_loss: 0.0437 - val_accuracy: 0.9892
    Epoch 655/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0340 - accuracy: 0.9926 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 656/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0357 - accuracy: 0.9918 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 657/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0335 - accuracy: 0.9928 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 658/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0344 - accuracy: 0.9918 - val_loss: 0.0484 - val_accuracy: 0.9892
    Epoch 659/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0358 - accuracy: 0.9913 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 660/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9923 - val_loss: 0.0453 - val_accuracy: 0.9885
    Epoch 661/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0332 - accuracy: 0.9931 - val_loss: 0.0448 - val_accuracy: 0.9885
    Epoch 662/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0329 - accuracy: 0.9928 - val_loss: 0.0442 - val_accuracy: 0.9892
    Epoch 663/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0322 - accuracy: 0.9928 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 664/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0354 - accuracy: 0.9913 - val_loss: 0.0502 - val_accuracy: 0.9877
    Epoch 665/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0356 - accuracy: 0.9913 - val_loss: 0.0501 - val_accuracy: 0.9885
    Epoch 666/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0352 - accuracy: 0.9913 - val_loss: 0.0442 - val_accuracy: 0.9900
    Epoch 667/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0334 - accuracy: 0.9926 - val_loss: 0.0442 - val_accuracy: 0.9908
    Epoch 668/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9920 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 669/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0330 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 670/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0335 - accuracy: 0.9926 - val_loss: 0.0474 - val_accuracy: 0.9869
    Epoch 671/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0358 - accuracy: 0.9905 - val_loss: 0.0502 - val_accuracy: 0.9862
    Epoch 672/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0342 - accuracy: 0.9926 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 673/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0333 - accuracy: 0.9928 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 674/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0328 - accuracy: 0.9926 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 675/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0325 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9900
    Epoch 676/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0331 - accuracy: 0.9928 - val_loss: 0.0521 - val_accuracy: 0.9854
    Epoch 677/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0389 - accuracy: 0.9905 - val_loss: 0.0575 - val_accuracy: 0.9838
    Epoch 678/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9910 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 679/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9926 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 680/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0327 - accuracy: 0.9926 - val_loss: 0.0451 - val_accuracy: 0.9908
    Epoch 681/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0330 - accuracy: 0.9933 - val_loss: 0.0441 - val_accuracy: 0.9900
    Epoch 682/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0328 - accuracy: 0.9928 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 683/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0333 - accuracy: 0.9918 - val_loss: 0.0442 - val_accuracy: 0.9908
    Epoch 684/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0327 - accuracy: 0.9931 - val_loss: 0.0487 - val_accuracy: 0.9892
    Epoch 685/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0354 - accuracy: 0.9908 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 686/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0382 - accuracy: 0.9895 - val_loss: 0.0458 - val_accuracy: 0.9869
    Epoch 687/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0334 - accuracy: 0.9926 - val_loss: 0.0454 - val_accuracy: 0.9885
    Epoch 688/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0328 - accuracy: 0.9926 - val_loss: 0.0496 - val_accuracy: 0.9862
    Epoch 689/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 690/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0329 - accuracy: 0.9913 - val_loss: 0.0434 - val_accuracy: 0.9908
    Epoch 691/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0338 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 692/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0352 - accuracy: 0.9913 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 693/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0330 - accuracy: 0.9928 - val_loss: 0.0456 - val_accuracy: 0.9908
    Epoch 694/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0337 - accuracy: 0.9923 - val_loss: 0.0464 - val_accuracy: 0.9900
    Epoch 695/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0372 - accuracy: 0.9908 - val_loss: 0.0444 - val_accuracy: 0.9892
    Epoch 696/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0331 - accuracy: 0.9936 - val_loss: 0.0508 - val_accuracy: 0.9854
    Epoch 697/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0341 - accuracy: 0.9905 - val_loss: 0.0482 - val_accuracy: 0.9869
    Epoch 698/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0327 - accuracy: 0.9926 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 699/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0321 - accuracy: 0.9928 - val_loss: 0.0441 - val_accuracy: 0.9892
    Epoch 700/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0323 - accuracy: 0.9923 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 701/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0331 - accuracy: 0.9928 - val_loss: 0.0435 - val_accuracy: 0.9908
    Epoch 702/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0334 - accuracy: 0.9918 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 703/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0320 - accuracy: 0.9931 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 704/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0328 - accuracy: 0.9926 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 705/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0349 - accuracy: 0.9918 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 706/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0338 - accuracy: 0.9926 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 707/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0334 - accuracy: 0.9920 - val_loss: 0.0472 - val_accuracy: 0.9862
    Epoch 708/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0327 - accuracy: 0.9933 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 709/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0319 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 710/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0321 - accuracy: 0.9923 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 711/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0330 - accuracy: 0.9926 - val_loss: 0.0441 - val_accuracy: 0.9892
    Epoch 712/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0320 - accuracy: 0.9928 - val_loss: 0.0447 - val_accuracy: 0.9885
    Epoch 713/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0324 - accuracy: 0.9928 - val_loss: 0.0457 - val_accuracy: 0.9908
    Epoch 714/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0362 - accuracy: 0.9908 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 715/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0358 - accuracy: 0.9913 - val_loss: 0.0436 - val_accuracy: 0.9900
    Epoch 716/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0323 - accuracy: 0.9926 - val_loss: 0.0466 - val_accuracy: 0.9877
    Epoch 717/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9910 - val_loss: 0.0564 - val_accuracy: 0.9838
    Epoch 718/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0365 - accuracy: 0.9915 - val_loss: 0.0478 - val_accuracy: 0.9877
    Epoch 719/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0336 - accuracy: 0.9931 - val_loss: 0.0463 - val_accuracy: 0.9869
    Epoch 720/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0324 - accuracy: 0.9915 - val_loss: 0.0480 - val_accuracy: 0.9869
    Epoch 721/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0339 - accuracy: 0.9920 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 722/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0326 - accuracy: 0.9933 - val_loss: 0.0434 - val_accuracy: 0.9892
    Epoch 723/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0318 - accuracy: 0.9933 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 724/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0318 - accuracy: 0.9936 - val_loss: 0.0442 - val_accuracy: 0.9892
    Epoch 725/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0327 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 726/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0328 - accuracy: 0.9920 - val_loss: 0.0446 - val_accuracy: 0.9908
    Epoch 727/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0335 - accuracy: 0.9928 - val_loss: 0.0516 - val_accuracy: 0.9862
    Epoch 728/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0356 - accuracy: 0.9923 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 729/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0318 - accuracy: 0.9923 - val_loss: 0.0440 - val_accuracy: 0.9900
    Epoch 730/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0318 - accuracy: 0.9923 - val_loss: 0.0450 - val_accuracy: 0.9908
    Epoch 731/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0323 - accuracy: 0.9938 - val_loss: 0.0435 - val_accuracy: 0.9892
    Epoch 732/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9928 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 733/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0321 - accuracy: 0.9931 - val_loss: 0.0466 - val_accuracy: 0.9900
    Epoch 734/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0326 - accuracy: 0.9926 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 735/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0355 - accuracy: 0.9915 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 736/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0323 - accuracy: 0.9931 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 737/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0329 - accuracy: 0.9933 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 738/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0336 - accuracy: 0.9931 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 739/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0354 - accuracy: 0.9923 - val_loss: 0.0444 - val_accuracy: 0.9908
    Epoch 740/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0323 - accuracy: 0.9931 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 741/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0314 - accuracy: 0.9928 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 742/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0337 - accuracy: 0.9926 - val_loss: 0.0441 - val_accuracy: 0.9885
    Epoch 743/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0322 - accuracy: 0.9920 - val_loss: 0.0439 - val_accuracy: 0.9892
    Epoch 744/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0320 - accuracy: 0.9931 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 745/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0321 - accuracy: 0.9928 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 746/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0324 - accuracy: 0.9931 - val_loss: 0.0449 - val_accuracy: 0.9877
    Epoch 747/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0319 - accuracy: 0.9933 - val_loss: 0.0471 - val_accuracy: 0.9869
    Epoch 748/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0347 - accuracy: 0.9915 - val_loss: 0.0436 - val_accuracy: 0.9900
    Epoch 749/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0315 - accuracy: 0.9933 - val_loss: 0.0467 - val_accuracy: 0.9869
    Epoch 750/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0321 - accuracy: 0.9933 - val_loss: 0.0448 - val_accuracy: 0.9885
    Epoch 751/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0334 - accuracy: 0.9913 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 752/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0321 - accuracy: 0.9928 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 753/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0333 - accuracy: 0.9918 - val_loss: 0.0441 - val_accuracy: 0.9892
    Epoch 754/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0336 - accuracy: 0.9915 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 755/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0309 - accuracy: 0.9926 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 756/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0322 - accuracy: 0.9920 - val_loss: 0.0518 - val_accuracy: 0.9862
    Epoch 757/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0326 - accuracy: 0.9923 - val_loss: 0.0455 - val_accuracy: 0.9877
    Epoch 758/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0316 - accuracy: 0.9931 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 759/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0325 - accuracy: 0.9920 - val_loss: 0.0493 - val_accuracy: 0.9862
    Epoch 760/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0338 - accuracy: 0.9926 - val_loss: 0.0464 - val_accuracy: 0.9877
    Epoch 761/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0328 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 762/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0338 - accuracy: 0.9920 - val_loss: 0.0531 - val_accuracy: 0.9862
    Epoch 763/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0363 - accuracy: 0.9910 - val_loss: 0.0695 - val_accuracy: 0.9808
    Epoch 764/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0417 - accuracy: 0.9885 - val_loss: 0.0656 - val_accuracy: 0.9815
    Epoch 765/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0390 - accuracy: 0.9913 - val_loss: 0.0498 - val_accuracy: 0.9869
    Epoch 766/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0341 - accuracy: 0.9931 - val_loss: 0.0470 - val_accuracy: 0.9869
    Epoch 767/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0331 - accuracy: 0.9933 - val_loss: 0.0495 - val_accuracy: 0.9862
    Epoch 768/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0323 - accuracy: 0.9936 - val_loss: 0.0470 - val_accuracy: 0.9862
    Epoch 769/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0323 - accuracy: 0.9928 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 770/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0311 - accuracy: 0.9926 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 771/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0309 - accuracy: 0.9926 - val_loss: 0.0450 - val_accuracy: 0.9908
    Epoch 772/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0318 - accuracy: 0.9936 - val_loss: 0.0433 - val_accuracy: 0.9908
    Epoch 773/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9933 - val_loss: 0.0446 - val_accuracy: 0.9885
    Epoch 774/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0319 - accuracy: 0.9931 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 775/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9933 - val_loss: 0.0434 - val_accuracy: 0.9908
    Epoch 776/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0331 - accuracy: 0.9926 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 777/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0313 - accuracy: 0.9931 - val_loss: 0.0450 - val_accuracy: 0.9885
    Epoch 778/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0312 - accuracy: 0.9933 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 779/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0322 - accuracy: 0.9933 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 780/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0311 - accuracy: 0.9931 - val_loss: 0.0433 - val_accuracy: 0.9908
    Epoch 781/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0319 - accuracy: 0.9915 - val_loss: 0.0442 - val_accuracy: 0.9885
    Epoch 782/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0345 - accuracy: 0.9918 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 783/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9918 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 784/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0363 - accuracy: 0.9910 - val_loss: 0.0450 - val_accuracy: 0.9915
    Epoch 785/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0356 - accuracy: 0.9895 - val_loss: 0.0448 - val_accuracy: 0.9908
    Epoch 786/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0313 - accuracy: 0.9933 - val_loss: 0.0434 - val_accuracy: 0.9900
    Epoch 787/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0309 - accuracy: 0.9938 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 788/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9928 - val_loss: 0.0436 - val_accuracy: 0.9900
    Epoch 789/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0312 - accuracy: 0.9931 - val_loss: 0.0442 - val_accuracy: 0.9900
    Epoch 790/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0311 - accuracy: 0.9931 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 791/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0343 - accuracy: 0.9923 - val_loss: 0.0501 - val_accuracy: 0.9862
    Epoch 792/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0317 - accuracy: 0.9933 - val_loss: 0.0449 - val_accuracy: 0.9885
    Epoch 793/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0310 - accuracy: 0.9931 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 794/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0320 - accuracy: 0.9931 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 795/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0323 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 796/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0325 - accuracy: 0.9920 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 797/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0332 - accuracy: 0.9926 - val_loss: 0.0455 - val_accuracy: 0.9885
    Epoch 798/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0328 - accuracy: 0.9920 - val_loss: 0.0453 - val_accuracy: 0.9869
    Epoch 799/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0310 - accuracy: 0.9920 - val_loss: 0.0481 - val_accuracy: 0.9862
    Epoch 800/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0375 - accuracy: 0.9897 - val_loss: 0.0548 - val_accuracy: 0.9846
    Epoch 801/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0428 - accuracy: 0.9890 - val_loss: 0.0458 - val_accuracy: 0.9892
    Epoch 802/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0362 - accuracy: 0.9913 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 803/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0339 - accuracy: 0.9910 - val_loss: 0.0452 - val_accuracy: 0.9915
    Epoch 804/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0348 - accuracy: 0.9923 - val_loss: 0.0467 - val_accuracy: 0.9892
    Epoch 805/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0362 - accuracy: 0.9918 - val_loss: 0.0453 - val_accuracy: 0.9908
    Epoch 806/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0327 - accuracy: 0.9928 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 807/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0306 - accuracy: 0.9938 - val_loss: 0.0457 - val_accuracy: 0.9869
    Epoch 808/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0312 - accuracy: 0.9931 - val_loss: 0.0463 - val_accuracy: 0.9869
    Epoch 809/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0343 - accuracy: 0.9908 - val_loss: 0.0436 - val_accuracy: 0.9900
    Epoch 810/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0349 - accuracy: 0.9913 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 811/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0318 - accuracy: 0.9923 - val_loss: 0.0473 - val_accuracy: 0.9908
    Epoch 812/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0340 - accuracy: 0.9918 - val_loss: 0.0457 - val_accuracy: 0.9908
    Epoch 813/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0319 - accuracy: 0.9923 - val_loss: 0.0446 - val_accuracy: 0.9908
    Epoch 814/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0362 - accuracy: 0.9897 - val_loss: 0.0440 - val_accuracy: 0.9900
    Epoch 815/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0344 - accuracy: 0.9918 - val_loss: 0.0457 - val_accuracy: 0.9908
    Epoch 816/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0333 - accuracy: 0.9933 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 817/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0325 - accuracy: 0.9923 - val_loss: 0.0452 - val_accuracy: 0.9877
    Epoch 818/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0329 - accuracy: 0.9920 - val_loss: 0.0539 - val_accuracy: 0.9854
    Epoch 819/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0332 - accuracy: 0.9913 - val_loss: 0.0493 - val_accuracy: 0.9869
    Epoch 820/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0348 - accuracy: 0.9923 - val_loss: 0.0455 - val_accuracy: 0.9892
    Epoch 821/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0367 - accuracy: 0.9908 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 822/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0325 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 823/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0312 - accuracy: 0.9933 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 824/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0315 - accuracy: 0.9926 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 825/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0304 - accuracy: 0.9938 - val_loss: 0.0464 - val_accuracy: 0.9869
    Epoch 826/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0342 - accuracy: 0.9931 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 827/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0325 - accuracy: 0.9918 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 828/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0324 - accuracy: 0.9928 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 829/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0321 - accuracy: 0.9923 - val_loss: 0.0459 - val_accuracy: 0.9885
    Epoch 830/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0317 - accuracy: 0.9926 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 831/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0305 - accuracy: 0.9936 - val_loss: 0.0444 - val_accuracy: 0.9908
    Epoch 832/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0346 - accuracy: 0.9928 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 833/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0329 - accuracy: 0.9931 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 834/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0315 - accuracy: 0.9920 - val_loss: 0.0453 - val_accuracy: 0.9885
    Epoch 835/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0315 - accuracy: 0.9928 - val_loss: 0.0464 - val_accuracy: 0.9869
    Epoch 836/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0322 - accuracy: 0.9933 - val_loss: 0.0472 - val_accuracy: 0.9869
    Epoch 837/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0306 - accuracy: 0.9936 - val_loss: 0.0470 - val_accuracy: 0.9892
    Epoch 838/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0319 - accuracy: 0.9923 - val_loss: 0.0472 - val_accuracy: 0.9908
    Epoch 839/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0326 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 840/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0311 - accuracy: 0.9928 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 841/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0306 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 842/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0305 - accuracy: 0.9931 - val_loss: 0.0445 - val_accuracy: 0.9908
    Epoch 843/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0310 - accuracy: 0.9926 - val_loss: 0.0444 - val_accuracy: 0.9915
    Epoch 844/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9928 - val_loss: 0.0435 - val_accuracy: 0.9908
    Epoch 845/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0348 - accuracy: 0.9918 - val_loss: 0.0438 - val_accuracy: 0.9892
    Epoch 846/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0363 - accuracy: 0.9897 - val_loss: 0.0436 - val_accuracy: 0.9892
    Epoch 847/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0362 - accuracy: 0.9905 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 848/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0304 - accuracy: 0.9931 - val_loss: 0.0435 - val_accuracy: 0.9892
    Epoch 849/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0307 - accuracy: 0.9931 - val_loss: 0.0452 - val_accuracy: 0.9892
    Epoch 850/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9926 - val_loss: 0.0439 - val_accuracy: 0.9892
    Epoch 851/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0309 - accuracy: 0.9931 - val_loss: 0.0511 - val_accuracy: 0.9862
    Epoch 852/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0324 - accuracy: 0.9933 - val_loss: 0.0466 - val_accuracy: 0.9877
    Epoch 853/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0313 - accuracy: 0.9933 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 854/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0335 - accuracy: 0.9926 - val_loss: 0.0439 - val_accuracy: 0.9892
    Epoch 855/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0312 - accuracy: 0.9918 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 856/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0306 - accuracy: 0.9926 - val_loss: 0.0440 - val_accuracy: 0.9892
    Epoch 857/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0310 - accuracy: 0.9920 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 858/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0307 - accuracy: 0.9926 - val_loss: 0.0446 - val_accuracy: 0.9885
    Epoch 859/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0311 - accuracy: 0.9918 - val_loss: 0.0434 - val_accuracy: 0.9900
    Epoch 860/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0317 - accuracy: 0.9918 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 861/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9931 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 862/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0317 - accuracy: 0.9928 - val_loss: 0.0466 - val_accuracy: 0.9869
    Epoch 863/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0316 - accuracy: 0.9931 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 864/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0298 - accuracy: 0.9938 - val_loss: 0.0434 - val_accuracy: 0.9908
    Epoch 865/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0303 - accuracy: 0.9928 - val_loss: 0.0435 - val_accuracy: 0.9908
    Epoch 866/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0299 - accuracy: 0.9926 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 867/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0323 - accuracy: 0.9931 - val_loss: 0.0451 - val_accuracy: 0.9908
    Epoch 868/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0326 - accuracy: 0.9918 - val_loss: 0.0441 - val_accuracy: 0.9908
    Epoch 869/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0314 - accuracy: 0.9920 - val_loss: 0.0433 - val_accuracy: 0.9900
    Epoch 870/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 871/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0300 - accuracy: 0.9931 - val_loss: 0.0437 - val_accuracy: 0.9900
    Epoch 872/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0298 - accuracy: 0.9936 - val_loss: 0.0434 - val_accuracy: 0.9900
    Epoch 873/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0302 - accuracy: 0.9936 - val_loss: 0.0465 - val_accuracy: 0.9908
    Epoch 874/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0329 - accuracy: 0.9931 - val_loss: 0.0450 - val_accuracy: 0.9908
    Epoch 875/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0323 - accuracy: 0.9920 - val_loss: 0.0440 - val_accuracy: 0.9900
    Epoch 876/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0308 - accuracy: 0.9941 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 877/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0305 - accuracy: 0.9933 - val_loss: 0.0434 - val_accuracy: 0.9908
    Epoch 878/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0317 - accuracy: 0.9931 - val_loss: 0.0431 - val_accuracy: 0.9900
    Epoch 879/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0303 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 880/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0443 - val_accuracy: 0.9885
    Epoch 881/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0312 - accuracy: 0.9923 - val_loss: 0.0466 - val_accuracy: 0.9877
    Epoch 882/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0346 - accuracy: 0.9905 - val_loss: 0.0544 - val_accuracy: 0.9854
    Epoch 883/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0341 - accuracy: 0.9926 - val_loss: 0.0523 - val_accuracy: 0.9862
    Epoch 884/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0326 - accuracy: 0.9920 - val_loss: 0.0548 - val_accuracy: 0.9846
    Epoch 885/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0342 - accuracy: 0.9923 - val_loss: 0.0460 - val_accuracy: 0.9877
    Epoch 886/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0332 - accuracy: 0.9915 - val_loss: 0.0471 - val_accuracy: 0.9869
    Epoch 887/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0332 - accuracy: 0.9920 - val_loss: 0.0485 - val_accuracy: 0.9869
    Epoch 888/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0327 - accuracy: 0.9920 - val_loss: 0.0497 - val_accuracy: 0.9869
    Epoch 889/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0328 - accuracy: 0.9923 - val_loss: 0.0444 - val_accuracy: 0.9885
    Epoch 890/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0319 - accuracy: 0.9923 - val_loss: 0.0432 - val_accuracy: 0.9900
    Epoch 891/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0315 - accuracy: 0.9926 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 892/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0337 - accuracy: 0.9915 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 893/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0347 - accuracy: 0.9918 - val_loss: 0.0438 - val_accuracy: 0.9900
    Epoch 894/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0314 - accuracy: 0.9915 - val_loss: 0.0436 - val_accuracy: 0.9892
    Epoch 895/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0321 - accuracy: 0.9923 - val_loss: 0.0437 - val_accuracy: 0.9892
    Epoch 896/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0299 - accuracy: 0.9928 - val_loss: 0.0445 - val_accuracy: 0.9885
    Epoch 897/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0433 - val_accuracy: 0.9900
    Epoch 898/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0295 - accuracy: 0.9936 - val_loss: 0.0443 - val_accuracy: 0.9915
    Epoch 899/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0299 - accuracy: 0.9926 - val_loss: 0.0431 - val_accuracy: 0.9892
    Epoch 900/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0295 - accuracy: 0.9928 - val_loss: 0.0436 - val_accuracy: 0.9892
    Epoch 901/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0304 - accuracy: 0.9936 - val_loss: 0.0554 - val_accuracy: 0.9854
    Epoch 902/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0366 - accuracy: 0.9913 - val_loss: 0.0620 - val_accuracy: 0.9831
    Epoch 903/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0368 - accuracy: 0.9900 - val_loss: 0.0525 - val_accuracy: 0.9869
    Epoch 904/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0439 - accuracy: 0.9874 - val_loss: 0.0657 - val_accuracy: 0.9831
    Epoch 905/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0444 - accuracy: 0.9864 - val_loss: 0.0477 - val_accuracy: 0.9877
    Epoch 906/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0338 - accuracy: 0.9931 - val_loss: 0.0437 - val_accuracy: 0.9908
    Epoch 907/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0317 - accuracy: 0.9920 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 908/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0327 - accuracy: 0.9931 - val_loss: 0.0432 - val_accuracy: 0.9915
    Epoch 909/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0310 - accuracy: 0.9933 - val_loss: 0.0465 - val_accuracy: 0.9900
    Epoch 910/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0313 - accuracy: 0.9928 - val_loss: 0.0453 - val_accuracy: 0.9900
    Epoch 911/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0327 - accuracy: 0.9905 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 912/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0311 - accuracy: 0.9933 - val_loss: 0.0468 - val_accuracy: 0.9900
    Epoch 913/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0313 - accuracy: 0.9926 - val_loss: 0.0433 - val_accuracy: 0.9900
    Epoch 914/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0308 - accuracy: 0.9928 - val_loss: 0.0437 - val_accuracy: 0.9892
    Epoch 915/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0292 - accuracy: 0.9936 - val_loss: 0.0449 - val_accuracy: 0.9885
    Epoch 916/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0305 - accuracy: 0.9926 - val_loss: 0.0436 - val_accuracy: 0.9892
    Epoch 917/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0301 - accuracy: 0.9926 - val_loss: 0.0431 - val_accuracy: 0.9908
    Epoch 918/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0294 - accuracy: 0.9938 - val_loss: 0.0467 - val_accuracy: 0.9869
    Epoch 919/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0298 - accuracy: 0.9936 - val_loss: 0.0439 - val_accuracy: 0.9915
    Epoch 920/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0298 - accuracy: 0.9936 - val_loss: 0.0432 - val_accuracy: 0.9892
    Epoch 921/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0294 - accuracy: 0.9933 - val_loss: 0.0430 - val_accuracy: 0.9900
    Epoch 922/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9931 - val_loss: 0.0428 - val_accuracy: 0.9892
    Epoch 923/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9920 - val_loss: 0.0487 - val_accuracy: 0.9869
    Epoch 924/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0343 - accuracy: 0.9923 - val_loss: 0.0528 - val_accuracy: 0.9869
    Epoch 925/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0319 - accuracy: 0.9931 - val_loss: 0.0481 - val_accuracy: 0.9869
    Epoch 926/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0304 - accuracy: 0.9923 - val_loss: 0.0429 - val_accuracy: 0.9900
    Epoch 927/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0292 - accuracy: 0.9938 - val_loss: 0.0429 - val_accuracy: 0.9908
    Epoch 928/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0294 - accuracy: 0.9931 - val_loss: 0.0437 - val_accuracy: 0.9892
    Epoch 929/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0294 - accuracy: 0.9928 - val_loss: 0.0442 - val_accuracy: 0.9915
    Epoch 930/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0310 - accuracy: 0.9933 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 931/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0384 - accuracy: 0.9902 - val_loss: 0.0589 - val_accuracy: 0.9846
    Epoch 932/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0348 - accuracy: 0.9920 - val_loss: 0.0453 - val_accuracy: 0.9892
    Epoch 933/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0302 - accuracy: 0.9931 - val_loss: 0.0435 - val_accuracy: 0.9892
    Epoch 934/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0301 - accuracy: 0.9931 - val_loss: 0.0430 - val_accuracy: 0.9900
    Epoch 935/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0432 - val_accuracy: 0.9892
    Epoch 936/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0297 - accuracy: 0.9931 - val_loss: 0.0441 - val_accuracy: 0.9885
    Epoch 937/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0314 - accuracy: 0.9933 - val_loss: 0.0434 - val_accuracy: 0.9892
    Epoch 938/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0375 - accuracy: 0.9887 - val_loss: 0.0479 - val_accuracy: 0.9869
    Epoch 939/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0392 - accuracy: 0.9882 - val_loss: 0.0439 - val_accuracy: 0.9900
    Epoch 940/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0328 - accuracy: 0.9920 - val_loss: 0.0432 - val_accuracy: 0.9900
    Epoch 941/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0318 - accuracy: 0.9933 - val_loss: 0.0440 - val_accuracy: 0.9908
    Epoch 942/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0305 - accuracy: 0.9923 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 943/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0307 - accuracy: 0.9920 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 944/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0317 - accuracy: 0.9928 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 945/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0293 - accuracy: 0.9936 - val_loss: 0.0431 - val_accuracy: 0.9908
    Epoch 946/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0296 - accuracy: 0.9936 - val_loss: 0.0427 - val_accuracy: 0.9908
    Epoch 947/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0304 - accuracy: 0.9928 - val_loss: 0.0426 - val_accuracy: 0.9908
    Epoch 948/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0341 - accuracy: 0.9900 - val_loss: 0.0444 - val_accuracy: 0.9915
    Epoch 949/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0323 - accuracy: 0.9926 - val_loss: 0.0474 - val_accuracy: 0.9885
    Epoch 950/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0300 - accuracy: 0.9936 - val_loss: 0.0433 - val_accuracy: 0.9908
    Epoch 951/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0300 - accuracy: 0.9936 - val_loss: 0.0431 - val_accuracy: 0.9885
    Epoch 952/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9923 - val_loss: 0.0436 - val_accuracy: 0.9885
    Epoch 953/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0306 - accuracy: 0.9933 - val_loss: 0.0429 - val_accuracy: 0.9892
    Epoch 954/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0299 - accuracy: 0.9920 - val_loss: 0.0424 - val_accuracy: 0.9908
    Epoch 955/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0298 - accuracy: 0.9928 - val_loss: 0.0432 - val_accuracy: 0.9900
    Epoch 956/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0348 - accuracy: 0.9915 - val_loss: 0.0514 - val_accuracy: 0.9862
    Epoch 957/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0330 - accuracy: 0.9923 - val_loss: 0.0465 - val_accuracy: 0.9869
    Epoch 958/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0310 - accuracy: 0.9931 - val_loss: 0.0442 - val_accuracy: 0.9885
    Epoch 959/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0302 - accuracy: 0.9936 - val_loss: 0.0435 - val_accuracy: 0.9900
    Epoch 960/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0292 - accuracy: 0.9944 - val_loss: 0.0426 - val_accuracy: 0.9908
    Epoch 961/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0301 - accuracy: 0.9931 - val_loss: 0.0426 - val_accuracy: 0.9915
    Epoch 962/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0298 - accuracy: 0.9933 - val_loss: 0.0588 - val_accuracy: 0.9838
    Epoch 963/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0365 - accuracy: 0.9908 - val_loss: 0.0542 - val_accuracy: 0.9846
    Epoch 964/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0334 - accuracy: 0.9915 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 965/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0302 - accuracy: 0.9928 - val_loss: 0.0454 - val_accuracy: 0.9900
    Epoch 966/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0299 - accuracy: 0.9936 - val_loss: 0.0428 - val_accuracy: 0.9908
    Epoch 967/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0315 - accuracy: 0.9926 - val_loss: 0.0457 - val_accuracy: 0.9877
    Epoch 968/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9928 - val_loss: 0.0429 - val_accuracy: 0.9900
    Epoch 969/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0313 - accuracy: 0.9931 - val_loss: 0.0425 - val_accuracy: 0.9900
    Epoch 970/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0288 - accuracy: 0.9931 - val_loss: 0.0429 - val_accuracy: 0.9900
    Epoch 971/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0340 - accuracy: 0.9902 - val_loss: 0.0427 - val_accuracy: 0.9908
    Epoch 972/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0373 - accuracy: 0.9908 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 973/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0468 - val_accuracy: 0.9892
    Epoch 974/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0318 - accuracy: 0.9923 - val_loss: 0.0512 - val_accuracy: 0.9869
    Epoch 975/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9913 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 976/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0299 - accuracy: 0.9933 - val_loss: 0.0442 - val_accuracy: 0.9900
    Epoch 977/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0298 - accuracy: 0.9933 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 978/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0303 - accuracy: 0.9938 - val_loss: 0.0471 - val_accuracy: 0.9877
    Epoch 979/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0305 - accuracy: 0.9928 - val_loss: 0.0492 - val_accuracy: 0.9869
    Epoch 980/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0311 - accuracy: 0.9928 - val_loss: 0.0439 - val_accuracy: 0.9908
    Epoch 981/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0282 - accuracy: 0.9936 - val_loss: 0.0433 - val_accuracy: 0.9892
    Epoch 982/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0289 - accuracy: 0.9933 - val_loss: 0.0424 - val_accuracy: 0.9908
    Epoch 983/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0290 - accuracy: 0.9926 - val_loss: 0.0437 - val_accuracy: 0.9885
    Epoch 984/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0290 - accuracy: 0.9928 - val_loss: 0.0425 - val_accuracy: 0.9908
    Epoch 985/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0342 - accuracy: 0.9915 - val_loss: 0.0459 - val_accuracy: 0.9892
    Epoch 986/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0309 - accuracy: 0.9928 - val_loss: 0.0458 - val_accuracy: 0.9892
    Epoch 987/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0297 - accuracy: 0.9933 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 988/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0306 - accuracy: 0.9931 - val_loss: 0.0433 - val_accuracy: 0.9900
    Epoch 989/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0299 - accuracy: 0.9941 - val_loss: 0.0438 - val_accuracy: 0.9885
    Epoch 990/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0299 - accuracy: 0.9936 - val_loss: 0.0436 - val_accuracy: 0.9908
    Epoch 991/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0291 - accuracy: 0.9933 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 992/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0284 - accuracy: 0.9938 - val_loss: 0.0527 - val_accuracy: 0.9869
    Epoch 993/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0321 - accuracy: 0.9913 - val_loss: 0.0581 - val_accuracy: 0.9846
    Epoch 994/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0333 - accuracy: 0.9908 - val_loss: 0.0487 - val_accuracy: 0.9885
    Epoch 995/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0297 - accuracy: 0.9920 - val_loss: 0.0544 - val_accuracy: 0.9862
    Epoch 996/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0310 - accuracy: 0.9928 - val_loss: 0.0503 - val_accuracy: 0.9877
    Epoch 997/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9920 - val_loss: 0.0446 - val_accuracy: 0.9908
    Epoch 998/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0287 - accuracy: 0.9928 - val_loss: 0.0438 - val_accuracy: 0.9908
    Epoch 999/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0282 - accuracy: 0.9936 - val_loss: 0.0442 - val_accuracy: 0.9885
    Epoch 1000/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0284 - accuracy: 0.9928 - val_loss: 0.0458 - val_accuracy: 0.9885
    Epoch 1001/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0322 - accuracy: 0.9918 - val_loss: 0.0458 - val_accuracy: 0.9900
    Epoch 1002/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0313 - accuracy: 0.9923 - val_loss: 0.0448 - val_accuracy: 0.9900
    Epoch 1003/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0314 - accuracy: 0.9918 - val_loss: 0.0499 - val_accuracy: 0.9877
    Epoch 1004/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0356 - accuracy: 0.9895 - val_loss: 0.0674 - val_accuracy: 0.9846
    Epoch 1005/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0347 - accuracy: 0.9902 - val_loss: 0.0522 - val_accuracy: 0.9869
    Epoch 1006/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0304 - accuracy: 0.9923 - val_loss: 0.0583 - val_accuracy: 0.9854
    Epoch 1007/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0346 - accuracy: 0.9918 - val_loss: 0.0566 - val_accuracy: 0.9869
    Epoch 1008/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0336 - accuracy: 0.9923 - val_loss: 0.0489 - val_accuracy: 0.9885
    Epoch 1009/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0369 - accuracy: 0.9895 - val_loss: 0.0492 - val_accuracy: 0.9885
    Epoch 1010/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0384 - accuracy: 0.9895 - val_loss: 0.0435 - val_accuracy: 0.9892
    Epoch 1011/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0335 - accuracy: 0.9918 - val_loss: 0.0478 - val_accuracy: 0.9885
    Epoch 1012/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0315 - accuracy: 0.9931 - val_loss: 0.0465 - val_accuracy: 0.9892
    Epoch 1013/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0299 - accuracy: 0.9941 - val_loss: 0.0469 - val_accuracy: 0.9892
    Epoch 1014/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0315 - accuracy: 0.9923 - val_loss: 0.0453 - val_accuracy: 0.9908
    Epoch 1015/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0285 - accuracy: 0.9936 - val_loss: 0.0444 - val_accuracy: 0.9885
    Epoch 1016/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0278 - accuracy: 0.9938 - val_loss: 0.0447 - val_accuracy: 0.9915
    Epoch 1017/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0284 - accuracy: 0.9936 - val_loss: 0.0444 - val_accuracy: 0.9900
    Epoch 1018/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0285 - accuracy: 0.9928 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1019/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0294 - accuracy: 0.9933 - val_loss: 0.0464 - val_accuracy: 0.9915
    Epoch 1020/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0284 - accuracy: 0.9931 - val_loss: 0.0466 - val_accuracy: 0.9908
    Epoch 1021/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0294 - accuracy: 0.9936 - val_loss: 0.0466 - val_accuracy: 0.9885
    Epoch 1022/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0287 - accuracy: 0.9928 - val_loss: 0.0521 - val_accuracy: 0.9869
    Epoch 1023/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0292 - accuracy: 0.9931 - val_loss: 0.0446 - val_accuracy: 0.9892
    Epoch 1024/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0273 - accuracy: 0.9944 - val_loss: 0.0448 - val_accuracy: 0.9900
    Epoch 1025/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0280 - accuracy: 0.9946 - val_loss: 0.0458 - val_accuracy: 0.9885
    Epoch 1026/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0282 - accuracy: 0.9941 - val_loss: 0.0494 - val_accuracy: 0.9885
    Epoch 1027/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0284 - accuracy: 0.9936 - val_loss: 0.0455 - val_accuracy: 0.9908
    Epoch 1028/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0278 - accuracy: 0.9938 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 1029/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0294 - accuracy: 0.9936 - val_loss: 0.0477 - val_accuracy: 0.9892
    Epoch 1030/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0276 - accuracy: 0.9944 - val_loss: 0.0457 - val_accuracy: 0.9915
    Epoch 1031/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0274 - accuracy: 0.9936 - val_loss: 0.0448 - val_accuracy: 0.9900
    Epoch 1032/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0279 - accuracy: 0.9941 - val_loss: 0.0457 - val_accuracy: 0.9892
    Epoch 1033/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0300 - accuracy: 0.9928 - val_loss: 0.0441 - val_accuracy: 0.9900
    Epoch 1034/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0313 - accuracy: 0.9920 - val_loss: 0.0447 - val_accuracy: 0.9900
    Epoch 1035/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0305 - accuracy: 0.9926 - val_loss: 0.0457 - val_accuracy: 0.9900
    Epoch 1036/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0285 - accuracy: 0.9941 - val_loss: 0.0455 - val_accuracy: 0.9915
    Epoch 1037/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0286 - accuracy: 0.9933 - val_loss: 0.0449 - val_accuracy: 0.9908
    Epoch 1038/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0295 - accuracy: 0.9941 - val_loss: 0.0463 - val_accuracy: 0.9892
    Epoch 1039/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0279 - accuracy: 0.9941 - val_loss: 0.0498 - val_accuracy: 0.9869
    Epoch 1040/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0291 - accuracy: 0.9923 - val_loss: 0.0565 - val_accuracy: 0.9862
    Epoch 1041/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0353 - accuracy: 0.9902 - val_loss: 0.0502 - val_accuracy: 0.9877
    Epoch 1042/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0310 - accuracy: 0.9918 - val_loss: 0.0471 - val_accuracy: 0.9892
    Epoch 1043/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0286 - accuracy: 0.9938 - val_loss: 0.0456 - val_accuracy: 0.9885
    Epoch 1044/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0277 - accuracy: 0.9931 - val_loss: 0.0445 - val_accuracy: 0.9892
    Epoch 1045/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0272 - accuracy: 0.9941 - val_loss: 0.0461 - val_accuracy: 0.9908
    Epoch 1046/2000
    8/8 [==============================] - 0s 30ms/step - loss: 0.0280 - accuracy: 0.9933 - val_loss: 0.0456 - val_accuracy: 0.9892
    Epoch 1047/2000
    8/8 [==============================] - 0s 34ms/step - loss: 0.0288 - accuracy: 0.9931 - val_loss: 0.0462 - val_accuracy: 0.9908
    Epoch 1048/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0280 - accuracy: 0.9941 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 1049/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0299 - accuracy: 0.9923 - val_loss: 0.0443 - val_accuracy: 0.9892
    Epoch 1050/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0275 - accuracy: 0.9938 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 1051/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0286 - accuracy: 0.9931 - val_loss: 0.0502 - val_accuracy: 0.9885
    Epoch 1052/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0302 - accuracy: 0.9931 - val_loss: 0.0509 - val_accuracy: 0.9877
    Epoch 1053/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0305 - accuracy: 0.9923 - val_loss: 0.0550 - val_accuracy: 0.9862
    Epoch 1054/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0321 - accuracy: 0.9926 - val_loss: 0.0537 - val_accuracy: 0.9869
    Epoch 1055/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0312 - accuracy: 0.9926 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1056/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0312 - accuracy: 0.9926 - val_loss: 0.0455 - val_accuracy: 0.9915
    Epoch 1057/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0278 - accuracy: 0.9938 - val_loss: 0.0443 - val_accuracy: 0.9908
    Epoch 1058/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0270 - accuracy: 0.9941 - val_loss: 0.0455 - val_accuracy: 0.9885
    Epoch 1059/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0271 - accuracy: 0.9938 - val_loss: 0.0462 - val_accuracy: 0.9885
    Epoch 1060/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0277 - accuracy: 0.9936 - val_loss: 0.0457 - val_accuracy: 0.9892
    Epoch 1061/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0272 - accuracy: 0.9944 - val_loss: 0.0458 - val_accuracy: 0.9892
    Epoch 1062/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0266 - accuracy: 0.9941 - val_loss: 0.0449 - val_accuracy: 0.9892
    Epoch 1063/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0270 - accuracy: 0.9944 - val_loss: 0.0502 - val_accuracy: 0.9877
    Epoch 1064/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0283 - accuracy: 0.9938 - val_loss: 0.0458 - val_accuracy: 0.9908
    Epoch 1065/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0274 - accuracy: 0.9928 - val_loss: 0.0459 - val_accuracy: 0.9892
    Epoch 1066/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0271 - accuracy: 0.9936 - val_loss: 0.0461 - val_accuracy: 0.9885
    Epoch 1067/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0267 - accuracy: 0.9949 - val_loss: 0.0454 - val_accuracy: 0.9908
    Epoch 1068/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0282 - accuracy: 0.9933 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 1069/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0310 - accuracy: 0.9913 - val_loss: 0.0523 - val_accuracy: 0.9869
    Epoch 1070/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0302 - accuracy: 0.9926 - val_loss: 0.0456 - val_accuracy: 0.9900
    Epoch 1071/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0279 - accuracy: 0.9923 - val_loss: 0.0482 - val_accuracy: 0.9892
    Epoch 1072/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0279 - accuracy: 0.9936 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1073/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0272 - accuracy: 0.9941 - val_loss: 0.0458 - val_accuracy: 0.9900
    Epoch 1074/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0266 - accuracy: 0.9941 - val_loss: 0.0454 - val_accuracy: 0.9900
    Epoch 1075/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0268 - accuracy: 0.9936 - val_loss: 0.0468 - val_accuracy: 0.9885
    Epoch 1076/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0271 - accuracy: 0.9936 - val_loss: 0.0458 - val_accuracy: 0.9892
    Epoch 1077/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0271 - accuracy: 0.9936 - val_loss: 0.0456 - val_accuracy: 0.9892
    Epoch 1078/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0266 - accuracy: 0.9938 - val_loss: 0.0467 - val_accuracy: 0.9892
    Epoch 1079/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0268 - accuracy: 0.9946 - val_loss: 0.0457 - val_accuracy: 0.9900
    Epoch 1080/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0267 - accuracy: 0.9941 - val_loss: 0.0462 - val_accuracy: 0.9908
    Epoch 1081/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0267 - accuracy: 0.9938 - val_loss: 0.0454 - val_accuracy: 0.9915
    Epoch 1082/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0278 - accuracy: 0.9938 - val_loss: 0.0576 - val_accuracy: 0.9862
    Epoch 1083/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0322 - accuracy: 0.9923 - val_loss: 0.0534 - val_accuracy: 0.9877
    Epoch 1084/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0300 - accuracy: 0.9933 - val_loss: 0.0457 - val_accuracy: 0.9892
    Epoch 1085/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0274 - accuracy: 0.9941 - val_loss: 0.0453 - val_accuracy: 0.9900
    Epoch 1086/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0284 - accuracy: 0.9928 - val_loss: 0.0467 - val_accuracy: 0.9892
    Epoch 1087/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0274 - accuracy: 0.9941 - val_loss: 0.0456 - val_accuracy: 0.9892
    Epoch 1088/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0274 - accuracy: 0.9941 - val_loss: 0.0490 - val_accuracy: 0.9885
    Epoch 1089/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0309 - accuracy: 0.9920 - val_loss: 0.0488 - val_accuracy: 0.9892
    Epoch 1090/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0292 - accuracy: 0.9938 - val_loss: 0.0452 - val_accuracy: 0.9885
    Epoch 1091/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0270 - accuracy: 0.9936 - val_loss: 0.0464 - val_accuracy: 0.9885
    Epoch 1092/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0268 - accuracy: 0.9941 - val_loss: 0.0463 - val_accuracy: 0.9885
    Epoch 1093/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0282 - accuracy: 0.9933 - val_loss: 0.0507 - val_accuracy: 0.9885
    Epoch 1094/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0280 - accuracy: 0.9938 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1095/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0275 - accuracy: 0.9951 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1096/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0282 - accuracy: 0.9923 - val_loss: 0.0486 - val_accuracy: 0.9885
    Epoch 1097/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0290 - accuracy: 0.9936 - val_loss: 0.0489 - val_accuracy: 0.9892
    Epoch 1098/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0318 - accuracy: 0.9918 - val_loss: 0.0686 - val_accuracy: 0.9831
    Epoch 1099/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0357 - accuracy: 0.9902 - val_loss: 0.0533 - val_accuracy: 0.9877
    Epoch 1100/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0285 - accuracy: 0.9933 - val_loss: 0.0455 - val_accuracy: 0.9885
    Epoch 1101/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0280 - accuracy: 0.9933 - val_loss: 0.0458 - val_accuracy: 0.9885
    Epoch 1102/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0269 - accuracy: 0.9944 - val_loss: 0.0460 - val_accuracy: 0.9900
    Epoch 1103/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0263 - accuracy: 0.9944 - val_loss: 0.0456 - val_accuracy: 0.9908
    Epoch 1104/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0279 - accuracy: 0.9936 - val_loss: 0.0476 - val_accuracy: 0.9892
    Epoch 1105/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0263 - accuracy: 0.9946 - val_loss: 0.0469 - val_accuracy: 0.9892
    Epoch 1106/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0263 - accuracy: 0.9944 - val_loss: 0.0465 - val_accuracy: 0.9892
    Epoch 1107/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0276 - accuracy: 0.9933 - val_loss: 0.0463 - val_accuracy: 0.9908
    Epoch 1108/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0274 - accuracy: 0.9928 - val_loss: 0.0473 - val_accuracy: 0.9908
    Epoch 1109/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0263 - accuracy: 0.9946 - val_loss: 0.0456 - val_accuracy: 0.9900
    Epoch 1110/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0265 - accuracy: 0.9954 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1111/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0264 - accuracy: 0.9946 - val_loss: 0.0471 - val_accuracy: 0.9892
    Epoch 1112/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0262 - accuracy: 0.9938 - val_loss: 0.0458 - val_accuracy: 0.9908
    Epoch 1113/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0263 - accuracy: 0.9944 - val_loss: 0.0464 - val_accuracy: 0.9892
    Epoch 1114/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0263 - accuracy: 0.9941 - val_loss: 0.0474 - val_accuracy: 0.9885
    Epoch 1115/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0265 - accuracy: 0.9944 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1116/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0277 - accuracy: 0.9923 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1117/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0272 - accuracy: 0.9928 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1118/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0254 - accuracy: 0.9949 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1119/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0283 - accuracy: 0.9931 - val_loss: 0.0531 - val_accuracy: 0.9877
    Epoch 1120/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0283 - accuracy: 0.9944 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 1121/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0261 - accuracy: 0.9944 - val_loss: 0.0460 - val_accuracy: 0.9892
    Epoch 1122/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0263 - accuracy: 0.9946 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1123/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0284 - accuracy: 0.9928 - val_loss: 0.0513 - val_accuracy: 0.9885
    Epoch 1124/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0278 - accuracy: 0.9936 - val_loss: 0.0532 - val_accuracy: 0.9885
    Epoch 1125/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0291 - accuracy: 0.9933 - val_loss: 0.0484 - val_accuracy: 0.9885
    Epoch 1126/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0273 - accuracy: 0.9933 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1127/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0282 - accuracy: 0.9926 - val_loss: 0.0476 - val_accuracy: 0.9892
    Epoch 1128/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0286 - accuracy: 0.9931 - val_loss: 0.0467 - val_accuracy: 0.9900
    Epoch 1129/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0252 - accuracy: 0.9949 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1130/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0291 - accuracy: 0.9923 - val_loss: 0.0533 - val_accuracy: 0.9885
    Epoch 1131/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0276 - accuracy: 0.9938 - val_loss: 0.0466 - val_accuracy: 0.9892
    Epoch 1132/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0256 - accuracy: 0.9944 - val_loss: 0.0482 - val_accuracy: 0.9892
    Epoch 1133/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0269 - accuracy: 0.9936 - val_loss: 0.0463 - val_accuracy: 0.9900
    Epoch 1134/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0258 - accuracy: 0.9946 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1135/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0257 - accuracy: 0.9946 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1136/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0256 - accuracy: 0.9946 - val_loss: 0.0475 - val_accuracy: 0.9915
    Epoch 1137/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0259 - accuracy: 0.9946 - val_loss: 0.0472 - val_accuracy: 0.9915
    Epoch 1138/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0271 - accuracy: 0.9938 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1139/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0275 - accuracy: 0.9946 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1140/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0251 - accuracy: 0.9941 - val_loss: 0.0484 - val_accuracy: 0.9892
    Epoch 1141/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0272 - accuracy: 0.9938 - val_loss: 0.0487 - val_accuracy: 0.9892
    Epoch 1142/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0261 - accuracy: 0.9938 - val_loss: 0.0465 - val_accuracy: 0.9900
    Epoch 1143/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0259 - accuracy: 0.9949 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1144/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0257 - accuracy: 0.9938 - val_loss: 0.0473 - val_accuracy: 0.9900
    Epoch 1145/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0252 - accuracy: 0.9954 - val_loss: 0.0470 - val_accuracy: 0.9900
    Epoch 1146/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0248 - accuracy: 0.9954 - val_loss: 0.0490 - val_accuracy: 0.9892
    Epoch 1147/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0253 - accuracy: 0.9944 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1148/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0250 - accuracy: 0.9951 - val_loss: 0.0482 - val_accuracy: 0.9908
    Epoch 1149/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0252 - accuracy: 0.9946 - val_loss: 0.0463 - val_accuracy: 0.9900
    Epoch 1150/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0251 - accuracy: 0.9951 - val_loss: 0.0476 - val_accuracy: 0.9908
    Epoch 1151/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0250 - accuracy: 0.9949 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1152/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0269 - accuracy: 0.9933 - val_loss: 0.0527 - val_accuracy: 0.9892
    Epoch 1153/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0286 - accuracy: 0.9933 - val_loss: 0.0502 - val_accuracy: 0.9892
    Epoch 1154/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0270 - accuracy: 0.9926 - val_loss: 0.0477 - val_accuracy: 0.9892
    Epoch 1155/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0281 - accuracy: 0.9944 - val_loss: 0.0480 - val_accuracy: 0.9892
    Epoch 1156/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0291 - accuracy: 0.9931 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1157/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0260 - accuracy: 0.9946 - val_loss: 0.0465 - val_accuracy: 0.9900
    Epoch 1158/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0259 - accuracy: 0.9951 - val_loss: 0.0469 - val_accuracy: 0.9908
    Epoch 1159/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0262 - accuracy: 0.9938 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1160/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0316 - accuracy: 0.9915 - val_loss: 0.0496 - val_accuracy: 0.9877
    Epoch 1161/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0306 - accuracy: 0.9923 - val_loss: 0.0484 - val_accuracy: 0.9892
    Epoch 1162/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0253 - accuracy: 0.9954 - val_loss: 0.0475 - val_accuracy: 0.9915
    Epoch 1163/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0304 - accuracy: 0.9923 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1164/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0297 - accuracy: 0.9928 - val_loss: 0.0483 - val_accuracy: 0.9892
    Epoch 1165/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0266 - accuracy: 0.9933 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1166/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0272 - accuracy: 0.9931 - val_loss: 0.0551 - val_accuracy: 0.9869
    Epoch 1167/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0306 - accuracy: 0.9928 - val_loss: 0.0588 - val_accuracy: 0.9869
    Epoch 1168/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0333 - accuracy: 0.9910 - val_loss: 0.0649 - val_accuracy: 0.9846
    Epoch 1169/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0371 - accuracy: 0.9887 - val_loss: 0.0511 - val_accuracy: 0.9885
    Epoch 1170/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0298 - accuracy: 0.9923 - val_loss: 0.0688 - val_accuracy: 0.9831
    Epoch 1171/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0316 - accuracy: 0.9900 - val_loss: 0.0630 - val_accuracy: 0.9846
    Epoch 1172/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0334 - accuracy: 0.9918 - val_loss: 0.0520 - val_accuracy: 0.9900
    Epoch 1173/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0284 - accuracy: 0.9926 - val_loss: 0.0465 - val_accuracy: 0.9908
    Epoch 1174/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0254 - accuracy: 0.9941 - val_loss: 0.0479 - val_accuracy: 0.9892
    Epoch 1175/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0255 - accuracy: 0.9954 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1176/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0249 - accuracy: 0.9944 - val_loss: 0.0465 - val_accuracy: 0.9908
    Epoch 1177/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0260 - accuracy: 0.9938 - val_loss: 0.0477 - val_accuracy: 0.9908
    Epoch 1178/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0260 - accuracy: 0.9949 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1179/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0251 - accuracy: 0.9941 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1180/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0252 - accuracy: 0.9941 - val_loss: 0.0477 - val_accuracy: 0.9908
    Epoch 1181/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0251 - accuracy: 0.9946 - val_loss: 0.0467 - val_accuracy: 0.9908
    Epoch 1182/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0244 - accuracy: 0.9951 - val_loss: 0.0471 - val_accuracy: 0.9908
    Epoch 1183/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9949 - val_loss: 0.0486 - val_accuracy: 0.9908
    Epoch 1184/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0255 - accuracy: 0.9944 - val_loss: 0.0492 - val_accuracy: 0.9908
    Epoch 1185/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0250 - accuracy: 0.9944 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1186/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0255 - accuracy: 0.9944 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1187/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0258 - accuracy: 0.9944 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1188/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9951 - val_loss: 0.0469 - val_accuracy: 0.9900
    Epoch 1189/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9949 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1190/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0247 - accuracy: 0.9954 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1191/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0255 - accuracy: 0.9946 - val_loss: 0.0451 - val_accuracy: 0.9915
    Epoch 1192/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0268 - accuracy: 0.9941 - val_loss: 0.0456 - val_accuracy: 0.9915
    Epoch 1193/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0254 - accuracy: 0.9944 - val_loss: 0.0459 - val_accuracy: 0.9900
    Epoch 1194/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0252 - accuracy: 0.9941 - val_loss: 0.0484 - val_accuracy: 0.9908
    Epoch 1195/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0282 - accuracy: 0.9946 - val_loss: 0.0611 - val_accuracy: 0.9862
    Epoch 1196/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0388 - accuracy: 0.9900 - val_loss: 0.0654 - val_accuracy: 0.9838
    Epoch 1197/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0321 - accuracy: 0.9915 - val_loss: 0.0528 - val_accuracy: 0.9892
    Epoch 1198/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0264 - accuracy: 0.9931 - val_loss: 0.0566 - val_accuracy: 0.9885
    Epoch 1199/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0310 - accuracy: 0.9918 - val_loss: 0.0545 - val_accuracy: 0.9892
    Epoch 1200/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0286 - accuracy: 0.9941 - val_loss: 0.0578 - val_accuracy: 0.9869
    Epoch 1201/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0305 - accuracy: 0.9910 - val_loss: 0.0516 - val_accuracy: 0.9885
    Epoch 1202/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0324 - accuracy: 0.9913 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1203/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9951 - val_loss: 0.0480 - val_accuracy: 0.9908
    Epoch 1204/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0249 - accuracy: 0.9946 - val_loss: 0.0458 - val_accuracy: 0.9908
    Epoch 1205/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0250 - accuracy: 0.9946 - val_loss: 0.0491 - val_accuracy: 0.9908
    Epoch 1206/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0252 - accuracy: 0.9949 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1207/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0278 - accuracy: 0.9936 - val_loss: 0.0525 - val_accuracy: 0.9892
    Epoch 1208/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0318 - accuracy: 0.9920 - val_loss: 0.0611 - val_accuracy: 0.9862
    Epoch 1209/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0338 - accuracy: 0.9923 - val_loss: 0.0542 - val_accuracy: 0.9885
    Epoch 1210/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0258 - accuracy: 0.9938 - val_loss: 0.0470 - val_accuracy: 0.9908
    Epoch 1211/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0248 - accuracy: 0.9951 - val_loss: 0.0468 - val_accuracy: 0.9908
    Epoch 1212/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0253 - accuracy: 0.9951 - val_loss: 0.0462 - val_accuracy: 0.9908
    Epoch 1213/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0287 - accuracy: 0.9926 - val_loss: 0.0470 - val_accuracy: 0.9915
    Epoch 1214/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0337 - accuracy: 0.9900 - val_loss: 0.0477 - val_accuracy: 0.9908
    Epoch 1215/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0318 - accuracy: 0.9918 - val_loss: 0.0515 - val_accuracy: 0.9892
    Epoch 1216/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0268 - accuracy: 0.9933 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1217/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0256 - accuracy: 0.9944 - val_loss: 0.0494 - val_accuracy: 0.9892
    Epoch 1218/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0250 - accuracy: 0.9941 - val_loss: 0.0493 - val_accuracy: 0.9892
    Epoch 1219/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0251 - accuracy: 0.9946 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1220/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0271 - accuracy: 0.9926 - val_loss: 0.0499 - val_accuracy: 0.9885
    Epoch 1221/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0256 - accuracy: 0.9936 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1222/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0280 - accuracy: 0.9931 - val_loss: 0.0523 - val_accuracy: 0.9892
    Epoch 1223/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0263 - accuracy: 0.9941 - val_loss: 0.0500 - val_accuracy: 0.9892
    Epoch 1224/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0252 - accuracy: 0.9946 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1225/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0252 - accuracy: 0.9941 - val_loss: 0.0531 - val_accuracy: 0.9900
    Epoch 1226/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0265 - accuracy: 0.9944 - val_loss: 0.0482 - val_accuracy: 0.9908
    Epoch 1227/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0243 - accuracy: 0.9956 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1228/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0273 - accuracy: 0.9938 - val_loss: 0.0485 - val_accuracy: 0.9892
    Epoch 1229/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0255 - accuracy: 0.9938 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1230/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0264 - accuracy: 0.9926 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1231/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0241 - accuracy: 0.9949 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1232/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0239 - accuracy: 0.9954 - val_loss: 0.0485 - val_accuracy: 0.9908
    Epoch 1233/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0246 - accuracy: 0.9936 - val_loss: 0.0486 - val_accuracy: 0.9892
    Epoch 1234/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9962 - val_loss: 0.0522 - val_accuracy: 0.9900
    Epoch 1235/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0275 - accuracy: 0.9931 - val_loss: 0.0569 - val_accuracy: 0.9885
    Epoch 1236/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0293 - accuracy: 0.9918 - val_loss: 0.0511 - val_accuracy: 0.9900
    Epoch 1237/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0275 - accuracy: 0.9931 - val_loss: 0.0481 - val_accuracy: 0.9908
    Epoch 1238/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0248 - accuracy: 0.9946 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1239/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0243 - accuracy: 0.9951 - val_loss: 0.0492 - val_accuracy: 0.9915
    Epoch 1240/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0238 - accuracy: 0.9956 - val_loss: 0.0477 - val_accuracy: 0.9900
    Epoch 1241/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0247 - accuracy: 0.9941 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1242/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0259 - accuracy: 0.9949 - val_loss: 0.0493 - val_accuracy: 0.9908
    Epoch 1243/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0252 - accuracy: 0.9938 - val_loss: 0.0544 - val_accuracy: 0.9892
    Epoch 1244/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0280 - accuracy: 0.9915 - val_loss: 0.0592 - val_accuracy: 0.9869
    Epoch 1245/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0316 - accuracy: 0.9905 - val_loss: 0.0513 - val_accuracy: 0.9892
    Epoch 1246/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0302 - accuracy: 0.9920 - val_loss: 0.0490 - val_accuracy: 0.9908
    Epoch 1247/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0246 - accuracy: 0.9946 - val_loss: 0.0486 - val_accuracy: 0.9900
    Epoch 1248/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0261 - accuracy: 0.9938 - val_loss: 0.0523 - val_accuracy: 0.9885
    Epoch 1249/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0268 - accuracy: 0.9933 - val_loss: 0.0535 - val_accuracy: 0.9877
    Epoch 1250/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0278 - accuracy: 0.9938 - val_loss: 0.0482 - val_accuracy: 0.9892
    Epoch 1251/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0258 - accuracy: 0.9938 - val_loss: 0.0495 - val_accuracy: 0.9892
    Epoch 1252/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0250 - accuracy: 0.9946 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1253/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0247 - accuracy: 0.9941 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1254/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0242 - accuracy: 0.9949 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1255/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9949 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1256/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0253 - accuracy: 0.9936 - val_loss: 0.0507 - val_accuracy: 0.9885
    Epoch 1257/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0285 - accuracy: 0.9928 - val_loss: 0.0476 - val_accuracy: 0.9900
    Epoch 1258/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0248 - accuracy: 0.9944 - val_loss: 0.0501 - val_accuracy: 0.9908
    Epoch 1259/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0243 - accuracy: 0.9944 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1260/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0260 - accuracy: 0.9938 - val_loss: 0.0481 - val_accuracy: 0.9900
    Epoch 1261/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0250 - accuracy: 0.9946 - val_loss: 0.0500 - val_accuracy: 0.9892
    Epoch 1262/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0251 - accuracy: 0.9951 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1263/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0264 - accuracy: 0.9936 - val_loss: 0.0524 - val_accuracy: 0.9892
    Epoch 1264/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0250 - accuracy: 0.9938 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1265/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0244 - accuracy: 0.9941 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1266/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0237 - accuracy: 0.9949 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1267/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0234 - accuracy: 0.9944 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1268/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0243 - accuracy: 0.9949 - val_loss: 0.0486 - val_accuracy: 0.9908
    Epoch 1269/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0241 - accuracy: 0.9954 - val_loss: 0.0496 - val_accuracy: 0.9908
    Epoch 1270/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0280 - accuracy: 0.9926 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1271/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0352 - accuracy: 0.9905 - val_loss: 0.0556 - val_accuracy: 0.9885
    Epoch 1272/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0311 - accuracy: 0.9908 - val_loss: 0.0524 - val_accuracy: 0.9900
    Epoch 1273/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0270 - accuracy: 0.9928 - val_loss: 0.0557 - val_accuracy: 0.9892
    Epoch 1274/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0292 - accuracy: 0.9938 - val_loss: 0.0604 - val_accuracy: 0.9854
    Epoch 1275/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0295 - accuracy: 0.9923 - val_loss: 0.0572 - val_accuracy: 0.9885
    Epoch 1276/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0253 - accuracy: 0.9936 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1277/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0244 - accuracy: 0.9951 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1278/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0246 - accuracy: 0.9938 - val_loss: 0.0488 - val_accuracy: 0.9908
    Epoch 1279/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0254 - accuracy: 0.9931 - val_loss: 0.0470 - val_accuracy: 0.9908
    Epoch 1280/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0254 - accuracy: 0.9936 - val_loss: 0.0499 - val_accuracy: 0.9892
    Epoch 1281/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0252 - accuracy: 0.9949 - val_loss: 0.0493 - val_accuracy: 0.9892
    Epoch 1282/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0243 - accuracy: 0.9949 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1283/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0240 - accuracy: 0.9938 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1284/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0232 - accuracy: 0.9962 - val_loss: 0.0485 - val_accuracy: 0.9915
    Epoch 1285/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0235 - accuracy: 0.9954 - val_loss: 0.0481 - val_accuracy: 0.9908
    Epoch 1286/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0236 - accuracy: 0.9949 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1287/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0243 - accuracy: 0.9946 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1288/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0251 - accuracy: 0.9938 - val_loss: 0.0491 - val_accuracy: 0.9892
    Epoch 1289/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0280 - accuracy: 0.9946 - val_loss: 0.0492 - val_accuracy: 0.9908
    Epoch 1290/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0356 - accuracy: 0.9890 - val_loss: 0.0473 - val_accuracy: 0.9900
    Epoch 1291/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0319 - accuracy: 0.9923 - val_loss: 0.0491 - val_accuracy: 0.9892
    Epoch 1292/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0339 - accuracy: 0.9897 - val_loss: 0.0521 - val_accuracy: 0.9885
    Epoch 1293/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0368 - accuracy: 0.9892 - val_loss: 0.0658 - val_accuracy: 0.9838
    Epoch 1294/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0353 - accuracy: 0.9908 - val_loss: 0.0602 - val_accuracy: 0.9854
    Epoch 1295/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0277 - accuracy: 0.9931 - val_loss: 0.0576 - val_accuracy: 0.9877
    Epoch 1296/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0269 - accuracy: 0.9928 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1297/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0244 - accuracy: 0.9946 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1298/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0244 - accuracy: 0.9944 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1299/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0242 - accuracy: 0.9954 - val_loss: 0.0495 - val_accuracy: 0.9908
    Epoch 1300/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0248 - accuracy: 0.9946 - val_loss: 0.0479 - val_accuracy: 0.9908
    Epoch 1301/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0257 - accuracy: 0.9946 - val_loss: 0.0533 - val_accuracy: 0.9885
    Epoch 1302/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0284 - accuracy: 0.9928 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1303/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0251 - accuracy: 0.9944 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1304/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0245 - accuracy: 0.9941 - val_loss: 0.0506 - val_accuracy: 0.9877
    Epoch 1305/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0265 - accuracy: 0.9949 - val_loss: 0.0495 - val_accuracy: 0.9892
    Epoch 1306/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0287 - accuracy: 0.9928 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1307/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0242 - accuracy: 0.9941 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1308/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0230 - accuracy: 0.9962 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1309/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0233 - accuracy: 0.9956 - val_loss: 0.0482 - val_accuracy: 0.9908
    Epoch 1310/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0232 - accuracy: 0.9959 - val_loss: 0.0496 - val_accuracy: 0.9885
    Epoch 1311/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0248 - accuracy: 0.9944 - val_loss: 0.0487 - val_accuracy: 0.9908
    Epoch 1312/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9954 - val_loss: 0.0471 - val_accuracy: 0.9908
    Epoch 1313/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0232 - accuracy: 0.9959 - val_loss: 0.0470 - val_accuracy: 0.9900
    Epoch 1314/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0233 - accuracy: 0.9951 - val_loss: 0.0481 - val_accuracy: 0.9915
    Epoch 1315/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0249 - accuracy: 0.9944 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 1316/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0242 - accuracy: 0.9951 - val_loss: 0.0481 - val_accuracy: 0.9900
    Epoch 1317/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0242 - accuracy: 0.9946 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1318/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0234 - accuracy: 0.9959 - val_loss: 0.0479 - val_accuracy: 0.9900
    Epoch 1319/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9956 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1320/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9938 - val_loss: 0.0501 - val_accuracy: 0.9885
    Epoch 1321/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0258 - accuracy: 0.9946 - val_loss: 0.0493 - val_accuracy: 0.9885
    Epoch 1322/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0262 - accuracy: 0.9944 - val_loss: 0.0461 - val_accuracy: 0.9900
    Epoch 1323/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0262 - accuracy: 0.9931 - val_loss: 0.0472 - val_accuracy: 0.9892
    Epoch 1324/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0252 - accuracy: 0.9946 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 1325/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0240 - accuracy: 0.9938 - val_loss: 0.0632 - val_accuracy: 0.9846
    Epoch 1326/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0296 - accuracy: 0.9926 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 1327/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0242 - accuracy: 0.9946 - val_loss: 0.0553 - val_accuracy: 0.9885
    Epoch 1328/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0255 - accuracy: 0.9949 - val_loss: 0.0526 - val_accuracy: 0.9900
    Epoch 1329/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0279 - accuracy: 0.9931 - val_loss: 0.0510 - val_accuracy: 0.9908
    Epoch 1330/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0236 - accuracy: 0.9938 - val_loss: 0.0528 - val_accuracy: 0.9892
    Epoch 1331/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0260 - accuracy: 0.9936 - val_loss: 0.0523 - val_accuracy: 0.9900
    Epoch 1332/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0247 - accuracy: 0.9949 - val_loss: 0.0479 - val_accuracy: 0.9908
    Epoch 1333/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0233 - accuracy: 0.9956 - val_loss: 0.0481 - val_accuracy: 0.9900
    Epoch 1334/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0242 - accuracy: 0.9946 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1335/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0257 - accuracy: 0.9933 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1336/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0248 - accuracy: 0.9938 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1337/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0252 - accuracy: 0.9951 - val_loss: 0.0506 - val_accuracy: 0.9892
    Epoch 1338/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0233 - accuracy: 0.9951 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1339/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0241 - accuracy: 0.9941 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1340/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0227 - accuracy: 0.9962 - val_loss: 0.0476 - val_accuracy: 0.9900
    Epoch 1341/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9959 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1342/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0234 - accuracy: 0.9956 - val_loss: 0.0479 - val_accuracy: 0.9908
    Epoch 1343/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0244 - accuracy: 0.9941 - val_loss: 0.0481 - val_accuracy: 0.9900
    Epoch 1344/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0245 - accuracy: 0.9951 - val_loss: 0.0496 - val_accuracy: 0.9892
    Epoch 1345/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9951 - val_loss: 0.0476 - val_accuracy: 0.9900
    Epoch 1346/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0230 - accuracy: 0.9951 - val_loss: 0.0465 - val_accuracy: 0.9900
    Epoch 1347/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0243 - accuracy: 0.9951 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1348/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0247 - accuracy: 0.9944 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1349/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0261 - accuracy: 0.9938 - val_loss: 0.0480 - val_accuracy: 0.9908
    Epoch 1350/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0281 - accuracy: 0.9918 - val_loss: 0.0524 - val_accuracy: 0.9908
    Epoch 1351/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0280 - accuracy: 0.9936 - val_loss: 0.0484 - val_accuracy: 0.9908
    Epoch 1352/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0261 - accuracy: 0.9936 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1353/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0254 - accuracy: 0.9946 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1354/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0233 - accuracy: 0.9949 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1355/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0242 - accuracy: 0.9951 - val_loss: 0.0471 - val_accuracy: 0.9892
    Epoch 1356/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9954 - val_loss: 0.0486 - val_accuracy: 0.9908
    Epoch 1357/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0256 - accuracy: 0.9949 - val_loss: 0.0535 - val_accuracy: 0.9900
    Epoch 1358/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0248 - accuracy: 0.9923 - val_loss: 0.0486 - val_accuracy: 0.9908
    Epoch 1359/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0227 - accuracy: 0.9954 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1360/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0238 - accuracy: 0.9946 - val_loss: 0.0492 - val_accuracy: 0.9908
    Epoch 1361/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0234 - accuracy: 0.9951 - val_loss: 0.0517 - val_accuracy: 0.9908
    Epoch 1362/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0246 - accuracy: 0.9944 - val_loss: 0.0538 - val_accuracy: 0.9900
    Epoch 1363/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0256 - accuracy: 0.9946 - val_loss: 0.0526 - val_accuracy: 0.9900
    Epoch 1364/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0233 - accuracy: 0.9962 - val_loss: 0.0522 - val_accuracy: 0.9908
    Epoch 1365/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0241 - accuracy: 0.9933 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1366/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0228 - accuracy: 0.9949 - val_loss: 0.0494 - val_accuracy: 0.9892
    Epoch 1367/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0225 - accuracy: 0.9962 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1368/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0224 - accuracy: 0.9962 - val_loss: 0.0486 - val_accuracy: 0.9892
    Epoch 1369/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0227 - accuracy: 0.9954 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1370/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9959 - val_loss: 0.0472 - val_accuracy: 0.9892
    Epoch 1371/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0231 - accuracy: 0.9956 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1372/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0229 - accuracy: 0.9951 - val_loss: 0.0488 - val_accuracy: 0.9908
    Epoch 1373/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0233 - accuracy: 0.9949 - val_loss: 0.0486 - val_accuracy: 0.9900
    Epoch 1374/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0229 - accuracy: 0.9954 - val_loss: 0.0504 - val_accuracy: 0.9915
    Epoch 1375/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0225 - accuracy: 0.9959 - val_loss: 0.0485 - val_accuracy: 0.9915
    Epoch 1376/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9956 - val_loss: 0.0468 - val_accuracy: 0.9900
    Epoch 1377/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0223 - accuracy: 0.9964 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1378/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0226 - accuracy: 0.9959 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1379/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0223 - accuracy: 0.9956 - val_loss: 0.0482 - val_accuracy: 0.9908
    Epoch 1380/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0225 - accuracy: 0.9964 - val_loss: 0.0479 - val_accuracy: 0.9892
    Epoch 1381/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0227 - accuracy: 0.9951 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1382/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9949 - val_loss: 0.0501 - val_accuracy: 0.9908
    Epoch 1383/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0232 - accuracy: 0.9962 - val_loss: 0.0486 - val_accuracy: 0.9900
    Epoch 1384/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0230 - accuracy: 0.9956 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1385/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0234 - accuracy: 0.9956 - val_loss: 0.0483 - val_accuracy: 0.9892
    Epoch 1386/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0233 - accuracy: 0.9949 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1387/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0224 - accuracy: 0.9959 - val_loss: 0.0481 - val_accuracy: 0.9908
    Epoch 1388/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0225 - accuracy: 0.9959 - val_loss: 0.0486 - val_accuracy: 0.9908
    Epoch 1389/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0509 - val_accuracy: 0.9908
    Epoch 1390/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0223 - accuracy: 0.9959 - val_loss: 0.0480 - val_accuracy: 0.9892
    Epoch 1391/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0218 - accuracy: 0.9959 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1392/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0243 - accuracy: 0.9949 - val_loss: 0.0494 - val_accuracy: 0.9892
    Epoch 1393/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0257 - accuracy: 0.9936 - val_loss: 0.0476 - val_accuracy: 0.9892
    Epoch 1394/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0286 - accuracy: 0.9918 - val_loss: 0.0640 - val_accuracy: 0.9846
    Epoch 1395/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0319 - accuracy: 0.9913 - val_loss: 0.0531 - val_accuracy: 0.9892
    Epoch 1396/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0297 - accuracy: 0.9920 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1397/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0238 - accuracy: 0.9949 - val_loss: 0.0493 - val_accuracy: 0.9908
    Epoch 1398/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0224 - accuracy: 0.9959 - val_loss: 0.0528 - val_accuracy: 0.9908
    Epoch 1399/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0227 - accuracy: 0.9962 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1400/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0233 - accuracy: 0.9946 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1401/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0247 - accuracy: 0.9938 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 1402/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0229 - accuracy: 0.9956 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 1403/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0224 - accuracy: 0.9954 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 1404/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0248 - accuracy: 0.9944 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1405/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0252 - accuracy: 0.9946 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1406/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0235 - accuracy: 0.9951 - val_loss: 0.0541 - val_accuracy: 0.9900
    Epoch 1407/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0231 - accuracy: 0.9951 - val_loss: 0.0510 - val_accuracy: 0.9908
    Epoch 1408/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0230 - accuracy: 0.9954 - val_loss: 0.0495 - val_accuracy: 0.9908
    Epoch 1409/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0238 - accuracy: 0.9946 - val_loss: 0.0509 - val_accuracy: 0.9908
    Epoch 1410/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0235 - accuracy: 0.9949 - val_loss: 0.0514 - val_accuracy: 0.9900
    Epoch 1411/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0240 - accuracy: 0.9951 - val_loss: 0.0515 - val_accuracy: 0.9908
    Epoch 1412/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0234 - accuracy: 0.9954 - val_loss: 0.0505 - val_accuracy: 0.9908
    Epoch 1413/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0227 - accuracy: 0.9954 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1414/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0228 - accuracy: 0.9946 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1415/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0242 - accuracy: 0.9954 - val_loss: 0.0515 - val_accuracy: 0.9908
    Epoch 1416/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0262 - accuracy: 0.9941 - val_loss: 0.0494 - val_accuracy: 0.9908
    Epoch 1417/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0237 - accuracy: 0.9954 - val_loss: 0.0489 - val_accuracy: 0.9892
    Epoch 1418/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0230 - accuracy: 0.9956 - val_loss: 0.0479 - val_accuracy: 0.9892
    Epoch 1419/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0220 - accuracy: 0.9956 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1420/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0229 - accuracy: 0.9951 - val_loss: 0.0473 - val_accuracy: 0.9900
    Epoch 1421/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0296 - accuracy: 0.9918 - val_loss: 0.0543 - val_accuracy: 0.9885
    Epoch 1422/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0264 - accuracy: 0.9933 - val_loss: 0.0486 - val_accuracy: 0.9892
    Epoch 1423/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0248 - accuracy: 0.9941 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1424/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0238 - accuracy: 0.9951 - val_loss: 0.0516 - val_accuracy: 0.9915
    Epoch 1425/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0234 - accuracy: 0.9944 - val_loss: 0.0491 - val_accuracy: 0.9892
    Epoch 1426/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9956 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1427/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0225 - accuracy: 0.9962 - val_loss: 0.0505 - val_accuracy: 0.9908
    Epoch 1428/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0226 - accuracy: 0.9954 - val_loss: 0.0513 - val_accuracy: 0.9908
    Epoch 1429/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0253 - accuracy: 0.9941 - val_loss: 0.0664 - val_accuracy: 0.9846
    Epoch 1430/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0269 - accuracy: 0.9931 - val_loss: 0.0536 - val_accuracy: 0.9915
    Epoch 1431/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0246 - accuracy: 0.9941 - val_loss: 0.0528 - val_accuracy: 0.9908
    Epoch 1432/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0233 - accuracy: 0.9949 - val_loss: 0.0480 - val_accuracy: 0.9908
    Epoch 1433/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0222 - accuracy: 0.9954 - val_loss: 0.0541 - val_accuracy: 0.9908
    Epoch 1434/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0240 - accuracy: 0.9944 - val_loss: 0.0590 - val_accuracy: 0.9885
    Epoch 1435/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0267 - accuracy: 0.9933 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1436/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0255 - accuracy: 0.9941 - val_loss: 0.0583 - val_accuracy: 0.9885
    Epoch 1437/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0256 - accuracy: 0.9938 - val_loss: 0.0540 - val_accuracy: 0.9900
    Epoch 1438/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0243 - accuracy: 0.9944 - val_loss: 0.0618 - val_accuracy: 0.9877
    Epoch 1439/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0273 - accuracy: 0.9928 - val_loss: 0.0616 - val_accuracy: 0.9877
    Epoch 1440/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0292 - accuracy: 0.9913 - val_loss: 0.0518 - val_accuracy: 0.9908
    Epoch 1441/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0227 - accuracy: 0.9951 - val_loss: 0.0527 - val_accuracy: 0.9908
    Epoch 1442/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0224 - accuracy: 0.9962 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1443/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0224 - accuracy: 0.9949 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1444/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0219 - accuracy: 0.9959 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1445/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0219 - accuracy: 0.9959 - val_loss: 0.0489 - val_accuracy: 0.9908
    Epoch 1446/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0231 - accuracy: 0.9949 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1447/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9959 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1448/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0225 - accuracy: 0.9956 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1449/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0221 - accuracy: 0.9956 - val_loss: 0.0500 - val_accuracy: 0.9908
    Epoch 1450/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0219 - accuracy: 0.9954 - val_loss: 0.0494 - val_accuracy: 0.9908
    Epoch 1451/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0217 - accuracy: 0.9954 - val_loss: 0.0527 - val_accuracy: 0.9908
    Epoch 1452/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1453/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0219 - accuracy: 0.9962 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1454/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0218 - accuracy: 0.9967 - val_loss: 0.0512 - val_accuracy: 0.9908
    Epoch 1455/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0222 - accuracy: 0.9956 - val_loss: 0.0506 - val_accuracy: 0.9908
    Epoch 1456/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0514 - val_accuracy: 0.9908
    Epoch 1457/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0232 - accuracy: 0.9946 - val_loss: 0.0494 - val_accuracy: 0.9892
    Epoch 1458/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0243 - accuracy: 0.9941 - val_loss: 0.0526 - val_accuracy: 0.9892
    Epoch 1459/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0257 - accuracy: 0.9944 - val_loss: 0.0475 - val_accuracy: 0.9892
    Epoch 1460/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0237 - accuracy: 0.9954 - val_loss: 0.0488 - val_accuracy: 0.9885
    Epoch 1461/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0222 - accuracy: 0.9949 - val_loss: 0.0505 - val_accuracy: 0.9908
    Epoch 1462/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0231 - accuracy: 0.9941 - val_loss: 0.0525 - val_accuracy: 0.9915
    Epoch 1463/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0241 - accuracy: 0.9949 - val_loss: 0.0537 - val_accuracy: 0.9908
    Epoch 1464/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0258 - accuracy: 0.9931 - val_loss: 0.0525 - val_accuracy: 0.9908
    Epoch 1465/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0250 - accuracy: 0.9938 - val_loss: 0.0608 - val_accuracy: 0.9877
    Epoch 1466/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0258 - accuracy: 0.9954 - val_loss: 0.0508 - val_accuracy: 0.9915
    Epoch 1467/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0231 - accuracy: 0.9951 - val_loss: 0.0533 - val_accuracy: 0.9908
    Epoch 1468/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0233 - accuracy: 0.9944 - val_loss: 0.0535 - val_accuracy: 0.9908
    Epoch 1469/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0244 - accuracy: 0.9944 - val_loss: 0.0566 - val_accuracy: 0.9900
    Epoch 1470/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0271 - accuracy: 0.9931 - val_loss: 0.0618 - val_accuracy: 0.9869
    Epoch 1471/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0251 - accuracy: 0.9946 - val_loss: 0.0518 - val_accuracy: 0.9892
    Epoch 1472/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0225 - accuracy: 0.9946 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1473/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0236 - accuracy: 0.9949 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1474/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9954 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1475/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0223 - accuracy: 0.9956 - val_loss: 0.0528 - val_accuracy: 0.9900
    Epoch 1476/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0231 - accuracy: 0.9949 - val_loss: 0.0528 - val_accuracy: 0.9908
    Epoch 1477/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9962 - val_loss: 0.0507 - val_accuracy: 0.9892
    Epoch 1478/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0222 - accuracy: 0.9954 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1479/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0213 - accuracy: 0.9959 - val_loss: 0.0521 - val_accuracy: 0.9915
    Epoch 1480/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0231 - accuracy: 0.9954 - val_loss: 0.0561 - val_accuracy: 0.9900
    Epoch 1481/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1482/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0219 - accuracy: 0.9959 - val_loss: 0.0541 - val_accuracy: 0.9908
    Epoch 1483/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0223 - accuracy: 0.9946 - val_loss: 0.0476 - val_accuracy: 0.9900
    Epoch 1484/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0211 - accuracy: 0.9959 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1485/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0234 - accuracy: 0.9944 - val_loss: 0.0487 - val_accuracy: 0.9892
    Epoch 1486/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1487/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0215 - accuracy: 0.9956 - val_loss: 0.0486 - val_accuracy: 0.9892
    Epoch 1488/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0235 - accuracy: 0.9938 - val_loss: 0.0512 - val_accuracy: 0.9892
    Epoch 1489/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0257 - accuracy: 0.9928 - val_loss: 0.0535 - val_accuracy: 0.9892
    Epoch 1490/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0245 - accuracy: 0.9944 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1491/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0213 - accuracy: 0.9956 - val_loss: 0.0480 - val_accuracy: 0.9892
    Epoch 1492/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0230 - accuracy: 0.9949 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1493/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0249 - accuracy: 0.9946 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1494/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0210 - accuracy: 0.9959 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1495/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0222 - accuracy: 0.9956 - val_loss: 0.0499 - val_accuracy: 0.9892
    Epoch 1496/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0244 - accuracy: 0.9936 - val_loss: 0.0473 - val_accuracy: 0.9900
    Epoch 1497/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0225 - accuracy: 0.9951 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1498/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0213 - accuracy: 0.9959 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1499/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0224 - accuracy: 0.9951 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 1500/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0231 - accuracy: 0.9954 - val_loss: 0.0478 - val_accuracy: 0.9892
    Epoch 1501/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0240 - accuracy: 0.9944 - val_loss: 0.0489 - val_accuracy: 0.9892
    Epoch 1502/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0219 - accuracy: 0.9949 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1503/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9954 - val_loss: 0.0544 - val_accuracy: 0.9908
    Epoch 1504/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9959 - val_loss: 0.0475 - val_accuracy: 0.9900
    Epoch 1505/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0223 - accuracy: 0.9944 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1506/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0247 - accuracy: 0.9946 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 1507/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0241 - accuracy: 0.9936 - val_loss: 0.0473 - val_accuracy: 0.9892
    Epoch 1508/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0238 - accuracy: 0.9946 - val_loss: 0.0489 - val_accuracy: 0.9892
    Epoch 1509/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0218 - accuracy: 0.9956 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1510/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0234 - accuracy: 0.9951 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1511/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0266 - accuracy: 0.9931 - val_loss: 0.0490 - val_accuracy: 0.9892
    Epoch 1512/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0217 - accuracy: 0.9956 - val_loss: 0.0499 - val_accuracy: 0.9892
    Epoch 1513/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0210 - accuracy: 0.9962 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1514/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0209 - accuracy: 0.9964 - val_loss: 0.0524 - val_accuracy: 0.9908
    Epoch 1515/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0220 - accuracy: 0.9956 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1516/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0206 - accuracy: 0.9964 - val_loss: 0.0477 - val_accuracy: 0.9892
    Epoch 1517/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0211 - accuracy: 0.9959 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1518/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0218 - accuracy: 0.9951 - val_loss: 0.0530 - val_accuracy: 0.9885
    Epoch 1519/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0292 - accuracy: 0.9913 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1520/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0227 - accuracy: 0.9946 - val_loss: 0.0521 - val_accuracy: 0.9900
    Epoch 1521/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9959 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1522/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0218 - accuracy: 0.9959 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1523/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0211 - accuracy: 0.9956 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1524/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0216 - accuracy: 0.9959 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1525/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0219 - accuracy: 0.9954 - val_loss: 0.0485 - val_accuracy: 0.9892
    Epoch 1526/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9959 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1527/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0233 - accuracy: 0.9938 - val_loss: 0.0541 - val_accuracy: 0.9908
    Epoch 1528/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0310 - accuracy: 0.9920 - val_loss: 0.0514 - val_accuracy: 0.9915
    Epoch 1529/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0256 - accuracy: 0.9933 - val_loss: 0.0517 - val_accuracy: 0.9892
    Epoch 1530/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0266 - accuracy: 0.9933 - val_loss: 0.0488 - val_accuracy: 0.9885
    Epoch 1531/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0227 - accuracy: 0.9944 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1532/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9962 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1533/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9944 - val_loss: 0.0511 - val_accuracy: 0.9908
    Epoch 1534/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0214 - accuracy: 0.9949 - val_loss: 0.0470 - val_accuracy: 0.9892
    Epoch 1535/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0237 - accuracy: 0.9931 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1536/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0231 - accuracy: 0.9954 - val_loss: 0.0525 - val_accuracy: 0.9908
    Epoch 1537/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9946 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1538/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0216 - accuracy: 0.9954 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1539/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0211 - accuracy: 0.9962 - val_loss: 0.0488 - val_accuracy: 0.9892
    Epoch 1540/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9951 - val_loss: 0.0483 - val_accuracy: 0.9892
    Epoch 1541/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0212 - accuracy: 0.9956 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1542/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0213 - accuracy: 0.9956 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1543/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0222 - accuracy: 0.9956 - val_loss: 0.0509 - val_accuracy: 0.9900
    Epoch 1544/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0220 - accuracy: 0.9959 - val_loss: 0.0520 - val_accuracy: 0.9908
    Epoch 1545/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0212 - accuracy: 0.9954 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1546/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0220 - accuracy: 0.9949 - val_loss: 0.0506 - val_accuracy: 0.9900
    Epoch 1547/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0227 - accuracy: 0.9951 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1548/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0549 - val_accuracy: 0.9900
    Epoch 1549/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0219 - accuracy: 0.9949 - val_loss: 0.0478 - val_accuracy: 0.9908
    Epoch 1550/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0241 - accuracy: 0.9941 - val_loss: 0.0501 - val_accuracy: 0.9915
    Epoch 1551/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0216 - accuracy: 0.9954 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1552/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0215 - accuracy: 0.9954 - val_loss: 0.0484 - val_accuracy: 0.9892
    Epoch 1553/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0212 - accuracy: 0.9956 - val_loss: 0.0484 - val_accuracy: 0.9892
    Epoch 1554/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0219 - accuracy: 0.9956 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1555/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0222 - accuracy: 0.9951 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1556/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0220 - accuracy: 0.9959 - val_loss: 0.0508 - val_accuracy: 0.9892
    Epoch 1557/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0222 - accuracy: 0.9954 - val_loss: 0.0518 - val_accuracy: 0.9892
    Epoch 1558/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9946 - val_loss: 0.0523 - val_accuracy: 0.9885
    Epoch 1559/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0212 - accuracy: 0.9949 - val_loss: 0.0493 - val_accuracy: 0.9892
    Epoch 1560/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9941 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1561/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0216 - accuracy: 0.9959 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1562/2000
    8/8 [==============================] - 0s 45ms/step - loss: 0.0204 - accuracy: 0.9962 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1563/2000
    8/8 [==============================] - 0s 45ms/step - loss: 0.0232 - accuracy: 0.9949 - val_loss: 0.0519 - val_accuracy: 0.9892
    Epoch 1564/2000
    8/8 [==============================] - 0s 51ms/step - loss: 0.0242 - accuracy: 0.9946 - val_loss: 0.0514 - val_accuracy: 0.9900
    Epoch 1565/2000
    8/8 [==============================] - 0s 51ms/step - loss: 0.0218 - accuracy: 0.9951 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1566/2000
    8/8 [==============================] - 0s 41ms/step - loss: 0.0209 - accuracy: 0.9956 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1567/2000
    8/8 [==============================] - 0s 42ms/step - loss: 0.0208 - accuracy: 0.9962 - val_loss: 0.0476 - val_accuracy: 0.9892
    Epoch 1568/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0220 - accuracy: 0.9959 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1569/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0243 - accuracy: 0.9938 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1570/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0265 - accuracy: 0.9920 - val_loss: 0.0506 - val_accuracy: 0.9908
    Epoch 1571/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0222 - accuracy: 0.9946 - val_loss: 0.0487 - val_accuracy: 0.9892
    Epoch 1572/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0209 - accuracy: 0.9956 - val_loss: 0.0514 - val_accuracy: 0.9900
    Epoch 1573/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0262 - accuracy: 0.9938 - val_loss: 0.0509 - val_accuracy: 0.9892
    Epoch 1574/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9951 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1575/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0212 - accuracy: 0.9956 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1576/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0219 - accuracy: 0.9956 - val_loss: 0.0509 - val_accuracy: 0.9892
    Epoch 1577/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0204 - accuracy: 0.9959 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1578/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0205 - accuracy: 0.9962 - val_loss: 0.0516 - val_accuracy: 0.9900
    Epoch 1579/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0264 - accuracy: 0.9920 - val_loss: 0.0483 - val_accuracy: 0.9892
    Epoch 1580/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0234 - accuracy: 0.9946 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1581/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0207 - accuracy: 0.9962 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1582/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0209 - accuracy: 0.9964 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1583/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0207 - accuracy: 0.9954 - val_loss: 0.0511 - val_accuracy: 0.9908
    Epoch 1584/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9962 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1585/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0214 - accuracy: 0.9941 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1586/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0208 - accuracy: 0.9959 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1587/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0208 - accuracy: 0.9956 - val_loss: 0.0516 - val_accuracy: 0.9900
    Epoch 1588/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0213 - accuracy: 0.9954 - val_loss: 0.0540 - val_accuracy: 0.9908
    Epoch 1589/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9946 - val_loss: 0.0525 - val_accuracy: 0.9900
    Epoch 1590/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0232 - accuracy: 0.9946 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1591/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0210 - accuracy: 0.9954 - val_loss: 0.0557 - val_accuracy: 0.9892
    Epoch 1592/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0206 - accuracy: 0.9962 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1593/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0206 - accuracy: 0.9962 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1594/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0231 - accuracy: 0.9944 - val_loss: 0.0536 - val_accuracy: 0.9900
    Epoch 1595/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0299 - accuracy: 0.9915 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1596/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0391 - accuracy: 0.9879 - val_loss: 0.0475 - val_accuracy: 0.9885
    Epoch 1597/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0284 - accuracy: 0.9905 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1598/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0233 - accuracy: 0.9944 - val_loss: 0.0676 - val_accuracy: 0.9877
    Epoch 1599/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0236 - accuracy: 0.9938 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1600/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0235 - accuracy: 0.9951 - val_loss: 0.0546 - val_accuracy: 0.9892
    Epoch 1601/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0225 - accuracy: 0.9949 - val_loss: 0.0526 - val_accuracy: 0.9892
    Epoch 1602/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0207 - accuracy: 0.9962 - val_loss: 0.0476 - val_accuracy: 0.9900
    Epoch 1603/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0207 - accuracy: 0.9964 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1604/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0205 - accuracy: 0.9967 - val_loss: 0.0509 - val_accuracy: 0.9900
    Epoch 1605/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0208 - accuracy: 0.9951 - val_loss: 0.0522 - val_accuracy: 0.9908
    Epoch 1606/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0204 - accuracy: 0.9956 - val_loss: 0.0511 - val_accuracy: 0.9900
    Epoch 1607/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0202 - accuracy: 0.9964 - val_loss: 0.0515 - val_accuracy: 0.9908
    Epoch 1608/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0207 - accuracy: 0.9956 - val_loss: 0.0533 - val_accuracy: 0.9900
    Epoch 1609/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0232 - accuracy: 0.9944 - val_loss: 0.0536 - val_accuracy: 0.9908
    Epoch 1610/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9962 - val_loss: 0.0513 - val_accuracy: 0.9900
    Epoch 1611/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9959 - val_loss: 0.0500 - val_accuracy: 0.9892
    Epoch 1612/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0207 - accuracy: 0.9954 - val_loss: 0.0530 - val_accuracy: 0.9900
    Epoch 1613/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0212 - accuracy: 0.9954 - val_loss: 0.0528 - val_accuracy: 0.9908
    Epoch 1614/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9959 - val_loss: 0.0519 - val_accuracy: 0.9908
    Epoch 1615/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0211 - accuracy: 0.9951 - val_loss: 0.0554 - val_accuracy: 0.9908
    Epoch 1616/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0257 - accuracy: 0.9926 - val_loss: 0.0578 - val_accuracy: 0.9900
    Epoch 1617/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0221 - accuracy: 0.9949 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1618/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0202 - accuracy: 0.9959 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1619/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9959 - val_loss: 0.0526 - val_accuracy: 0.9892
    Epoch 1620/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0242 - accuracy: 0.9936 - val_loss: 0.0520 - val_accuracy: 0.9900
    Epoch 1621/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0257 - accuracy: 0.9936 - val_loss: 0.0491 - val_accuracy: 0.9885
    Epoch 1622/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0250 - accuracy: 0.9936 - val_loss: 0.0674 - val_accuracy: 0.9815
    Epoch 1623/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0271 - accuracy: 0.9918 - val_loss: 0.0515 - val_accuracy: 0.9885
    Epoch 1624/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0233 - accuracy: 0.9928 - val_loss: 0.0547 - val_accuracy: 0.9885
    Epoch 1625/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0247 - accuracy: 0.9938 - val_loss: 0.0571 - val_accuracy: 0.9869
    Epoch 1626/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0250 - accuracy: 0.9931 - val_loss: 0.0485 - val_accuracy: 0.9892
    Epoch 1627/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0221 - accuracy: 0.9956 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1628/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0203 - accuracy: 0.9959 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1629/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0198 - accuracy: 0.9964 - val_loss: 0.0513 - val_accuracy: 0.9892
    Epoch 1630/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0216 - accuracy: 0.9949 - val_loss: 0.0477 - val_accuracy: 0.9892
    Epoch 1631/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0227 - accuracy: 0.9941 - val_loss: 0.0518 - val_accuracy: 0.9908
    Epoch 1632/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0218 - accuracy: 0.9946 - val_loss: 0.0538 - val_accuracy: 0.9908
    Epoch 1633/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0218 - accuracy: 0.9949 - val_loss: 0.0632 - val_accuracy: 0.9869
    Epoch 1634/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0265 - accuracy: 0.9923 - val_loss: 0.0599 - val_accuracy: 0.9908
    Epoch 1635/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9931 - val_loss: 0.0567 - val_accuracy: 0.9900
    Epoch 1636/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0282 - accuracy: 0.9915 - val_loss: 0.0555 - val_accuracy: 0.9908
    Epoch 1637/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0242 - accuracy: 0.9941 - val_loss: 0.0519 - val_accuracy: 0.9908
    Epoch 1638/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0212 - accuracy: 0.9954 - val_loss: 0.0531 - val_accuracy: 0.9908
    Epoch 1639/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0210 - accuracy: 0.9954 - val_loss: 0.0527 - val_accuracy: 0.9892
    Epoch 1640/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0203 - accuracy: 0.9962 - val_loss: 0.0488 - val_accuracy: 0.9900
    Epoch 1641/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0202 - accuracy: 0.9964 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1642/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0217 - accuracy: 0.9959 - val_loss: 0.0550 - val_accuracy: 0.9892
    Epoch 1643/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0228 - accuracy: 0.9941 - val_loss: 0.0552 - val_accuracy: 0.9885
    Epoch 1644/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0221 - accuracy: 0.9954 - val_loss: 0.0558 - val_accuracy: 0.9892
    Epoch 1645/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0245 - accuracy: 0.9936 - val_loss: 0.0600 - val_accuracy: 0.9877
    Epoch 1646/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0313 - accuracy: 0.9902 - val_loss: 0.0754 - val_accuracy: 0.9846
    Epoch 1647/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0324 - accuracy: 0.9902 - val_loss: 0.0702 - val_accuracy: 0.9846
    Epoch 1648/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0392 - accuracy: 0.9877 - val_loss: 0.0800 - val_accuracy: 0.9831
    Epoch 1649/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0422 - accuracy: 0.9864 - val_loss: 0.0631 - val_accuracy: 0.9854
    Epoch 1650/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0300 - accuracy: 0.9913 - val_loss: 0.0461 - val_accuracy: 0.9908
    Epoch 1651/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0262 - accuracy: 0.9936 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1652/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0231 - accuracy: 0.9949 - val_loss: 0.0528 - val_accuracy: 0.9892
    Epoch 1653/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0289 - accuracy: 0.9918 - val_loss: 0.0580 - val_accuracy: 0.9877
    Epoch 1654/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0407 - accuracy: 0.9879 - val_loss: 0.1052 - val_accuracy: 0.9708
    Epoch 1655/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0423 - accuracy: 0.9851 - val_loss: 0.0585 - val_accuracy: 0.9869
    Epoch 1656/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0363 - accuracy: 0.9885 - val_loss: 0.0596 - val_accuracy: 0.9877
    Epoch 1657/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0331 - accuracy: 0.9902 - val_loss: 0.0520 - val_accuracy: 0.9885
    Epoch 1658/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0228 - accuracy: 0.9944 - val_loss: 0.0631 - val_accuracy: 0.9877
    Epoch 1659/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0226 - accuracy: 0.9951 - val_loss: 0.0529 - val_accuracy: 0.9900
    Epoch 1660/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0243 - accuracy: 0.9944 - val_loss: 0.0537 - val_accuracy: 0.9885
    Epoch 1661/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0240 - accuracy: 0.9941 - val_loss: 0.0521 - val_accuracy: 0.9900
    Epoch 1662/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0234 - accuracy: 0.9946 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1663/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0208 - accuracy: 0.9964 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1664/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0210 - accuracy: 0.9964 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1665/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0201 - accuracy: 0.9959 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1666/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0208 - accuracy: 0.9962 - val_loss: 0.0544 - val_accuracy: 0.9900
    Epoch 1667/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0222 - accuracy: 0.9949 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1668/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0209 - accuracy: 0.9959 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1669/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0201 - accuracy: 0.9959 - val_loss: 0.0513 - val_accuracy: 0.9900
    Epoch 1670/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0206 - accuracy: 0.9959 - val_loss: 0.0551 - val_accuracy: 0.9915
    Epoch 1671/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0255 - accuracy: 0.9926 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1672/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0255 - accuracy: 0.9938 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1673/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0227 - accuracy: 0.9949 - val_loss: 0.0500 - val_accuracy: 0.9892
    Epoch 1674/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0212 - accuracy: 0.9956 - val_loss: 0.0524 - val_accuracy: 0.9900
    Epoch 1675/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0208 - accuracy: 0.9964 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1676/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0207 - accuracy: 0.9951 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1677/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0200 - accuracy: 0.9962 - val_loss: 0.0508 - val_accuracy: 0.9900
    Epoch 1678/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9962 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1679/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0207 - accuracy: 0.9956 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1680/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0203 - accuracy: 0.9959 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1681/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0199 - accuracy: 0.9956 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1682/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0199 - accuracy: 0.9964 - val_loss: 0.0513 - val_accuracy: 0.9900
    Epoch 1683/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0200 - accuracy: 0.9962 - val_loss: 0.0482 - val_accuracy: 0.9892
    Epoch 1684/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0204 - accuracy: 0.9959 - val_loss: 0.0496 - val_accuracy: 0.9892
    Epoch 1685/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0204 - accuracy: 0.9962 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1686/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0221 - accuracy: 0.9949 - val_loss: 0.0486 - val_accuracy: 0.9892
    Epoch 1687/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0224 - accuracy: 0.9938 - val_loss: 0.0521 - val_accuracy: 0.9900
    Epoch 1688/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0233 - accuracy: 0.9936 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1689/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0220 - accuracy: 0.9951 - val_loss: 0.0539 - val_accuracy: 0.9908
    Epoch 1690/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0197 - accuracy: 0.9962 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1691/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0199 - accuracy: 0.9956 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1692/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0212 - accuracy: 0.9951 - val_loss: 0.0569 - val_accuracy: 0.9908
    Epoch 1693/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0216 - accuracy: 0.9949 - val_loss: 0.0534 - val_accuracy: 0.9908
    Epoch 1694/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0205 - accuracy: 0.9954 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1695/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0205 - accuracy: 0.9954 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1696/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0222 - accuracy: 0.9949 - val_loss: 0.0511 - val_accuracy: 0.9900
    Epoch 1697/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0221 - accuracy: 0.9949 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1698/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0213 - accuracy: 0.9949 - val_loss: 0.0507 - val_accuracy: 0.9900
    Epoch 1699/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9951 - val_loss: 0.0521 - val_accuracy: 0.9915
    Epoch 1700/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0214 - accuracy: 0.9944 - val_loss: 0.0605 - val_accuracy: 0.9892
    Epoch 1701/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0219 - accuracy: 0.9946 - val_loss: 0.0530 - val_accuracy: 0.9900
    Epoch 1702/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0218 - accuracy: 0.9954 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1703/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0195 - accuracy: 0.9956 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1704/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0200 - accuracy: 0.9962 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1705/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0202 - accuracy: 0.9956 - val_loss: 0.0532 - val_accuracy: 0.9892
    Epoch 1706/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9954 - val_loss: 0.0568 - val_accuracy: 0.9900
    Epoch 1707/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0226 - accuracy: 0.9946 - val_loss: 0.0556 - val_accuracy: 0.9900
    Epoch 1708/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0205 - accuracy: 0.9959 - val_loss: 0.0536 - val_accuracy: 0.9900
    Epoch 1709/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0204 - accuracy: 0.9956 - val_loss: 0.0510 - val_accuracy: 0.9892
    Epoch 1710/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0195 - accuracy: 0.9962 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1711/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0198 - accuracy: 0.9962 - val_loss: 0.0520 - val_accuracy: 0.9900
    Epoch 1712/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0193 - accuracy: 0.9964 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1713/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9959 - val_loss: 0.0506 - val_accuracy: 0.9900
    Epoch 1714/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0191 - accuracy: 0.9959 - val_loss: 0.0507 - val_accuracy: 0.9900
    Epoch 1715/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0198 - accuracy: 0.9959 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1716/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0192 - accuracy: 0.9967 - val_loss: 0.0487 - val_accuracy: 0.9892
    Epoch 1717/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9956 - val_loss: 0.0481 - val_accuracy: 0.9892
    Epoch 1718/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0195 - accuracy: 0.9959 - val_loss: 0.0506 - val_accuracy: 0.9900
    Epoch 1719/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9962 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1720/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0193 - accuracy: 0.9962 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1721/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0191 - accuracy: 0.9967 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1722/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0196 - accuracy: 0.9956 - val_loss: 0.0577 - val_accuracy: 0.9892
    Epoch 1723/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0213 - accuracy: 0.9944 - val_loss: 0.0526 - val_accuracy: 0.9908
    Epoch 1724/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0206 - accuracy: 0.9951 - val_loss: 0.0522 - val_accuracy: 0.9900
    Epoch 1725/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0202 - accuracy: 0.9954 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1726/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0205 - accuracy: 0.9954 - val_loss: 0.0517 - val_accuracy: 0.9892
    Epoch 1727/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0200 - accuracy: 0.9951 - val_loss: 0.0544 - val_accuracy: 0.9892
    Epoch 1728/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9946 - val_loss: 0.0496 - val_accuracy: 0.9892
    Epoch 1729/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9959 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1730/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9962 - val_loss: 0.0524 - val_accuracy: 0.9900
    Epoch 1731/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0201 - accuracy: 0.9951 - val_loss: 0.0526 - val_accuracy: 0.9908
    Epoch 1732/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0255 - accuracy: 0.9933 - val_loss: 0.0516 - val_accuracy: 0.9892
    Epoch 1733/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0298 - accuracy: 0.9910 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1734/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0232 - accuracy: 0.9944 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1735/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0216 - accuracy: 0.9941 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1736/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0199 - accuracy: 0.9956 - val_loss: 0.0504 - val_accuracy: 0.9908
    Epoch 1737/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0210 - accuracy: 0.9954 - val_loss: 0.0504 - val_accuracy: 0.9892
    Epoch 1738/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0206 - accuracy: 0.9956 - val_loss: 0.0551 - val_accuracy: 0.9900
    Epoch 1739/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0208 - accuracy: 0.9951 - val_loss: 0.0676 - val_accuracy: 0.9862
    Epoch 1740/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0300 - accuracy: 0.9915 - val_loss: 0.0544 - val_accuracy: 0.9908
    Epoch 1741/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0241 - accuracy: 0.9926 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1742/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0196 - accuracy: 0.9962 - val_loss: 0.0523 - val_accuracy: 0.9900
    Epoch 1743/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0200 - accuracy: 0.9951 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1744/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0199 - accuracy: 0.9956 - val_loss: 0.0496 - val_accuracy: 0.9908
    Epoch 1745/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0193 - accuracy: 0.9959 - val_loss: 0.0562 - val_accuracy: 0.9908
    Epoch 1746/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0206 - accuracy: 0.9954 - val_loss: 0.0534 - val_accuracy: 0.9915
    Epoch 1747/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0228 - accuracy: 0.9938 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1748/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0215 - accuracy: 0.9951 - val_loss: 0.0489 - val_accuracy: 0.9892
    Epoch 1749/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0206 - accuracy: 0.9949 - val_loss: 0.0470 - val_accuracy: 0.9892
    Epoch 1750/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0202 - accuracy: 0.9959 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1751/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9959 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1752/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0191 - accuracy: 0.9962 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1753/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0194 - accuracy: 0.9956 - val_loss: 0.0562 - val_accuracy: 0.9908
    Epoch 1754/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9954 - val_loss: 0.0539 - val_accuracy: 0.9900
    Epoch 1755/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0223 - accuracy: 0.9938 - val_loss: 0.0516 - val_accuracy: 0.9900
    Epoch 1756/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0240 - accuracy: 0.9923 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1757/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0240 - accuracy: 0.9936 - val_loss: 0.0650 - val_accuracy: 0.9838
    Epoch 1758/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0314 - accuracy: 0.9902 - val_loss: 0.0598 - val_accuracy: 0.9877
    Epoch 1759/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0308 - accuracy: 0.9908 - val_loss: 0.0693 - val_accuracy: 0.9838
    Epoch 1760/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0287 - accuracy: 0.9928 - val_loss: 0.0538 - val_accuracy: 0.9869
    Epoch 1761/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0232 - accuracy: 0.9944 - val_loss: 0.0496 - val_accuracy: 0.9892
    Epoch 1762/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0199 - accuracy: 0.9962 - val_loss: 0.0508 - val_accuracy: 0.9908
    Epoch 1763/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0200 - accuracy: 0.9959 - val_loss: 0.0483 - val_accuracy: 0.9908
    Epoch 1764/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0192 - accuracy: 0.9959 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1765/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9962 - val_loss: 0.0510 - val_accuracy: 0.9892
    Epoch 1766/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0220 - accuracy: 0.9944 - val_loss: 0.0485 - val_accuracy: 0.9892
    Epoch 1767/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0198 - accuracy: 0.9954 - val_loss: 0.0552 - val_accuracy: 0.9892
    Epoch 1768/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9954 - val_loss: 0.0484 - val_accuracy: 0.9900
    Epoch 1769/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0195 - accuracy: 0.9954 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1770/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0186 - accuracy: 0.9964 - val_loss: 0.0541 - val_accuracy: 0.9908
    Epoch 1771/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0192 - accuracy: 0.9962 - val_loss: 0.0490 - val_accuracy: 0.9892
    Epoch 1772/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0198 - accuracy: 0.9954 - val_loss: 0.0509 - val_accuracy: 0.9900
    Epoch 1773/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0197 - accuracy: 0.9959 - val_loss: 0.0518 - val_accuracy: 0.9900
    Epoch 1774/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0185 - accuracy: 0.9959 - val_loss: 0.0491 - val_accuracy: 0.9892
    Epoch 1775/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0191 - accuracy: 0.9954 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1776/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0191 - accuracy: 0.9959 - val_loss: 0.0522 - val_accuracy: 0.9900
    Epoch 1777/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0199 - accuracy: 0.9962 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1778/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0189 - accuracy: 0.9964 - val_loss: 0.0528 - val_accuracy: 0.9892
    Epoch 1779/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9951 - val_loss: 0.0485 - val_accuracy: 0.9900
    Epoch 1780/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0193 - accuracy: 0.9962 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1781/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0197 - accuracy: 0.9959 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1782/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0193 - accuracy: 0.9959 - val_loss: 0.0486 - val_accuracy: 0.9900
    Epoch 1783/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0194 - accuracy: 0.9959 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1784/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0188 - accuracy: 0.9959 - val_loss: 0.0574 - val_accuracy: 0.9908
    Epoch 1785/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0198 - accuracy: 0.9951 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1786/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0193 - accuracy: 0.9959 - val_loss: 0.0515 - val_accuracy: 0.9908
    Epoch 1787/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0204 - accuracy: 0.9954 - val_loss: 0.0526 - val_accuracy: 0.9908
    Epoch 1788/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0200 - accuracy: 0.9956 - val_loss: 0.0528 - val_accuracy: 0.9915
    Epoch 1789/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0189 - accuracy: 0.9962 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1790/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0197 - accuracy: 0.9951 - val_loss: 0.0506 - val_accuracy: 0.9892
    Epoch 1791/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0257 - accuracy: 0.9926 - val_loss: 0.0478 - val_accuracy: 0.9908
    Epoch 1792/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0202 - accuracy: 0.9956 - val_loss: 0.0508 - val_accuracy: 0.9900
    Epoch 1793/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0207 - accuracy: 0.9962 - val_loss: 0.0536 - val_accuracy: 0.9892
    Epoch 1794/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9949 - val_loss: 0.0530 - val_accuracy: 0.9900
    Epoch 1795/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0188 - accuracy: 0.9956 - val_loss: 0.0547 - val_accuracy: 0.9908
    Epoch 1796/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0217 - accuracy: 0.9956 - val_loss: 0.0552 - val_accuracy: 0.9900
    Epoch 1797/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0220 - accuracy: 0.9949 - val_loss: 0.0529 - val_accuracy: 0.9915
    Epoch 1798/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0203 - accuracy: 0.9946 - val_loss: 0.0504 - val_accuracy: 0.9908
    Epoch 1799/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0192 - accuracy: 0.9962 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1800/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9962 - val_loss: 0.0515 - val_accuracy: 0.9892
    Epoch 1801/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0221 - accuracy: 0.9951 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1802/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0218 - accuracy: 0.9944 - val_loss: 0.0510 - val_accuracy: 0.9892
    Epoch 1803/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0209 - accuracy: 0.9951 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1804/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9956 - val_loss: 0.0508 - val_accuracy: 0.9900
    Epoch 1805/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9962 - val_loss: 0.0522 - val_accuracy: 0.9892
    Epoch 1806/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0220 - accuracy: 0.9938 - val_loss: 0.0513 - val_accuracy: 0.9900
    Epoch 1807/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0226 - accuracy: 0.9936 - val_loss: 0.0557 - val_accuracy: 0.9900
    Epoch 1808/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0217 - accuracy: 0.9941 - val_loss: 0.0599 - val_accuracy: 0.9892
    Epoch 1809/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0211 - accuracy: 0.9949 - val_loss: 0.0555 - val_accuracy: 0.9900
    Epoch 1810/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0203 - accuracy: 0.9954 - val_loss: 0.0587 - val_accuracy: 0.9892
    Epoch 1811/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0208 - accuracy: 0.9954 - val_loss: 0.0542 - val_accuracy: 0.9900
    Epoch 1812/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0200 - accuracy: 0.9962 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1813/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0185 - accuracy: 0.9962 - val_loss: 0.0533 - val_accuracy: 0.9900
    Epoch 1814/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0188 - accuracy: 0.9956 - val_loss: 0.0501 - val_accuracy: 0.9900
    Epoch 1815/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0198 - accuracy: 0.9956 - val_loss: 0.0510 - val_accuracy: 0.9900
    Epoch 1816/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0218 - accuracy: 0.9949 - val_loss: 0.0521 - val_accuracy: 0.9892
    Epoch 1817/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0208 - accuracy: 0.9944 - val_loss: 0.0544 - val_accuracy: 0.9892
    Epoch 1818/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0247 - accuracy: 0.9928 - val_loss: 0.0656 - val_accuracy: 0.9838
    Epoch 1819/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0335 - accuracy: 0.9897 - val_loss: 0.0727 - val_accuracy: 0.9823
    Epoch 1820/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0429 - accuracy: 0.9867 - val_loss: 0.0506 - val_accuracy: 0.9885
    Epoch 1821/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0265 - accuracy: 0.9918 - val_loss: 0.0505 - val_accuracy: 0.9908
    Epoch 1822/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0236 - accuracy: 0.9926 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1823/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0241 - accuracy: 0.9944 - val_loss: 0.0519 - val_accuracy: 0.9892
    Epoch 1824/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0211 - accuracy: 0.9946 - val_loss: 0.0515 - val_accuracy: 0.9892
    Epoch 1825/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0194 - accuracy: 0.9956 - val_loss: 0.0496 - val_accuracy: 0.9908
    Epoch 1826/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0202 - accuracy: 0.9962 - val_loss: 0.0464 - val_accuracy: 0.9900
    Epoch 1827/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9956 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 1828/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0190 - accuracy: 0.9959 - val_loss: 0.0484 - val_accuracy: 0.9908
    Epoch 1829/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0206 - accuracy: 0.9951 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1830/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0191 - accuracy: 0.9951 - val_loss: 0.0582 - val_accuracy: 0.9892
    Epoch 1831/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0213 - accuracy: 0.9956 - val_loss: 0.0534 - val_accuracy: 0.9892
    Epoch 1832/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0238 - accuracy: 0.9936 - val_loss: 0.0490 - val_accuracy: 0.9892
    Epoch 1833/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0228 - accuracy: 0.9928 - val_loss: 0.0498 - val_accuracy: 0.9908
    Epoch 1834/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0246 - accuracy: 0.9920 - val_loss: 0.0490 - val_accuracy: 0.9908
    Epoch 1835/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0202 - accuracy: 0.9956 - val_loss: 0.0525 - val_accuracy: 0.9900
    Epoch 1836/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0195 - accuracy: 0.9959 - val_loss: 0.0527 - val_accuracy: 0.9900
    Epoch 1837/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0190 - accuracy: 0.9964 - val_loss: 0.0474 - val_accuracy: 0.9892
    Epoch 1838/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0190 - accuracy: 0.9959 - val_loss: 0.0502 - val_accuracy: 0.9900
    Epoch 1839/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0190 - accuracy: 0.9962 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1840/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0185 - accuracy: 0.9956 - val_loss: 0.0547 - val_accuracy: 0.9900
    Epoch 1841/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0191 - accuracy: 0.9959 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1842/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0192 - accuracy: 0.9956 - val_loss: 0.0496 - val_accuracy: 0.9900
    Epoch 1843/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0185 - accuracy: 0.9959 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1844/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0186 - accuracy: 0.9962 - val_loss: 0.0474 - val_accuracy: 0.9900
    Epoch 1845/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0188 - accuracy: 0.9959 - val_loss: 0.0538 - val_accuracy: 0.9900
    Epoch 1846/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0207 - accuracy: 0.9954 - val_loss: 0.0528 - val_accuracy: 0.9892
    Epoch 1847/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0213 - accuracy: 0.9944 - val_loss: 0.0605 - val_accuracy: 0.9877
    Epoch 1848/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0208 - accuracy: 0.9941 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1849/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0230 - accuracy: 0.9944 - val_loss: 0.0489 - val_accuracy: 0.9900
    Epoch 1850/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0208 - accuracy: 0.9951 - val_loss: 0.0495 - val_accuracy: 0.9892
    Epoch 1851/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0196 - accuracy: 0.9951 - val_loss: 0.0533 - val_accuracy: 0.9900
    Epoch 1852/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0198 - accuracy: 0.9964 - val_loss: 0.0509 - val_accuracy: 0.9892
    Epoch 1853/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0201 - accuracy: 0.9951 - val_loss: 0.0486 - val_accuracy: 0.9900
    Epoch 1854/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0198 - accuracy: 0.9959 - val_loss: 0.0478 - val_accuracy: 0.9900
    Epoch 1855/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0185 - accuracy: 0.9962 - val_loss: 0.0519 - val_accuracy: 0.9900
    Epoch 1856/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0189 - accuracy: 0.9959 - val_loss: 0.0503 - val_accuracy: 0.9900
    Epoch 1857/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0190 - accuracy: 0.9956 - val_loss: 0.0564 - val_accuracy: 0.9892
    Epoch 1858/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0192 - accuracy: 0.9956 - val_loss: 0.0523 - val_accuracy: 0.9900
    Epoch 1859/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0186 - accuracy: 0.9962 - val_loss: 0.0516 - val_accuracy: 0.9900
    Epoch 1860/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0185 - accuracy: 0.9956 - val_loss: 0.0508 - val_accuracy: 0.9892
    Epoch 1861/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0187 - accuracy: 0.9962 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1862/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0182 - accuracy: 0.9959 - val_loss: 0.0480 - val_accuracy: 0.9892
    Epoch 1863/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0183 - accuracy: 0.9964 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1864/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0184 - accuracy: 0.9964 - val_loss: 0.0527 - val_accuracy: 0.9892
    Epoch 1865/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0201 - accuracy: 0.9954 - val_loss: 0.0565 - val_accuracy: 0.9900
    Epoch 1866/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0199 - accuracy: 0.9951 - val_loss: 0.0616 - val_accuracy: 0.9877
    Epoch 1867/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0207 - accuracy: 0.9946 - val_loss: 0.0546 - val_accuracy: 0.9892
    Epoch 1868/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9949 - val_loss: 0.0517 - val_accuracy: 0.9900
    Epoch 1869/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0193 - accuracy: 0.9964 - val_loss: 0.0519 - val_accuracy: 0.9892
    Epoch 1870/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0204 - accuracy: 0.9956 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1871/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0183 - accuracy: 0.9962 - val_loss: 0.0508 - val_accuracy: 0.9900
    Epoch 1872/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0180 - accuracy: 0.9962 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1873/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0183 - accuracy: 0.9962 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1874/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0184 - accuracy: 0.9964 - val_loss: 0.0492 - val_accuracy: 0.9892
    Epoch 1875/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0192 - accuracy: 0.9959 - val_loss: 0.0501 - val_accuracy: 0.9892
    Epoch 1876/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0221 - accuracy: 0.9938 - val_loss: 0.0554 - val_accuracy: 0.9877
    Epoch 1877/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0231 - accuracy: 0.9941 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1878/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0206 - accuracy: 0.9951 - val_loss: 0.0576 - val_accuracy: 0.9892
    Epoch 1879/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0211 - accuracy: 0.9936 - val_loss: 0.0588 - val_accuracy: 0.9900
    Epoch 1880/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0196 - accuracy: 0.9949 - val_loss: 0.0508 - val_accuracy: 0.9892
    Epoch 1881/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0204 - accuracy: 0.9941 - val_loss: 0.0612 - val_accuracy: 0.9869
    Epoch 1882/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0202 - accuracy: 0.9956 - val_loss: 0.0538 - val_accuracy: 0.9900
    Epoch 1883/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9946 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1884/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0195 - accuracy: 0.9956 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1885/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0185 - accuracy: 0.9964 - val_loss: 0.0473 - val_accuracy: 0.9900
    Epoch 1886/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0183 - accuracy: 0.9962 - val_loss: 0.0503 - val_accuracy: 0.9892
    Epoch 1887/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0214 - accuracy: 0.9946 - val_loss: 0.0533 - val_accuracy: 0.9892
    Epoch 1888/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0222 - accuracy: 0.9944 - val_loss: 0.0513 - val_accuracy: 0.9908
    Epoch 1889/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0219 - accuracy: 0.9949 - val_loss: 0.0538 - val_accuracy: 0.9885
    Epoch 1890/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0200 - accuracy: 0.9951 - val_loss: 0.0564 - val_accuracy: 0.9892
    Epoch 1891/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0187 - accuracy: 0.9962 - val_loss: 0.0537 - val_accuracy: 0.9892
    Epoch 1892/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0189 - accuracy: 0.9954 - val_loss: 0.0542 - val_accuracy: 0.9908
    Epoch 1893/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0189 - accuracy: 0.9962 - val_loss: 0.0524 - val_accuracy: 0.9900
    Epoch 1894/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0189 - accuracy: 0.9962 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1895/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9954 - val_loss: 0.0538 - val_accuracy: 0.9892
    Epoch 1896/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0187 - accuracy: 0.9956 - val_loss: 0.0520 - val_accuracy: 0.9892
    Epoch 1897/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0196 - accuracy: 0.9949 - val_loss: 0.0561 - val_accuracy: 0.9892
    Epoch 1898/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0200 - accuracy: 0.9951 - val_loss: 0.0620 - val_accuracy: 0.9885
    Epoch 1899/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0219 - accuracy: 0.9944 - val_loss: 0.0550 - val_accuracy: 0.9900
    Epoch 1900/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0212 - accuracy: 0.9941 - val_loss: 0.0747 - val_accuracy: 0.9854
    Epoch 1901/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0240 - accuracy: 0.9938 - val_loss: 0.0560 - val_accuracy: 0.9900
    Epoch 1902/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0195 - accuracy: 0.9949 - val_loss: 0.0539 - val_accuracy: 0.9900
    Epoch 1903/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0184 - accuracy: 0.9962 - val_loss: 0.0528 - val_accuracy: 0.9908
    Epoch 1904/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0188 - accuracy: 0.9959 - val_loss: 0.0522 - val_accuracy: 0.9892
    Epoch 1905/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0183 - accuracy: 0.9954 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1906/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0182 - accuracy: 0.9962 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1907/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0182 - accuracy: 0.9954 - val_loss: 0.0543 - val_accuracy: 0.9892
    Epoch 1908/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0192 - accuracy: 0.9962 - val_loss: 0.0533 - val_accuracy: 0.9892
    Epoch 1909/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0201 - accuracy: 0.9954 - val_loss: 0.0498 - val_accuracy: 0.9908
    Epoch 1910/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0185 - accuracy: 0.9962 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1911/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0189 - accuracy: 0.9959 - val_loss: 0.0520 - val_accuracy: 0.9900
    Epoch 1912/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0193 - accuracy: 0.9951 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1913/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0189 - accuracy: 0.9964 - val_loss: 0.0494 - val_accuracy: 0.9900
    Epoch 1914/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0193 - accuracy: 0.9949 - val_loss: 0.0497 - val_accuracy: 0.9892
    Epoch 1915/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0180 - accuracy: 0.9959 - val_loss: 0.0563 - val_accuracy: 0.9892
    Epoch 1916/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0178 - accuracy: 0.9959 - val_loss: 0.0493 - val_accuracy: 0.9900
    Epoch 1917/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0178 - accuracy: 0.9964 - val_loss: 0.0483 - val_accuracy: 0.9900
    Epoch 1918/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0195 - accuracy: 0.9954 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1919/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0181 - accuracy: 0.9962 - val_loss: 0.0498 - val_accuracy: 0.9900
    Epoch 1920/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0213 - accuracy: 0.9941 - val_loss: 0.0576 - val_accuracy: 0.9869
    Epoch 1921/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0212 - accuracy: 0.9941 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1922/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9956 - val_loss: 0.0514 - val_accuracy: 0.9900
    Epoch 1923/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0184 - accuracy: 0.9959 - val_loss: 0.0524 - val_accuracy: 0.9892
    Epoch 1924/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0190 - accuracy: 0.9951 - val_loss: 0.0536 - val_accuracy: 0.9908
    Epoch 1925/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0178 - accuracy: 0.9964 - val_loss: 0.0502 - val_accuracy: 0.9908
    Epoch 1926/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0186 - accuracy: 0.9954 - val_loss: 0.0509 - val_accuracy: 0.9900
    Epoch 1927/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0182 - accuracy: 0.9964 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1928/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0223 - accuracy: 0.9946 - val_loss: 0.0603 - val_accuracy: 0.9869
    Epoch 1929/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0254 - accuracy: 0.9915 - val_loss: 0.0654 - val_accuracy: 0.9854
    Epoch 1930/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0277 - accuracy: 0.9910 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1931/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0239 - accuracy: 0.9938 - val_loss: 0.0522 - val_accuracy: 0.9900
    Epoch 1932/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0219 - accuracy: 0.9941 - val_loss: 0.0523 - val_accuracy: 0.9900
    Epoch 1933/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0224 - accuracy: 0.9936 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1934/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0217 - accuracy: 0.9941 - val_loss: 0.0506 - val_accuracy: 0.9900
    Epoch 1935/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9936 - val_loss: 0.0535 - val_accuracy: 0.9892
    Epoch 1936/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0185 - accuracy: 0.9956 - val_loss: 0.0539 - val_accuracy: 0.9892
    Epoch 1937/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0187 - accuracy: 0.9956 - val_loss: 0.0509 - val_accuracy: 0.9892
    Epoch 1938/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0186 - accuracy: 0.9964 - val_loss: 0.0480 - val_accuracy: 0.9900
    Epoch 1939/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0182 - accuracy: 0.9962 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1940/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0179 - accuracy: 0.9959 - val_loss: 0.0495 - val_accuracy: 0.9900
    Epoch 1941/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0174 - accuracy: 0.9964 - val_loss: 0.0551 - val_accuracy: 0.9892
    Epoch 1942/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0185 - accuracy: 0.9959 - val_loss: 0.0539 - val_accuracy: 0.9900
    Epoch 1943/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0188 - accuracy: 0.9951 - val_loss: 0.0541 - val_accuracy: 0.9892
    Epoch 1944/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0190 - accuracy: 0.9949 - val_loss: 0.0507 - val_accuracy: 0.9900
    Epoch 1945/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0180 - accuracy: 0.9956 - val_loss: 0.0520 - val_accuracy: 0.9900
    Epoch 1946/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0185 - accuracy: 0.9954 - val_loss: 0.0515 - val_accuracy: 0.9900
    Epoch 1947/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0183 - accuracy: 0.9956 - val_loss: 0.0538 - val_accuracy: 0.9900
    Epoch 1948/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0178 - accuracy: 0.9964 - val_loss: 0.0490 - val_accuracy: 0.9892
    Epoch 1949/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0186 - accuracy: 0.9959 - val_loss: 0.0506 - val_accuracy: 0.9900
    Epoch 1950/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0189 - accuracy: 0.9951 - val_loss: 0.0542 - val_accuracy: 0.9892
    Epoch 1951/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9954 - val_loss: 0.0595 - val_accuracy: 0.9885
    Epoch 1952/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0192 - accuracy: 0.9959 - val_loss: 0.0538 - val_accuracy: 0.9900
    Epoch 1953/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.0179 - accuracy: 0.9964 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1954/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9962 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1955/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0191 - accuracy: 0.9962 - val_loss: 0.0487 - val_accuracy: 0.9900
    Epoch 1956/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0197 - accuracy: 0.9946 - val_loss: 0.0526 - val_accuracy: 0.9900
    Epoch 1957/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0198 - accuracy: 0.9956 - val_loss: 0.0540 - val_accuracy: 0.9900
    Epoch 1958/2000
    8/8 [==============================] - 0s 25ms/step - loss: 0.0197 - accuracy: 0.9946 - val_loss: 0.0491 - val_accuracy: 0.9900
    Epoch 1959/2000
    8/8 [==============================] - 0s 38ms/step - loss: 0.0174 - accuracy: 0.9964 - val_loss: 0.0572 - val_accuracy: 0.9892
    Epoch 1960/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0189 - accuracy: 0.9962 - val_loss: 0.0561 - val_accuracy: 0.9900
    Epoch 1961/2000
    8/8 [==============================] - 0s 24ms/step - loss: 0.0176 - accuracy: 0.9959 - val_loss: 0.0509 - val_accuracy: 0.9892
    Epoch 1962/2000
    8/8 [==============================] - 0s 25ms/step - loss: 0.0180 - accuracy: 0.9959 - val_loss: 0.0532 - val_accuracy: 0.9900
    Epoch 1963/2000
    8/8 [==============================] - 0s 28ms/step - loss: 0.0186 - accuracy: 0.9956 - val_loss: 0.0605 - val_accuracy: 0.9877
    Epoch 1964/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0235 - accuracy: 0.9926 - val_loss: 0.0492 - val_accuracy: 0.9900
    Epoch 1965/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0187 - accuracy: 0.9951 - val_loss: 0.0513 - val_accuracy: 0.9900
    Epoch 1966/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0181 - accuracy: 0.9956 - val_loss: 0.0571 - val_accuracy: 0.9892
    Epoch 1967/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0177 - accuracy: 0.9967 - val_loss: 0.0557 - val_accuracy: 0.9900
    Epoch 1968/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0171 - accuracy: 0.9962 - val_loss: 0.0505 - val_accuracy: 0.9900
    Epoch 1969/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0195 - accuracy: 0.9954 - val_loss: 0.0499 - val_accuracy: 0.9900
    Epoch 1970/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0180 - accuracy: 0.9956 - val_loss: 0.0553 - val_accuracy: 0.9892
    Epoch 1971/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0176 - accuracy: 0.9964 - val_loss: 0.0545 - val_accuracy: 0.9900
    Epoch 1972/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0193 - accuracy: 0.9949 - val_loss: 0.0532 - val_accuracy: 0.9900
    Epoch 1973/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0177 - accuracy: 0.9964 - val_loss: 0.0524 - val_accuracy: 0.9885
    Epoch 1974/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0182 - accuracy: 0.9956 - val_loss: 0.0498 - val_accuracy: 0.9892
    Epoch 1975/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0176 - accuracy: 0.9959 - val_loss: 0.0490 - val_accuracy: 0.9900
    Epoch 1976/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0174 - accuracy: 0.9959 - val_loss: 0.0512 - val_accuracy: 0.9900
    Epoch 1977/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0170 - accuracy: 0.9967 - val_loss: 0.0508 - val_accuracy: 0.9892
    Epoch 1978/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0170 - accuracy: 0.9964 - val_loss: 0.0513 - val_accuracy: 0.9908
    Epoch 1979/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0170 - accuracy: 0.9964 - val_loss: 0.0527 - val_accuracy: 0.9892
    Epoch 1980/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0183 - accuracy: 0.9954 - val_loss: 0.0524 - val_accuracy: 0.9900
    Epoch 1981/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0177 - accuracy: 0.9956 - val_loss: 0.0511 - val_accuracy: 0.9900
    Epoch 1982/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0181 - accuracy: 0.9959 - val_loss: 0.0519 - val_accuracy: 0.9900
    Epoch 1983/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0168 - accuracy: 0.9962 - val_loss: 0.0508 - val_accuracy: 0.9900
    Epoch 1984/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0179 - accuracy: 0.9959 - val_loss: 0.0574 - val_accuracy: 0.9885
    Epoch 1985/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0195 - accuracy: 0.9956 - val_loss: 0.0517 - val_accuracy: 0.9900
    Epoch 1986/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0184 - accuracy: 0.9954 - val_loss: 0.0504 - val_accuracy: 0.9900
    Epoch 1987/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0188 - accuracy: 0.9949 - val_loss: 0.0514 - val_accuracy: 0.9900
    Epoch 1988/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0189 - accuracy: 0.9959 - val_loss: 0.0500 - val_accuracy: 0.9900
    Epoch 1989/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0174 - accuracy: 0.9964 - val_loss: 0.0497 - val_accuracy: 0.9900
    Epoch 1990/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0173 - accuracy: 0.9959 - val_loss: 0.0536 - val_accuracy: 0.9900
    Epoch 1991/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0193 - accuracy: 0.9959 - val_loss: 0.0521 - val_accuracy: 0.9900
    Epoch 1992/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0191 - accuracy: 0.9956 - val_loss: 0.0530 - val_accuracy: 0.9908
    Epoch 1993/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0235 - accuracy: 0.9933 - val_loss: 0.0525 - val_accuracy: 0.9892
    Epoch 1994/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0194 - accuracy: 0.9944 - val_loss: 0.0485 - val_accuracy: 0.9908
    Epoch 1995/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0201 - accuracy: 0.9954 - val_loss: 0.0546 - val_accuracy: 0.9892
    Epoch 1996/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0239 - accuracy: 0.9913 - val_loss: 0.0668 - val_accuracy: 0.9854
    Epoch 1997/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0362 - accuracy: 0.9882 - val_loss: 0.0618 - val_accuracy: 0.9862
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
    Epoch 4/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.3014 - accuracy: 0.8753 - val_loss: 0.3327 - val_accuracy: 0.8592
    Epoch 5/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.3146 - accuracy: 0.8676 - val_loss: 0.3123 - val_accuracy: 0.8692
    Epoch 6/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.2813 - accuracy: 0.8820 - val_loss: 0.2762 - val_accuracy: 0.8877
    Epoch 7/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.2613 - accuracy: 0.9048 - val_loss: 0.2523 - val_accuracy: 0.8969
    Epoch 8/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.2367 - accuracy: 0.9110 - val_loss: 0.2358 - val_accuracy: 0.9077
    Epoch 9/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.2239 - accuracy: 0.9158 - val_loss: 0.2235 - val_accuracy: 0.9100
    Epoch 10/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.2177 - accuracy: 0.9197 - val_loss: 0.2221 - val_accuracy: 0.9146
    Epoch 11/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.2135 - accuracy: 0.9215 - val_loss: 0.2169 - val_accuracy: 0.9146
    Epoch 12/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.2094 - accuracy: 0.9225 - val_loss: 0.2132 - val_accuracy: 0.9162
    Epoch 13/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.2055 - accuracy: 0.9240 - val_loss: 0.2108 - val_accuracy: 0.9208
    Epoch 14/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.2028 - accuracy: 0.9274 - val_loss: 0.2067 - val_accuracy: 0.9208
    Epoch 15/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.2006 - accuracy: 0.9294 - val_loss: 0.2024 - val_accuracy: 0.9285
    Epoch 16/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.1960 - accuracy: 0.9310 - val_loss: 0.2007 - val_accuracy: 0.9246
    Epoch 17/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.1945 - accuracy: 0.9310 - val_loss: 0.1975 - val_accuracy: 0.9277
    Epoch 18/2000
    8/8 [==============================] - 0s 19ms/step - loss: 0.1925 - accuracy: 0.9312 - val_loss: 0.1968 - val_accuracy: 0.9277
    Epoch 19/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.1907 - accuracy: 0.9346 - val_loss: 0.1949 - val_accuracy: 0.9292
    Epoch 20/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.1892 - accuracy: 0.9341 - val_loss: 0.1915 - val_accuracy: 0.9292
    Epoch 21/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1878 - accuracy: 0.9369 - val_loss: 0.1928 - val_accuracy: 0.9315
    Epoch 22/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.1864 - accuracy: 0.9358 - val_loss: 0.1889 - val_accuracy: 0.9346
    Epoch 23/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1850 - accuracy: 0.9366 - val_loss: 0.1883 - val_accuracy: 0.9354
    Epoch 24/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1835 - accuracy: 0.9369 - val_loss: 0.1855 - val_accuracy: 0.9362
    Epoch 25/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1814 - accuracy: 0.9379 - val_loss: 0.1844 - val_accuracy: 0.9369
    Epoch 26/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1797 - accuracy: 0.9384 - val_loss: 0.1844 - val_accuracy: 0.9385
    Epoch 27/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1780 - accuracy: 0.9374 - val_loss: 0.1815 - val_accuracy: 0.9385
    Epoch 28/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1767 - accuracy: 0.9394 - val_loss: 0.1801 - val_accuracy: 0.9385
    Epoch 29/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.1751 - accuracy: 0.9392 - val_loss: 0.1802 - val_accuracy: 0.9385
    Epoch 30/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1737 - accuracy: 0.9402 - val_loss: 0.1769 - val_accuracy: 0.9392
    Epoch 31/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1724 - accuracy: 0.9397 - val_loss: 0.1749 - val_accuracy: 0.9392
    Epoch 32/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.1701 - accuracy: 0.9410 - val_loss: 0.1756 - val_accuracy: 0.9385
    Epoch 33/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1690 - accuracy: 0.9400 - val_loss: 0.1704 - val_accuracy: 0.9400
    Epoch 34/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1662 - accuracy: 0.9410 - val_loss: 0.1677 - val_accuracy: 0.9408
    Epoch 35/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1622 - accuracy: 0.9402 - val_loss: 0.1622 - val_accuracy: 0.9400
    Epoch 36/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1590 - accuracy: 0.9423 - val_loss: 0.1574 - val_accuracy: 0.9408
    Epoch 37/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1565 - accuracy: 0.9435 - val_loss: 0.1546 - val_accuracy: 0.9431
    Epoch 38/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1538 - accuracy: 0.9451 - val_loss: 0.1530 - val_accuracy: 0.9431
    Epoch 39/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1507 - accuracy: 0.9456 - val_loss: 0.1495 - val_accuracy: 0.9477
    Epoch 40/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.1485 - accuracy: 0.9469 - val_loss: 0.1479 - val_accuracy: 0.9477
    Epoch 41/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.1454 - accuracy: 0.9474 - val_loss: 0.1537 - val_accuracy: 0.9454
    Epoch 42/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1500 - accuracy: 0.9453 - val_loss: 0.1454 - val_accuracy: 0.9500
    Epoch 43/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1486 - accuracy: 0.9474 - val_loss: 0.1428 - val_accuracy: 0.9538
    Epoch 44/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1436 - accuracy: 0.9492 - val_loss: 0.1427 - val_accuracy: 0.9477
    Epoch 45/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1394 - accuracy: 0.9484 - val_loss: 0.1420 - val_accuracy: 0.9477
    Epoch 46/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1370 - accuracy: 0.9505 - val_loss: 0.1380 - val_accuracy: 0.9492
    Epoch 47/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1398 - accuracy: 0.9494 - val_loss: 0.1337 - val_accuracy: 0.9508
    Epoch 48/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1354 - accuracy: 0.9497 - val_loss: 0.1272 - val_accuracy: 0.9562
    Epoch 49/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1301 - accuracy: 0.9500 - val_loss: 0.1251 - val_accuracy: 0.9569
    Epoch 50/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.1290 - accuracy: 0.9497 - val_loss: 0.1294 - val_accuracy: 0.9585
    Epoch 51/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1308 - accuracy: 0.9512 - val_loss: 0.1214 - val_accuracy: 0.9615
    Epoch 52/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1256 - accuracy: 0.9536 - val_loss: 0.1201 - val_accuracy: 0.9562
    Epoch 53/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1234 - accuracy: 0.9523 - val_loss: 0.1180 - val_accuracy: 0.9615
    Epoch 54/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1229 - accuracy: 0.9528 - val_loss: 0.1169 - val_accuracy: 0.9600
    Epoch 55/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1207 - accuracy: 0.9523 - val_loss: 0.1161 - val_accuracy: 0.9569
    Epoch 56/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.1222 - accuracy: 0.9520 - val_loss: 0.1172 - val_accuracy: 0.9662
    Epoch 57/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1255 - accuracy: 0.9559 - val_loss: 0.1144 - val_accuracy: 0.9646
    Epoch 58/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1236 - accuracy: 0.9579 - val_loss: 0.1128 - val_accuracy: 0.9638
    Epoch 59/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1195 - accuracy: 0.9587 - val_loss: 0.1117 - val_accuracy: 0.9638
    Epoch 60/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1170 - accuracy: 0.9592 - val_loss: 0.1102 - val_accuracy: 0.9615
    Epoch 61/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1145 - accuracy: 0.9600 - val_loss: 0.1138 - val_accuracy: 0.9554
    Epoch 62/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.1143 - accuracy: 0.9582 - val_loss: 0.1112 - val_accuracy: 0.9577
    Epoch 63/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.1140 - accuracy: 0.9589 - val_loss: 0.1118 - val_accuracy: 0.9562
    Epoch 64/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1119 - accuracy: 0.9592 - val_loss: 0.1071 - val_accuracy: 0.9615
    Epoch 65/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1109 - accuracy: 0.9589 - val_loss: 0.1069 - val_accuracy: 0.9615
    Epoch 66/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1100 - accuracy: 0.9610 - val_loss: 0.1050 - val_accuracy: 0.9638
    Epoch 67/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1086 - accuracy: 0.9597 - val_loss: 0.1040 - val_accuracy: 0.9646
    Epoch 68/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.1079 - accuracy: 0.9602 - val_loss: 0.1036 - val_accuracy: 0.9646
    Epoch 69/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.1072 - accuracy: 0.9625 - val_loss: 0.1043 - val_accuracy: 0.9608
    Epoch 70/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.1069 - accuracy: 0.9613 - val_loss: 0.1023 - val_accuracy: 0.9662
    Epoch 71/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.1080 - accuracy: 0.9615 - val_loss: 0.1067 - val_accuracy: 0.9700
    Epoch 72/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1075 - accuracy: 0.9630 - val_loss: 0.0996 - val_accuracy: 0.9662
    Epoch 73/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1044 - accuracy: 0.9648 - val_loss: 0.0989 - val_accuracy: 0.9662
    Epoch 74/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1035 - accuracy: 0.9628 - val_loss: 0.0987 - val_accuracy: 0.9669
    Epoch 75/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.1039 - accuracy: 0.9666 - val_loss: 0.0987 - val_accuracy: 0.9700
    Epoch 76/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.1020 - accuracy: 0.9659 - val_loss: 0.0964 - val_accuracy: 0.9669
    Epoch 77/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.1004 - accuracy: 0.9664 - val_loss: 0.0960 - val_accuracy: 0.9685
    Epoch 78/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0994 - accuracy: 0.9677 - val_loss: 0.0978 - val_accuracy: 0.9654
    Epoch 79/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0992 - accuracy: 0.9687 - val_loss: 0.0958 - val_accuracy: 0.9654
    Epoch 80/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0978 - accuracy: 0.9682 - val_loss: 0.0940 - val_accuracy: 0.9677
    Epoch 81/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0970 - accuracy: 0.9707 - val_loss: 0.0934 - val_accuracy: 0.9677
    Epoch 82/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0982 - accuracy: 0.9687 - val_loss: 0.0961 - val_accuracy: 0.9662
    Epoch 83/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0964 - accuracy: 0.9695 - val_loss: 0.0922 - val_accuracy: 0.9700
    Epoch 84/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0968 - accuracy: 0.9690 - val_loss: 0.0962 - val_accuracy: 0.9731
    Epoch 85/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0985 - accuracy: 0.9684 - val_loss: 0.0930 - val_accuracy: 0.9723
    Epoch 86/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0940 - accuracy: 0.9702 - val_loss: 0.0904 - val_accuracy: 0.9692
    Epoch 87/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0933 - accuracy: 0.9707 - val_loss: 0.0895 - val_accuracy: 0.9723
    Epoch 88/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0944 - accuracy: 0.9713 - val_loss: 0.0894 - val_accuracy: 0.9708
    Epoch 89/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0919 - accuracy: 0.9713 - val_loss: 0.0893 - val_accuracy: 0.9723
    Epoch 90/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0926 - accuracy: 0.9697 - val_loss: 0.0919 - val_accuracy: 0.9731
    Epoch 91/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0931 - accuracy: 0.9687 - val_loss: 0.0864 - val_accuracy: 0.9715
    Epoch 92/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0916 - accuracy: 0.9723 - val_loss: 0.0864 - val_accuracy: 0.9715
    Epoch 93/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0942 - accuracy: 0.9697 - val_loss: 0.0854 - val_accuracy: 0.9723
    Epoch 94/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0964 - accuracy: 0.9682 - val_loss: 0.0858 - val_accuracy: 0.9738
    Epoch 95/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0893 - accuracy: 0.9710 - val_loss: 0.0882 - val_accuracy: 0.9700
    Epoch 96/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0881 - accuracy: 0.9715 - val_loss: 0.0848 - val_accuracy: 0.9731
    Epoch 97/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0865 - accuracy: 0.9749 - val_loss: 0.0844 - val_accuracy: 0.9731
    Epoch 98/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0869 - accuracy: 0.9728 - val_loss: 0.0869 - val_accuracy: 0.9738
    Epoch 99/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0878 - accuracy: 0.9733 - val_loss: 0.0820 - val_accuracy: 0.9746
    Epoch 100/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0858 - accuracy: 0.9738 - val_loss: 0.0831 - val_accuracy: 0.9738
    Epoch 101/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0875 - accuracy: 0.9723 - val_loss: 0.0856 - val_accuracy: 0.9692
    Epoch 102/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0845 - accuracy: 0.9743 - val_loss: 0.0807 - val_accuracy: 0.9754
    Epoch 103/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0838 - accuracy: 0.9743 - val_loss: 0.0817 - val_accuracy: 0.9746
    Epoch 104/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0837 - accuracy: 0.9751 - val_loss: 0.0810 - val_accuracy: 0.9738
    Epoch 105/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0864 - accuracy: 0.9728 - val_loss: 0.0797 - val_accuracy: 0.9746
    Epoch 106/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0830 - accuracy: 0.9733 - val_loss: 0.0787 - val_accuracy: 0.9785
    Epoch 107/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0824 - accuracy: 0.9749 - val_loss: 0.0800 - val_accuracy: 0.9762
    Epoch 108/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0819 - accuracy: 0.9754 - val_loss: 0.0779 - val_accuracy: 0.9777
    Epoch 109/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0819 - accuracy: 0.9749 - val_loss: 0.0804 - val_accuracy: 0.9738
    Epoch 110/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0830 - accuracy: 0.9764 - val_loss: 0.0869 - val_accuracy: 0.9677
    Epoch 111/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0889 - accuracy: 0.9713 - val_loss: 0.0819 - val_accuracy: 0.9731
    Epoch 112/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0896 - accuracy: 0.9715 - val_loss: 0.0819 - val_accuracy: 0.9731
    Epoch 113/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0868 - accuracy: 0.9728 - val_loss: 0.0766 - val_accuracy: 0.9785
    Epoch 114/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0807 - accuracy: 0.9756 - val_loss: 0.0778 - val_accuracy: 0.9754
    Epoch 115/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0832 - accuracy: 0.9749 - val_loss: 0.0778 - val_accuracy: 0.9762
    Epoch 116/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0808 - accuracy: 0.9759 - val_loss: 0.0852 - val_accuracy: 0.9723
    Epoch 117/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0822 - accuracy: 0.9741 - val_loss: 0.0758 - val_accuracy: 0.9762
    Epoch 118/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0802 - accuracy: 0.9764 - val_loss: 0.0742 - val_accuracy: 0.9769
    Epoch 119/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0780 - accuracy: 0.9759 - val_loss: 0.0739 - val_accuracy: 0.9785
    Epoch 120/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0764 - accuracy: 0.9761 - val_loss: 0.0747 - val_accuracy: 0.9769
    Epoch 121/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0789 - accuracy: 0.9756 - val_loss: 0.0728 - val_accuracy: 0.9785
    Epoch 122/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0781 - accuracy: 0.9766 - val_loss: 0.0740 - val_accuracy: 0.9785
    Epoch 123/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0775 - accuracy: 0.9769 - val_loss: 0.0752 - val_accuracy: 0.9762
    Epoch 124/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0763 - accuracy: 0.9787 - val_loss: 0.0726 - val_accuracy: 0.9785
    Epoch 125/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0754 - accuracy: 0.9772 - val_loss: 0.0726 - val_accuracy: 0.9777
    Epoch 126/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0749 - accuracy: 0.9774 - val_loss: 0.0727 - val_accuracy: 0.9777
    Epoch 127/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0740 - accuracy: 0.9764 - val_loss: 0.0707 - val_accuracy: 0.9792
    Epoch 128/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0740 - accuracy: 0.9772 - val_loss: 0.0716 - val_accuracy: 0.9777
    Epoch 129/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0737 - accuracy: 0.9769 - val_loss: 0.0726 - val_accuracy: 0.9769
    Epoch 130/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0746 - accuracy: 0.9772 - val_loss: 0.0699 - val_accuracy: 0.9785
    Epoch 131/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0734 - accuracy: 0.9777 - val_loss: 0.0695 - val_accuracy: 0.9792
    Epoch 132/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0723 - accuracy: 0.9772 - val_loss: 0.0691 - val_accuracy: 0.9785
    Epoch 133/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0724 - accuracy: 0.9782 - val_loss: 0.0748 - val_accuracy: 0.9754
    Epoch 134/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0752 - accuracy: 0.9759 - val_loss: 0.0755 - val_accuracy: 0.9754
    Epoch 135/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0742 - accuracy: 0.9774 - val_loss: 0.0727 - val_accuracy: 0.9754
    Epoch 136/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0728 - accuracy: 0.9774 - val_loss: 0.0682 - val_accuracy: 0.9777
    Epoch 137/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0718 - accuracy: 0.9787 - val_loss: 0.0683 - val_accuracy: 0.9769
    Epoch 138/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0714 - accuracy: 0.9772 - val_loss: 0.0673 - val_accuracy: 0.9800
    Epoch 139/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0708 - accuracy: 0.9769 - val_loss: 0.0728 - val_accuracy: 0.9762
    Epoch 140/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0727 - accuracy: 0.9774 - val_loss: 0.0687 - val_accuracy: 0.9800
    Epoch 141/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0713 - accuracy: 0.9774 - val_loss: 0.0696 - val_accuracy: 0.9792
    Epoch 142/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0716 - accuracy: 0.9769 - val_loss: 0.0791 - val_accuracy: 0.9746
    Epoch 143/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0773 - accuracy: 0.9738 - val_loss: 0.0708 - val_accuracy: 0.9777
    Epoch 144/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0733 - accuracy: 0.9782 - val_loss: 0.0651 - val_accuracy: 0.9800
    Epoch 145/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0724 - accuracy: 0.9779 - val_loss: 0.0655 - val_accuracy: 0.9800
    Epoch 146/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0719 - accuracy: 0.9769 - val_loss: 0.0681 - val_accuracy: 0.9769
    Epoch 147/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0719 - accuracy: 0.9769 - val_loss: 0.0663 - val_accuracy: 0.9785
    Epoch 148/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0695 - accuracy: 0.9790 - val_loss: 0.0725 - val_accuracy: 0.9754
    Epoch 149/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0719 - accuracy: 0.9769 - val_loss: 0.0643 - val_accuracy: 0.9800
    Epoch 150/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0704 - accuracy: 0.9784 - val_loss: 0.0696 - val_accuracy: 0.9792
    Epoch 151/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0696 - accuracy: 0.9790 - val_loss: 0.0649 - val_accuracy: 0.9769
    Epoch 152/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0681 - accuracy: 0.9784 - val_loss: 0.0634 - val_accuracy: 0.9808
    Epoch 153/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0678 - accuracy: 0.9779 - val_loss: 0.0638 - val_accuracy: 0.9792
    Epoch 154/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0674 - accuracy: 0.9787 - val_loss: 0.0638 - val_accuracy: 0.9792
    Epoch 155/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0679 - accuracy: 0.9782 - val_loss: 0.0642 - val_accuracy: 0.9792
    Epoch 156/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0705 - accuracy: 0.9787 - val_loss: 0.0682 - val_accuracy: 0.9785
    Epoch 157/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0675 - accuracy: 0.9787 - val_loss: 0.0652 - val_accuracy: 0.9785
    Epoch 158/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0676 - accuracy: 0.9784 - val_loss: 0.0633 - val_accuracy: 0.9800
    Epoch 159/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0664 - accuracy: 0.9790 - val_loss: 0.0626 - val_accuracy: 0.9800
    Epoch 160/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0665 - accuracy: 0.9782 - val_loss: 0.0631 - val_accuracy: 0.9800
    Epoch 161/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0665 - accuracy: 0.9787 - val_loss: 0.0721 - val_accuracy: 0.9762
    Epoch 162/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0718 - accuracy: 0.9790 - val_loss: 0.0630 - val_accuracy: 0.9808
    Epoch 163/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0655 - accuracy: 0.9782 - val_loss: 0.0616 - val_accuracy: 0.9815
    Epoch 164/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0663 - accuracy: 0.9782 - val_loss: 0.0635 - val_accuracy: 0.9808
    Epoch 165/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0667 - accuracy: 0.9805 - val_loss: 0.0636 - val_accuracy: 0.9808
    Epoch 166/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0677 - accuracy: 0.9787 - val_loss: 0.0660 - val_accuracy: 0.9800
    Epoch 167/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0691 - accuracy: 0.9769 - val_loss: 0.0730 - val_accuracy: 0.9762
    Epoch 168/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0686 - accuracy: 0.9782 - val_loss: 0.0644 - val_accuracy: 0.9785
    Epoch 169/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0678 - accuracy: 0.9784 - val_loss: 0.0633 - val_accuracy: 0.9792
    Epoch 170/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0682 - accuracy: 0.9779 - val_loss: 0.0624 - val_accuracy: 0.9800
    Epoch 171/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0670 - accuracy: 0.9792 - val_loss: 0.0625 - val_accuracy: 0.9792
    Epoch 172/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0644 - accuracy: 0.9790 - val_loss: 0.0600 - val_accuracy: 0.9808
    Epoch 173/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0642 - accuracy: 0.9792 - val_loss: 0.0607 - val_accuracy: 0.9808
    Epoch 174/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0647 - accuracy: 0.9797 - val_loss: 0.0615 - val_accuracy: 0.9808
    Epoch 175/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0650 - accuracy: 0.9792 - val_loss: 0.0594 - val_accuracy: 0.9823
    Epoch 176/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0637 - accuracy: 0.9790 - val_loss: 0.0619 - val_accuracy: 0.9800
    Epoch 177/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0642 - accuracy: 0.9797 - val_loss: 0.0595 - val_accuracy: 0.9800
    Epoch 178/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0633 - accuracy: 0.9795 - val_loss: 0.0589 - val_accuracy: 0.9808
    Epoch 179/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0628 - accuracy: 0.9808 - val_loss: 0.0588 - val_accuracy: 0.9808
    Epoch 180/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0623 - accuracy: 0.9795 - val_loss: 0.0591 - val_accuracy: 0.9823
    Epoch 181/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0626 - accuracy: 0.9797 - val_loss: 0.0597 - val_accuracy: 0.9815
    Epoch 182/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0640 - accuracy: 0.9802 - val_loss: 0.0615 - val_accuracy: 0.9792
    Epoch 183/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0640 - accuracy: 0.9810 - val_loss: 0.0625 - val_accuracy: 0.9785
    Epoch 184/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0631 - accuracy: 0.9808 - val_loss: 0.0585 - val_accuracy: 0.9815
    Epoch 185/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0617 - accuracy: 0.9813 - val_loss: 0.0587 - val_accuracy: 0.9815
    Epoch 186/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0616 - accuracy: 0.9813 - val_loss: 0.0583 - val_accuracy: 0.9831
    Epoch 187/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0617 - accuracy: 0.9802 - val_loss: 0.0582 - val_accuracy: 0.9815
    Epoch 188/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0612 - accuracy: 0.9800 - val_loss: 0.0584 - val_accuracy: 0.9815
    Epoch 189/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0630 - accuracy: 0.9802 - val_loss: 0.0582 - val_accuracy: 0.9815
    Epoch 190/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0644 - accuracy: 0.9802 - val_loss: 0.0595 - val_accuracy: 0.9815
    Epoch 191/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0611 - accuracy: 0.9802 - val_loss: 0.0581 - val_accuracy: 0.9808
    Epoch 192/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0610 - accuracy: 0.9800 - val_loss: 0.0638 - val_accuracy: 0.9815
    Epoch 193/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0656 - accuracy: 0.9805 - val_loss: 0.0588 - val_accuracy: 0.9815
    Epoch 194/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0604 - accuracy: 0.9813 - val_loss: 0.0573 - val_accuracy: 0.9815
    Epoch 195/2000
    8/8 [==============================] - 0s 20ms/step - loss: 0.0607 - accuracy: 0.9820 - val_loss: 0.0573 - val_accuracy: 0.9831
    Epoch 196/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0612 - accuracy: 0.9810 - val_loss: 0.0575 - val_accuracy: 0.9831
    Epoch 197/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0613 - accuracy: 0.9800 - val_loss: 0.0569 - val_accuracy: 0.9815
    Epoch 198/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0608 - accuracy: 0.9813 - val_loss: 0.0580 - val_accuracy: 0.9823
    Epoch 199/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0600 - accuracy: 0.9805 - val_loss: 0.0608 - val_accuracy: 0.9808
    Epoch 200/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0625 - accuracy: 0.9810 - val_loss: 0.0643 - val_accuracy: 0.9831
    Epoch 201/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0613 - accuracy: 0.9820 - val_loss: 0.0571 - val_accuracy: 0.9823
    Epoch 202/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0592 - accuracy: 0.9818 - val_loss: 0.0563 - val_accuracy: 0.9815
    Epoch 203/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0591 - accuracy: 0.9818 - val_loss: 0.0571 - val_accuracy: 0.9831
    Epoch 204/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0604 - accuracy: 0.9823 - val_loss: 0.0565 - val_accuracy: 0.9831
    Epoch 205/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0586 - accuracy: 0.9823 - val_loss: 0.0601 - val_accuracy: 0.9815
    Epoch 206/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0633 - accuracy: 0.9815 - val_loss: 0.0728 - val_accuracy: 0.9785
    Epoch 207/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0682 - accuracy: 0.9802 - val_loss: 0.0650 - val_accuracy: 0.9808
    Epoch 208/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0618 - accuracy: 0.9810 - val_loss: 0.0580 - val_accuracy: 0.9808
    Epoch 209/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0595 - accuracy: 0.9815 - val_loss: 0.0568 - val_accuracy: 0.9815
    Epoch 210/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0589 - accuracy: 0.9826 - val_loss: 0.0548 - val_accuracy: 0.9815
    Epoch 211/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0584 - accuracy: 0.9831 - val_loss: 0.0545 - val_accuracy: 0.9808
    Epoch 212/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0600 - accuracy: 0.9828 - val_loss: 0.0570 - val_accuracy: 0.9800
    Epoch 213/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0598 - accuracy: 0.9800 - val_loss: 0.0559 - val_accuracy: 0.9808
    Epoch 214/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0581 - accuracy: 0.9826 - val_loss: 0.0557 - val_accuracy: 0.9823
    Epoch 215/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0574 - accuracy: 0.9823 - val_loss: 0.0563 - val_accuracy: 0.9815
    Epoch 216/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0581 - accuracy: 0.9831 - val_loss: 0.0585 - val_accuracy: 0.9815
    Epoch 217/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0583 - accuracy: 0.9828 - val_loss: 0.0540 - val_accuracy: 0.9831
    Epoch 218/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0574 - accuracy: 0.9831 - val_loss: 0.0553 - val_accuracy: 0.9815
    Epoch 219/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0577 - accuracy: 0.9818 - val_loss: 0.0561 - val_accuracy: 0.9815
    Epoch 220/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0577 - accuracy: 0.9841 - val_loss: 0.0530 - val_accuracy: 0.9823
    Epoch 221/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0582 - accuracy: 0.9820 - val_loss: 0.0590 - val_accuracy: 0.9808
    Epoch 222/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0597 - accuracy: 0.9826 - val_loss: 0.0575 - val_accuracy: 0.9808
    Epoch 223/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0584 - accuracy: 0.9836 - val_loss: 0.0551 - val_accuracy: 0.9800
    Epoch 224/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0581 - accuracy: 0.9818 - val_loss: 0.0537 - val_accuracy: 0.9815
    Epoch 225/2000
    8/8 [==============================] - 0s 22ms/step - loss: 0.0566 - accuracy: 0.9828 - val_loss: 0.0539 - val_accuracy: 0.9823
    Epoch 226/2000
    8/8 [==============================] - 0s 30ms/step - loss: 0.0587 - accuracy: 0.9818 - val_loss: 0.0533 - val_accuracy: 0.9831
    Epoch 227/2000
    8/8 [==============================] - 0s 38ms/step - loss: 0.0577 - accuracy: 0.9828 - val_loss: 0.0542 - val_accuracy: 0.9823
    Epoch 228/2000
    8/8 [==============================] - 0s 29ms/step - loss: 0.0563 - accuracy: 0.9831 - val_loss: 0.0535 - val_accuracy: 0.9823
    Epoch 229/2000
    8/8 [==============================] - 0s 36ms/step - loss: 0.0579 - accuracy: 0.9833 - val_loss: 0.0684 - val_accuracy: 0.9808
    Epoch 230/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0622 - accuracy: 0.9795 - val_loss: 0.0542 - val_accuracy: 0.9815
    Epoch 231/2000
    8/8 [==============================] - 0s 49ms/step - loss: 0.0578 - accuracy: 0.9818 - val_loss: 0.0529 - val_accuracy: 0.9831
    Epoch 232/2000
    8/8 [==============================] - 0s 33ms/step - loss: 0.0570 - accuracy: 0.9826 - val_loss: 0.0526 - val_accuracy: 0.9823
    Epoch 233/2000
    8/8 [==============================] - 0s 28ms/step - loss: 0.0554 - accuracy: 0.9838 - val_loss: 0.0574 - val_accuracy: 0.9815
    Epoch 234/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0576 - accuracy: 0.9826 - val_loss: 0.0538 - val_accuracy: 0.9823
    Epoch 235/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0590 - accuracy: 0.9818 - val_loss: 0.0531 - val_accuracy: 0.9823
    Epoch 236/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0578 - accuracy: 0.9826 - val_loss: 0.0522 - val_accuracy: 0.9846
    Epoch 237/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0572 - accuracy: 0.9826 - val_loss: 0.0525 - val_accuracy: 0.9823
    Epoch 238/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0585 - accuracy: 0.9831 - val_loss: 0.0726 - val_accuracy: 0.9769
    Epoch 239/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0646 - accuracy: 0.9790 - val_loss: 0.0625 - val_accuracy: 0.9800
    Epoch 240/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0583 - accuracy: 0.9831 - val_loss: 0.0533 - val_accuracy: 0.9808
    Epoch 241/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0575 - accuracy: 0.9818 - val_loss: 0.0538 - val_accuracy: 0.9823
    Epoch 242/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0568 - accuracy: 0.9831 - val_loss: 0.0532 - val_accuracy: 0.9823
    Epoch 243/2000
    8/8 [==============================] - 0s 22ms/step - loss: 0.0552 - accuracy: 0.9833 - val_loss: 0.0518 - val_accuracy: 0.9831
    Epoch 244/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0542 - accuracy: 0.9838 - val_loss: 0.0555 - val_accuracy: 0.9831
    Epoch 245/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0550 - accuracy: 0.9841 - val_loss: 0.0515 - val_accuracy: 0.9831
    Epoch 246/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0560 - accuracy: 0.9828 - val_loss: 0.0532 - val_accuracy: 0.9823
    Epoch 247/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0565 - accuracy: 0.9831 - val_loss: 0.0527 - val_accuracy: 0.9831
    Epoch 248/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0552 - accuracy: 0.9833 - val_loss: 0.0526 - val_accuracy: 0.9823
    Epoch 249/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0579 - accuracy: 0.9836 - val_loss: 0.0526 - val_accuracy: 0.9838
    Epoch 250/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0583 - accuracy: 0.9818 - val_loss: 0.0521 - val_accuracy: 0.9831
    Epoch 251/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0611 - accuracy: 0.9810 - val_loss: 0.0546 - val_accuracy: 0.9823
    Epoch 252/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0595 - accuracy: 0.9836 - val_loss: 0.0613 - val_accuracy: 0.9808
    Epoch 253/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0555 - accuracy: 0.9828 - val_loss: 0.0507 - val_accuracy: 0.9800
    Epoch 254/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0542 - accuracy: 0.9838 - val_loss: 0.0523 - val_accuracy: 0.9831
    Epoch 255/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0581 - accuracy: 0.9823 - val_loss: 0.0523 - val_accuracy: 0.9831
    Epoch 256/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0653 - accuracy: 0.9802 - val_loss: 0.0508 - val_accuracy: 0.9815
    Epoch 257/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0611 - accuracy: 0.9813 - val_loss: 0.0523 - val_accuracy: 0.9854
    Epoch 258/2000
    8/8 [==============================] - 0s 6ms/step - loss: 0.0539 - accuracy: 0.9836 - val_loss: 0.0513 - val_accuracy: 0.9831
    Epoch 259/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0537 - accuracy: 0.9838 - val_loss: 0.0510 - val_accuracy: 0.9846
    Epoch 260/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0535 - accuracy: 0.9851 - val_loss: 0.0544 - val_accuracy: 0.9846
    Epoch 261/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0540 - accuracy: 0.9836 - val_loss: 0.0511 - val_accuracy: 0.9823
    Epoch 262/2000
    8/8 [==============================] - 0s 28ms/step - loss: 0.0528 - accuracy: 0.9851 - val_loss: 0.0504 - val_accuracy: 0.9831
    Epoch 263/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0546 - accuracy: 0.9854 - val_loss: 0.0562 - val_accuracy: 0.9823
    Epoch 264/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0549 - accuracy: 0.9838 - val_loss: 0.0507 - val_accuracy: 0.9823
    Epoch 265/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0524 - accuracy: 0.9841 - val_loss: 0.0500 - val_accuracy: 0.9838
    Epoch 266/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0522 - accuracy: 0.9843 - val_loss: 0.0516 - val_accuracy: 0.9823
    Epoch 267/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0533 - accuracy: 0.9854 - val_loss: 0.0512 - val_accuracy: 0.9831
    Epoch 268/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0547 - accuracy: 0.9838 - val_loss: 0.0526 - val_accuracy: 0.9823
    Epoch 269/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0537 - accuracy: 0.9856 - val_loss: 0.0538 - val_accuracy: 0.9823
    Epoch 270/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0545 - accuracy: 0.9833 - val_loss: 0.0517 - val_accuracy: 0.9800
    Epoch 271/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0523 - accuracy: 0.9836 - val_loss: 0.0501 - val_accuracy: 0.9838
    Epoch 272/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0525 - accuracy: 0.9856 - val_loss: 0.0556 - val_accuracy: 0.9831
    Epoch 273/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0561 - accuracy: 0.9813 - val_loss: 0.0553 - val_accuracy: 0.9831
    Epoch 274/2000
    8/8 [==============================] - 0s 12ms/step - loss: 0.0571 - accuracy: 0.9833 - val_loss: 0.0561 - val_accuracy: 0.9831
    Epoch 275/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0577 - accuracy: 0.9833 - val_loss: 0.0519 - val_accuracy: 0.9838
    Epoch 276/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0558 - accuracy: 0.9851 - val_loss: 0.0504 - val_accuracy: 0.9823
    Epoch 277/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0604 - accuracy: 0.9813 - val_loss: 0.0544 - val_accuracy: 0.9823
    Epoch 278/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0589 - accuracy: 0.9828 - val_loss: 0.0495 - val_accuracy: 0.9808
    Epoch 279/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0547 - accuracy: 0.9836 - val_loss: 0.0505 - val_accuracy: 0.9831
    Epoch 280/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0539 - accuracy: 0.9836 - val_loss: 0.0502 - val_accuracy: 0.9838
    Epoch 281/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0543 - accuracy: 0.9849 - val_loss: 0.0499 - val_accuracy: 0.9823
    Epoch 282/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0521 - accuracy: 0.9838 - val_loss: 0.0507 - val_accuracy: 0.9846
    Epoch 283/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0508 - accuracy: 0.9851 - val_loss: 0.0519 - val_accuracy: 0.9854
    Epoch 284/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0525 - accuracy: 0.9841 - val_loss: 0.0512 - val_accuracy: 0.9838
    Epoch 285/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0524 - accuracy: 0.9846 - val_loss: 0.0500 - val_accuracy: 0.9838
    Epoch 286/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0512 - accuracy: 0.9854 - val_loss: 0.0493 - val_accuracy: 0.9831
    Epoch 287/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0514 - accuracy: 0.9851 - val_loss: 0.0495 - val_accuracy: 0.9823
    Epoch 288/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0546 - accuracy: 0.9849 - val_loss: 0.0545 - val_accuracy: 0.9831
    Epoch 289/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0569 - accuracy: 0.9833 - val_loss: 0.0717 - val_accuracy: 0.9762
    Epoch 290/2000
    8/8 [==============================] - 0s 14ms/step - loss: 0.0599 - accuracy: 0.9813 - val_loss: 0.0533 - val_accuracy: 0.9823
    Epoch 291/2000
    8/8 [==============================] - 0s 18ms/step - loss: 0.0531 - accuracy: 0.9849 - val_loss: 0.0479 - val_accuracy: 0.9823
    Epoch 292/2000
    8/8 [==============================] - 0s 23ms/step - loss: 0.0505 - accuracy: 0.9851 - val_loss: 0.0508 - val_accuracy: 0.9838
    Epoch 293/2000
    8/8 [==============================] - 0s 9ms/step - loss: 0.0498 - accuracy: 0.9864 - val_loss: 0.0502 - val_accuracy: 0.9823
    Epoch 294/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0497 - accuracy: 0.9859 - val_loss: 0.0500 - val_accuracy: 0.9831
    Epoch 295/2000
    8/8 [==============================] - 0s 8ms/step - loss: 0.0497 - accuracy: 0.9861 - val_loss: 0.0492 - val_accuracy: 0.9831
    Epoch 296/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0500 - accuracy: 0.9849 - val_loss: 0.0489 - val_accuracy: 0.9846
    Epoch 297/2000
    8/8 [==============================] - 0s 26ms/step - loss: 0.0503 - accuracy: 0.9846 - val_loss: 0.0490 - val_accuracy: 0.9846
    Epoch 298/2000
    8/8 [==============================] - 0s 36ms/step - loss: 0.0501 - accuracy: 0.9867 - val_loss: 0.0499 - val_accuracy: 0.9815
    Epoch 299/2000
    8/8 [==============================] - 0s 33ms/step - loss: 0.0504 - accuracy: 0.9846 - val_loss: 0.0497 - val_accuracy: 0.9838
    Epoch 300/2000
    8/8 [==============================] - 0s 21ms/step - loss: 0.0507 - accuracy: 0.9843 - val_loss: 0.0502 - val_accuracy: 0.9854
    Epoch 301/2000
    8/8 [==============================] - 0s 22ms/step - loss: 0.0497 - accuracy: 0.9859 - val_loss: 0.0487 - val_accuracy: 0.9831
    Epoch 302/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0495 - accuracy: 0.9854 - val_loss: 0.0488 - val_accuracy: 0.9831
    Epoch 303/2000
    8/8 [==============================] - 0s 35ms/step - loss: 0.0497 - accuracy: 0.9861 - val_loss: 0.0476 - val_accuracy: 0.9838
    Epoch 304/2000
    8/8 [==============================] - 0s 17ms/step - loss: 0.0502 - accuracy: 0.9856 - val_loss: 0.0500 - val_accuracy: 0.9854
    Epoch 305/2000
    8/8 [==============================] - 0s 31ms/step - loss: 0.0499 - accuracy: 0.9838 - val_loss: 0.0554 - val_accuracy: 0.9838
    Epoch 306/2000
    8/8 [==============================] - 0s 33ms/step - loss: 0.0540 - accuracy: 0.9841 - val_loss: 0.0590 - val_accuracy: 0.9838
    Epoch 307/2000
    8/8 [==============================] - 0s 50ms/step - loss: 0.0524 - accuracy: 0.9836 - val_loss: 0.0498 - val_accuracy: 0.9838
    Epoch 308/2000
    8/8 [==============================] - 0s 48ms/step - loss: 0.0493 - accuracy: 0.9867 - val_loss: 0.0489 - val_accuracy: 0.9831
    Epoch 309/2000
    8/8 [==============================] - 0s 57ms/step - loss: 0.0492 - accuracy: 0.9851 - val_loss: 0.0506 - val_accuracy: 0.9854
    Epoch 310/2000
    8/8 [==============================] - 0s 45ms/step - loss: 0.0490 - accuracy: 0.9861 - val_loss: 0.0493 - val_accuracy: 0.9838
    Epoch 311/2000
    8/8 [==============================] - 0s 34ms/step - loss: 0.0494 - accuracy: 0.9861 - val_loss: 0.0489 - val_accuracy: 0.9838
    Epoch 312/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0486 - accuracy: 0.9851 - val_loss: 0.0487 - val_accuracy: 0.9846
    Epoch 313/2000
    8/8 [==============================] - 0s 13ms/step - loss: 0.0493 - accuracy: 0.9856 - val_loss: 0.0523 - val_accuracy: 0.9862
    Epoch 314/2000
    8/8 [==============================] - 0s 16ms/step - loss: 0.0533 - accuracy: 0.9843 - val_loss: 0.0551 - val_accuracy: 0.9862
    Epoch 315/2000
    8/8 [==============================] - 0s 11ms/step - loss: 0.0513 - accuracy: 0.9838 - val_loss: 0.0501 - val_accuracy: 0.9846
    Epoch 316/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0486 - accuracy: 0.9861 - val_loss: 0.0481 - val_accuracy: 0.9838
    Epoch 317/2000
    8/8 [==============================] - 0s 10ms/step - loss: 0.0503 - accuracy: 0.9864 - val_loss: 0.0477 - val_accuracy: 0.9823
    Epoch 318/2000
    8/8 [==============================] - 0s 15ms/step - loss: 0.0479 - accuracy: 0.9851 - val_loss: 0.0531 - val_accuracy: 0.9862
    Epoch 319/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0503 - accuracy: 0.9836 - val_loss: 0.0549 - val_accuracy: 0.9846
    Epoch 320/2000
    8/8 [==============================] - 0s 7ms/step - loss: 0.0510 - accuracy: 0.9843 - val_loss: 0.0582 - val_accuracy: 0.9838
    Epoch 321/2000
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
    Epoch 3/2000
    1/1 [==============================] - 0s 31ms/step - loss: 20.1548 - accuracy: 0.6478 - val_loss: 18.7394 - val_accuracy: 0.6039
    Epoch 4/2000
    1/1 [==============================] - 0s 32ms/step - loss: 19.0236 - accuracy: 0.6478 - val_loss: 17.6046 - val_accuracy: 0.6039
    Epoch 5/2000
    1/1 [==============================] - 0s 32ms/step - loss: 17.9013 - accuracy: 0.6478 - val_loss: 16.4782 - val_accuracy: 0.6039
    Epoch 6/2000
    1/1 [==============================] - 0s 37ms/step - loss: 16.7871 - accuracy: 0.6478 - val_loss: 15.3633 - val_accuracy: 0.6039
    Epoch 7/2000
    1/1 [==============================] - 0s 32ms/step - loss: 15.6775 - accuracy: 0.6478 - val_loss: 14.2777 - val_accuracy: 0.6039
    Epoch 8/2000
    1/1 [==============================] - 0s 32ms/step - loss: 14.5664 - accuracy: 0.6478 - val_loss: 13.1947 - val_accuracy: 0.6039
    Epoch 9/2000
    1/1 [==============================] - 0s 34ms/step - loss: 13.4610 - accuracy: 0.6478 - val_loss: 12.1120 - val_accuracy: 0.6039
    Epoch 10/2000
    1/1 [==============================] - 0s 31ms/step - loss: 12.3700 - accuracy: 0.6478 - val_loss: 11.0750 - val_accuracy: 0.6039
    Epoch 11/2000
    1/1 [==============================] - 0s 32ms/step - loss: 11.3032 - accuracy: 0.6457 - val_loss: 10.0879 - val_accuracy: 0.6039
    Epoch 12/2000
    1/1 [==============================] - 0s 36ms/step - loss: 10.2757 - accuracy: 0.6457 - val_loss: 9.1878 - val_accuracy: 0.6039
    Epoch 13/2000
    1/1 [==============================] - 0s 40ms/step - loss: 9.3044 - accuracy: 0.6435 - val_loss: 8.3915 - val_accuracy: 0.6039
    Epoch 14/2000
    1/1 [==============================] - 0s 32ms/step - loss: 8.4186 - accuracy: 0.6413 - val_loss: 7.6508 - val_accuracy: 0.5974
    Epoch 15/2000
    1/1 [==============================] - 0s 32ms/step - loss: 7.6036 - accuracy: 0.6435 - val_loss: 6.9521 - val_accuracy: 0.5974
    Epoch 16/2000
    1/1 [==============================] - 0s 33ms/step - loss: 6.8431 - accuracy: 0.6413 - val_loss: 6.2778 - val_accuracy: 0.5909
    Epoch 17/2000
    1/1 [==============================] - 0s 33ms/step - loss: 6.1511 - accuracy: 0.6304 - val_loss: 5.6376 - val_accuracy: 0.5779
    Epoch 18/2000
    1/1 [==============================] - 0s 35ms/step - loss: 5.5010 - accuracy: 0.6196 - val_loss: 5.0299 - val_accuracy: 0.5584
    Epoch 19/2000
    1/1 [==============================] - 0s 30ms/step - loss: 4.8878 - accuracy: 0.6022 - val_loss: 4.4415 - val_accuracy: 0.5584
    Epoch 20/2000
    1/1 [==============================] - 0s 35ms/step - loss: 4.3066 - accuracy: 0.5804 - val_loss: 3.8915 - val_accuracy: 0.5065
    Epoch 21/2000
    1/1 [==============================] - 0s 53ms/step - loss: 3.7617 - accuracy: 0.5435 - val_loss: 3.4134 - val_accuracy: 0.4870
    Epoch 22/2000
    1/1 [==============================] - 0s 33ms/step - loss: 3.2765 - accuracy: 0.5174 - val_loss: 3.0266 - val_accuracy: 0.4481
    Epoch 23/2000
    1/1 [==============================] - 0s 44ms/step - loss: 2.8815 - accuracy: 0.4978 - val_loss: 2.7450 - val_accuracy: 0.4416
    Epoch 24/2000
    1/1 [==============================] - 0s 32ms/step - loss: 2.6068 - accuracy: 0.4543 - val_loss: 2.5720 - val_accuracy: 0.4416
    Epoch 25/2000
    1/1 [==============================] - 0s 35ms/step - loss: 2.4537 - accuracy: 0.4174 - val_loss: 2.5016 - val_accuracy: 0.4026
    Epoch 26/2000
    1/1 [==============================] - 0s 33ms/step - loss: 2.3937 - accuracy: 0.4065 - val_loss: 2.4964 - val_accuracy: 0.4026
    Epoch 27/2000
    1/1 [==============================] - 0s 34ms/step - loss: 2.3929 - accuracy: 0.3652 - val_loss: 2.5112 - val_accuracy: 0.4091
    Epoch 28/2000
    1/1 [==============================] - 0s 34ms/step - loss: 2.4141 - accuracy: 0.3717 - val_loss: 2.5231 - val_accuracy: 0.3896
    Epoch 29/2000
    1/1 [==============================] - 0s 38ms/step - loss: 2.4406 - accuracy: 0.3761 - val_loss: 2.5193 - val_accuracy: 0.3961
    Epoch 30/2000
    1/1 [==============================] - 0s 31ms/step - loss: 2.4558 - accuracy: 0.3804 - val_loss: 2.4879 - val_accuracy: 0.4091
    Epoch 31/2000
    1/1 [==============================] - 0s 35ms/step - loss: 2.4487 - accuracy: 0.3804 - val_loss: 2.4365 - val_accuracy: 0.4026
    Epoch 32/2000
    1/1 [==============================] - 0s 38ms/step - loss: 2.4154 - accuracy: 0.3891 - val_loss: 2.3633 - val_accuracy: 0.4091
    Epoch 33/2000
    1/1 [==============================] - 0s 38ms/step - loss: 2.3567 - accuracy: 0.3826 - val_loss: 2.2687 - val_accuracy: 0.4156
    Epoch 34/2000
    1/1 [==============================] - 0s 33ms/step - loss: 2.2736 - accuracy: 0.3848 - val_loss: 2.1601 - val_accuracy: 0.4221
    Epoch 35/2000
    1/1 [==============================] - 0s 38ms/step - loss: 2.1722 - accuracy: 0.3957 - val_loss: 2.0424 - val_accuracy: 0.4286
    Epoch 36/2000
    1/1 [==============================] - 0s 39ms/step - loss: 2.0576 - accuracy: 0.3913 - val_loss: 1.9206 - val_accuracy: 0.4545
    Epoch 37/2000
    1/1 [==============================] - 0s 33ms/step - loss: 1.9346 - accuracy: 0.4087 - val_loss: 1.7990 - val_accuracy: 0.4610
    Epoch 38/2000
    1/1 [==============================] - 0s 30ms/step - loss: 1.8074 - accuracy: 0.4130 - val_loss: 1.6840 - val_accuracy: 0.4805
    Epoch 39/2000
    1/1 [==============================] - 0s 39ms/step - loss: 1.6834 - accuracy: 0.4348 - val_loss: 1.5813 - val_accuracy: 0.5130
    Epoch 40/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.5696 - accuracy: 0.4500 - val_loss: 1.4911 - val_accuracy: 0.5195
    Epoch 41/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.4731 - accuracy: 0.4739 - val_loss: 1.4236 - val_accuracy: 0.5325
    Epoch 42/2000
    1/1 [==============================] - 0s 39ms/step - loss: 1.4036 - accuracy: 0.4913 - val_loss: 1.3914 - val_accuracy: 0.5519
    Epoch 43/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.3612 - accuracy: 0.5065 - val_loss: 1.3849 - val_accuracy: 0.5584
    Epoch 44/2000
    1/1 [==============================] - 0s 39ms/step - loss: 1.3450 - accuracy: 0.5174 - val_loss: 1.3961 - val_accuracy: 0.5649
    Epoch 45/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.3463 - accuracy: 0.5478 - val_loss: 1.4117 - val_accuracy: 0.5455
    Epoch 46/2000
    1/1 [==============================] - 0s 36ms/step - loss: 1.3503 - accuracy: 0.5696 - val_loss: 1.4201 - val_accuracy: 0.5390
    Epoch 47/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.3459 - accuracy: 0.5870 - val_loss: 1.4154 - val_accuracy: 0.5519
    Epoch 48/2000
    1/1 [==============================] - 0s 42ms/step - loss: 1.3309 - accuracy: 0.5935 - val_loss: 1.3980 - val_accuracy: 0.5779
    Epoch 49/2000
    1/1 [==============================] - 0s 49ms/step - loss: 1.3071 - accuracy: 0.5957 - val_loss: 1.3715 - val_accuracy: 0.5584
    Epoch 50/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.2760 - accuracy: 0.5978 - val_loss: 1.3458 - val_accuracy: 0.5714
    Epoch 51/2000
    1/1 [==============================] - 0s 33ms/step - loss: 1.2412 - accuracy: 0.6043 - val_loss: 1.3258 - val_accuracy: 0.5779
    Epoch 52/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.2062 - accuracy: 0.6109 - val_loss: 1.3091 - val_accuracy: 0.5714
    Epoch 53/2000
    1/1 [==============================] - 0s 36ms/step - loss: 1.1771 - accuracy: 0.6065 - val_loss: 1.2986 - val_accuracy: 0.5584
    Epoch 54/2000
    1/1 [==============================] - 0s 41ms/step - loss: 1.1543 - accuracy: 0.6087 - val_loss: 1.2897 - val_accuracy: 0.5519
    Epoch 55/2000
    1/1 [==============================] - 0s 32ms/step - loss: 1.1371 - accuracy: 0.6087 - val_loss: 1.2794 - val_accuracy: 0.5649
    Epoch 56/2000
    1/1 [==============================] - 0s 33ms/step - loss: 1.1259 - accuracy: 0.6087 - val_loss: 1.2694 - val_accuracy: 0.5649
    Epoch 57/2000
    1/1 [==============================] - 0s 37ms/step - loss: 1.1158 - accuracy: 0.6065 - val_loss: 1.2581 - val_accuracy: 0.5649
    Epoch 58/2000
    1/1 [==============================] - 0s 36ms/step - loss: 1.1057 - accuracy: 0.6130 - val_loss: 1.2451 - val_accuracy: 0.5649
    Epoch 59/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.0950 - accuracy: 0.6130 - val_loss: 1.2302 - val_accuracy: 0.5714
    Epoch 60/2000
    1/1 [==============================] - 0s 35ms/step - loss: 1.0830 - accuracy: 0.6174 - val_loss: 1.2130 - val_accuracy: 0.5779
    Epoch 61/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.0694 - accuracy: 0.6130 - val_loss: 1.1938 - val_accuracy: 0.5844
    Epoch 62/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.0540 - accuracy: 0.6152 - val_loss: 1.1730 - val_accuracy: 0.5844
    Epoch 63/2000
    1/1 [==============================] - 0s 34ms/step - loss: 1.0366 - accuracy: 0.6174 - val_loss: 1.1515 - val_accuracy: 0.5909
    Epoch 64/2000
    1/1 [==============================] - 0s 31ms/step - loss: 1.0173 - accuracy: 0.6217 - val_loss: 1.1271 - val_accuracy: 0.6039
    Epoch 65/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.9962 - accuracy: 0.6239 - val_loss: 1.1009 - val_accuracy: 0.6039
    Epoch 66/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.9746 - accuracy: 0.6217 - val_loss: 1.0763 - val_accuracy: 0.6039
    Epoch 67/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.9547 - accuracy: 0.6217 - val_loss: 1.0556 - val_accuracy: 0.6039
    Epoch 68/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.9379 - accuracy: 0.6174 - val_loss: 1.0369 - val_accuracy: 0.6169
    Epoch 69/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.9248 - accuracy: 0.6217 - val_loss: 1.0196 - val_accuracy: 0.6169
    Epoch 70/2000
    1/1 [==============================] - 0s 30ms/step - loss: 0.9149 - accuracy: 0.6239 - val_loss: 1.0031 - val_accuracy: 0.6234
    Epoch 71/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.9062 - accuracy: 0.6261 - val_loss: 0.9865 - val_accuracy: 0.6364
    Epoch 72/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.8971 - accuracy: 0.6196 - val_loss: 0.9708 - val_accuracy: 0.6299
    Epoch 73/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.8864 - accuracy: 0.6174 - val_loss: 0.9561 - val_accuracy: 0.6234
    Epoch 74/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.8733 - accuracy: 0.6196 - val_loss: 0.9433 - val_accuracy: 0.6169
    Epoch 75/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.8588 - accuracy: 0.6217 - val_loss: 0.9335 - val_accuracy: 0.6104
    Epoch 76/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.8448 - accuracy: 0.6239 - val_loss: 0.9269 - val_accuracy: 0.5974
    Epoch 77/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.8330 - accuracy: 0.6196 - val_loss: 0.9228 - val_accuracy: 0.5909
    Epoch 78/2000
    1/1 [==============================] - 0s 30ms/step - loss: 0.8240 - accuracy: 0.6217 - val_loss: 0.9198 - val_accuracy: 0.5909
    Epoch 79/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.8171 - accuracy: 0.6283 - val_loss: 0.9159 - val_accuracy: 0.5974
    Epoch 80/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.8109 - accuracy: 0.6326 - val_loss: 0.9100 - val_accuracy: 0.5909
    Epoch 81/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.8041 - accuracy: 0.6348 - val_loss: 0.9016 - val_accuracy: 0.5909
    Epoch 82/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.7963 - accuracy: 0.6370 - val_loss: 0.8910 - val_accuracy: 0.5909
    Epoch 83/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.7878 - accuracy: 0.6391 - val_loss: 0.8791 - val_accuracy: 0.5844
    Epoch 84/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.7793 - accuracy: 0.6413 - val_loss: 0.8671 - val_accuracy: 0.5844
    Epoch 85/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.7722 - accuracy: 0.6457 - val_loss: 0.8570 - val_accuracy: 0.5844
    Epoch 86/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.7666 - accuracy: 0.6478 - val_loss: 0.8490 - val_accuracy: 0.5909
    Epoch 87/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.7619 - accuracy: 0.6457 - val_loss: 0.8426 - val_accuracy: 0.5909
    Epoch 88/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.7571 - accuracy: 0.6457 - val_loss: 0.8376 - val_accuracy: 0.5974
    Epoch 89/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.7515 - accuracy: 0.6543 - val_loss: 0.8339 - val_accuracy: 0.5974
    Epoch 90/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.7451 - accuracy: 0.6587 - val_loss: 0.8313 - val_accuracy: 0.6039
    Epoch 91/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.7385 - accuracy: 0.6587 - val_loss: 0.8300 - val_accuracy: 0.6104
    Epoch 92/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.7323 - accuracy: 0.6696 - val_loss: 0.8293 - val_accuracy: 0.6039
    Epoch 93/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.7269 - accuracy: 0.6652 - val_loss: 0.8280 - val_accuracy: 0.6104
    Epoch 94/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.7220 - accuracy: 0.6652 - val_loss: 0.8252 - val_accuracy: 0.6169
    Epoch 95/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.7174 - accuracy: 0.6674 - val_loss: 0.8205 - val_accuracy: 0.6169
    Epoch 96/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.7125 - accuracy: 0.6717 - val_loss: 0.8140 - val_accuracy: 0.6104
    Epoch 97/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.7074 - accuracy: 0.6761 - val_loss: 0.8064 - val_accuracy: 0.6104
    Epoch 98/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.7025 - accuracy: 0.6783 - val_loss: 0.7986 - val_accuracy: 0.6169
    Epoch 99/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.6980 - accuracy: 0.6783 - val_loss: 0.7911 - val_accuracy: 0.6234
    Epoch 100/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.6941 - accuracy: 0.6848 - val_loss: 0.7847 - val_accuracy: 0.6234
    Epoch 101/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.6907 - accuracy: 0.6913 - val_loss: 0.7793 - val_accuracy: 0.6299
    Epoch 102/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6874 - accuracy: 0.6913 - val_loss: 0.7749 - val_accuracy: 0.6364
    Epoch 103/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6840 - accuracy: 0.6935 - val_loss: 0.7713 - val_accuracy: 0.6429
    Epoch 104/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.6805 - accuracy: 0.6935 - val_loss: 0.7683 - val_accuracy: 0.6429
    Epoch 105/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.6770 - accuracy: 0.6935 - val_loss: 0.7658 - val_accuracy: 0.6364
    Epoch 106/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6737 - accuracy: 0.6913 - val_loss: 0.7632 - val_accuracy: 0.6364
    Epoch 107/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.6707 - accuracy: 0.7000 - val_loss: 0.7604 - val_accuracy: 0.6299
    Epoch 108/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6678 - accuracy: 0.7022 - val_loss: 0.7569 - val_accuracy: 0.6299
    Epoch 109/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6649 - accuracy: 0.7043 - val_loss: 0.7527 - val_accuracy: 0.6299
    Epoch 110/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6619 - accuracy: 0.7043 - val_loss: 0.7482 - val_accuracy: 0.6494
    Epoch 111/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.6589 - accuracy: 0.7022 - val_loss: 0.7435 - val_accuracy: 0.6494
    Epoch 112/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.6559 - accuracy: 0.7087 - val_loss: 0.7387 - val_accuracy: 0.6494
    Epoch 113/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6530 - accuracy: 0.7043 - val_loss: 0.7342 - val_accuracy: 0.6494
    Epoch 114/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.6504 - accuracy: 0.7065 - val_loss: 0.7305 - val_accuracy: 0.6494
    Epoch 115/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6481 - accuracy: 0.7109 - val_loss: 0.7272 - val_accuracy: 0.6558
    Epoch 116/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.6457 - accuracy: 0.7152 - val_loss: 0.7245 - val_accuracy: 0.6623
    Epoch 117/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.6433 - accuracy: 0.7152 - val_loss: 0.7223 - val_accuracy: 0.6623
    Epoch 118/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.6410 - accuracy: 0.7174 - val_loss: 0.7203 - val_accuracy: 0.6623
    Epoch 119/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.6389 - accuracy: 0.7174 - val_loss: 0.7184 - val_accuracy: 0.6623
    Epoch 120/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.6369 - accuracy: 0.7196 - val_loss: 0.7162 - val_accuracy: 0.6688
    Epoch 121/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6351 - accuracy: 0.7196 - val_loss: 0.7138 - val_accuracy: 0.6623
    Epoch 122/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.6331 - accuracy: 0.7217 - val_loss: 0.7111 - val_accuracy: 0.6623
    Epoch 123/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.6312 - accuracy: 0.7174 - val_loss: 0.7083 - val_accuracy: 0.6623
    Epoch 124/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.6295 - accuracy: 0.7174 - val_loss: 0.7057 - val_accuracy: 0.6558
    Epoch 125/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6281 - accuracy: 0.7152 - val_loss: 0.7033 - val_accuracy: 0.6558
    Epoch 126/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6268 - accuracy: 0.7174 - val_loss: 0.7013 - val_accuracy: 0.6558
    Epoch 127/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.6255 - accuracy: 0.7152 - val_loss: 0.6998 - val_accuracy: 0.6623
    Epoch 128/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.6242 - accuracy: 0.7174 - val_loss: 0.6987 - val_accuracy: 0.6558
    Epoch 129/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6228 - accuracy: 0.7174 - val_loss: 0.6977 - val_accuracy: 0.6558
    Epoch 130/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6215 - accuracy: 0.7174 - val_loss: 0.6968 - val_accuracy: 0.6558
    Epoch 131/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.6203 - accuracy: 0.7196 - val_loss: 0.6956 - val_accuracy: 0.6494
    Epoch 132/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.6190 - accuracy: 0.7217 - val_loss: 0.6943 - val_accuracy: 0.6558
    Epoch 133/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6176 - accuracy: 0.7239 - val_loss: 0.6928 - val_accuracy: 0.6558
    Epoch 134/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.6163 - accuracy: 0.7239 - val_loss: 0.6913 - val_accuracy: 0.6558
    Epoch 135/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.6149 - accuracy: 0.7239 - val_loss: 0.6899 - val_accuracy: 0.6623
    Epoch 136/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.6137 - accuracy: 0.7217 - val_loss: 0.6885 - val_accuracy: 0.6623
    Epoch 137/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6125 - accuracy: 0.7261 - val_loss: 0.6871 - val_accuracy: 0.6623
    Epoch 138/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.6112 - accuracy: 0.7239 - val_loss: 0.6856 - val_accuracy: 0.6623
    Epoch 139/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.6099 - accuracy: 0.7239 - val_loss: 0.6841 - val_accuracy: 0.6623
    Epoch 140/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6086 - accuracy: 0.7217 - val_loss: 0.6826 - val_accuracy: 0.6623
    Epoch 141/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.6074 - accuracy: 0.7217 - val_loss: 0.6811 - val_accuracy: 0.6623
    Epoch 142/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.6063 - accuracy: 0.7217 - val_loss: 0.6795 - val_accuracy: 0.6688
    Epoch 143/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.6052 - accuracy: 0.7217 - val_loss: 0.6780 - val_accuracy: 0.6688
    Epoch 144/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.6041 - accuracy: 0.7217 - val_loss: 0.6766 - val_accuracy: 0.6753
    Epoch 145/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.6030 - accuracy: 0.7217 - val_loss: 0.6753 - val_accuracy: 0.6753
    Epoch 146/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.6020 - accuracy: 0.7196 - val_loss: 0.6742 - val_accuracy: 0.6818
    Epoch 147/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.6009 - accuracy: 0.7217 - val_loss: 0.6733 - val_accuracy: 0.6818
    Epoch 148/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.5999 - accuracy: 0.7196 - val_loss: 0.6724 - val_accuracy: 0.6818
    Epoch 149/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.5989 - accuracy: 0.7196 - val_loss: 0.6718 - val_accuracy: 0.6818
    Epoch 150/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5979 - accuracy: 0.7196 - val_loss: 0.6712 - val_accuracy: 0.6818
    Epoch 151/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5970 - accuracy: 0.7174 - val_loss: 0.6707 - val_accuracy: 0.6818
    Epoch 152/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5961 - accuracy: 0.7174 - val_loss: 0.6702 - val_accuracy: 0.6818
    Epoch 153/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5952 - accuracy: 0.7174 - val_loss: 0.6697 - val_accuracy: 0.6818
    Epoch 154/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5944 - accuracy: 0.7174 - val_loss: 0.6692 - val_accuracy: 0.6818
    Epoch 155/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5936 - accuracy: 0.7152 - val_loss: 0.6686 - val_accuracy: 0.6883
    Epoch 156/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5929 - accuracy: 0.7174 - val_loss: 0.6680 - val_accuracy: 0.6883
    Epoch 157/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5921 - accuracy: 0.7196 - val_loss: 0.6672 - val_accuracy: 0.6883
    Epoch 158/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5913 - accuracy: 0.7196 - val_loss: 0.6665 - val_accuracy: 0.6883
    Epoch 159/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5905 - accuracy: 0.7196 - val_loss: 0.6658 - val_accuracy: 0.6883
    Epoch 160/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5898 - accuracy: 0.7196 - val_loss: 0.6650 - val_accuracy: 0.6883
    Epoch 161/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5890 - accuracy: 0.7174 - val_loss: 0.6644 - val_accuracy: 0.6883
    Epoch 162/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5883 - accuracy: 0.7174 - val_loss: 0.6637 - val_accuracy: 0.6883
    Epoch 163/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5876 - accuracy: 0.7174 - val_loss: 0.6631 - val_accuracy: 0.6883
    Epoch 164/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.5868 - accuracy: 0.7174 - val_loss: 0.6624 - val_accuracy: 0.6883
    Epoch 165/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5860 - accuracy: 0.7174 - val_loss: 0.6618 - val_accuracy: 0.6883
    Epoch 166/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5853 - accuracy: 0.7174 - val_loss: 0.6611 - val_accuracy: 0.6883
    Epoch 167/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5846 - accuracy: 0.7174 - val_loss: 0.6604 - val_accuracy: 0.6883
    Epoch 168/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5838 - accuracy: 0.7196 - val_loss: 0.6598 - val_accuracy: 0.6818
    Epoch 169/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5831 - accuracy: 0.7174 - val_loss: 0.6593 - val_accuracy: 0.6818
    Epoch 170/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5825 - accuracy: 0.7152 - val_loss: 0.6589 - val_accuracy: 0.6818
    Epoch 171/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5818 - accuracy: 0.7174 - val_loss: 0.6585 - val_accuracy: 0.6818
    Epoch 172/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5811 - accuracy: 0.7196 - val_loss: 0.6582 - val_accuracy: 0.6753
    Epoch 173/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5805 - accuracy: 0.7217 - val_loss: 0.6580 - val_accuracy: 0.6753
    Epoch 174/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5799 - accuracy: 0.7217 - val_loss: 0.6579 - val_accuracy: 0.6753
    Epoch 175/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5793 - accuracy: 0.7261 - val_loss: 0.6578 - val_accuracy: 0.6818
    Epoch 176/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5787 - accuracy: 0.7261 - val_loss: 0.6577 - val_accuracy: 0.6818
    Epoch 177/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5782 - accuracy: 0.7283 - val_loss: 0.6576 - val_accuracy: 0.6883
    Epoch 178/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.5777 - accuracy: 0.7283 - val_loss: 0.6574 - val_accuracy: 0.6883
    Epoch 179/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5772 - accuracy: 0.7261 - val_loss: 0.6572 - val_accuracy: 0.6883
    Epoch 180/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5767 - accuracy: 0.7261 - val_loss: 0.6569 - val_accuracy: 0.6883
    Epoch 181/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5762 - accuracy: 0.7261 - val_loss: 0.6566 - val_accuracy: 0.6883
    Epoch 182/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.5758 - accuracy: 0.7261 - val_loss: 0.6562 - val_accuracy: 0.6883
    Epoch 183/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5753 - accuracy: 0.7261 - val_loss: 0.6559 - val_accuracy: 0.6883
    Epoch 184/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5749 - accuracy: 0.7283 - val_loss: 0.6555 - val_accuracy: 0.6883
    Epoch 185/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5744 - accuracy: 0.7283 - val_loss: 0.6551 - val_accuracy: 0.6883
    Epoch 186/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5740 - accuracy: 0.7304 - val_loss: 0.6546 - val_accuracy: 0.6883
    Epoch 187/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5736 - accuracy: 0.7304 - val_loss: 0.6541 - val_accuracy: 0.6883
    Epoch 188/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5732 - accuracy: 0.7304 - val_loss: 0.6537 - val_accuracy: 0.6818
    Epoch 189/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5728 - accuracy: 0.7326 - val_loss: 0.6532 - val_accuracy: 0.6818
    Epoch 190/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5724 - accuracy: 0.7326 - val_loss: 0.6527 - val_accuracy: 0.6818
    Epoch 191/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5720 - accuracy: 0.7348 - val_loss: 0.6522 - val_accuracy: 0.6818
    Epoch 192/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5717 - accuracy: 0.7370 - val_loss: 0.6518 - val_accuracy: 0.6818
    Epoch 193/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5714 - accuracy: 0.7391 - val_loss: 0.6514 - val_accuracy: 0.6818
    Epoch 194/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.5710 - accuracy: 0.7391 - val_loss: 0.6511 - val_accuracy: 0.6883
    Epoch 195/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5707 - accuracy: 0.7391 - val_loss: 0.6509 - val_accuracy: 0.6883
    Epoch 196/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.5703 - accuracy: 0.7391 - val_loss: 0.6506 - val_accuracy: 0.6883
    Epoch 197/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5700 - accuracy: 0.7391 - val_loss: 0.6504 - val_accuracy: 0.6883
    Epoch 198/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.5697 - accuracy: 0.7391 - val_loss: 0.6502 - val_accuracy: 0.6883
    Epoch 199/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5693 - accuracy: 0.7413 - val_loss: 0.6502 - val_accuracy: 0.6883
    Epoch 200/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5690 - accuracy: 0.7413 - val_loss: 0.6501 - val_accuracy: 0.6883
    Epoch 201/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5687 - accuracy: 0.7413 - val_loss: 0.6501 - val_accuracy: 0.6948
    Epoch 202/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5684 - accuracy: 0.7413 - val_loss: 0.6501 - val_accuracy: 0.6948
    Epoch 203/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5681 - accuracy: 0.7435 - val_loss: 0.6502 - val_accuracy: 0.6948
    Epoch 204/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5678 - accuracy: 0.7435 - val_loss: 0.6502 - val_accuracy: 0.6948
    Epoch 205/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5675 - accuracy: 0.7435 - val_loss: 0.6503 - val_accuracy: 0.6948
    Epoch 206/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5672 - accuracy: 0.7435 - val_loss: 0.6503 - val_accuracy: 0.6948
    Epoch 207/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.5670 - accuracy: 0.7457 - val_loss: 0.6504 - val_accuracy: 0.6948
    Epoch 208/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.5667 - accuracy: 0.7457 - val_loss: 0.6504 - val_accuracy: 0.6948
    Epoch 209/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.5664 - accuracy: 0.7457 - val_loss: 0.6503 - val_accuracy: 0.6948
    Epoch 210/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5661 - accuracy: 0.7435 - val_loss: 0.6502 - val_accuracy: 0.6948
    Epoch 211/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.5659 - accuracy: 0.7435 - val_loss: 0.6500 - val_accuracy: 0.6948
    Epoch 212/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5656 - accuracy: 0.7457 - val_loss: 0.6499 - val_accuracy: 0.6948
    Epoch 213/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5653 - accuracy: 0.7457 - val_loss: 0.6497 - val_accuracy: 0.6948
    Epoch 214/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.5651 - accuracy: 0.7457 - val_loss: 0.6495 - val_accuracy: 0.6948
    Epoch 215/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5648 - accuracy: 0.7457 - val_loss: 0.6494 - val_accuracy: 0.6948
    Epoch 216/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5645 - accuracy: 0.7478 - val_loss: 0.6492 - val_accuracy: 0.6948
    Epoch 217/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5643 - accuracy: 0.7478 - val_loss: 0.6489 - val_accuracy: 0.6948
    Epoch 218/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5640 - accuracy: 0.7478 - val_loss: 0.6487 - val_accuracy: 0.6948
    Epoch 219/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5638 - accuracy: 0.7478 - val_loss: 0.6484 - val_accuracy: 0.6948
    Epoch 220/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5635 - accuracy: 0.7457 - val_loss: 0.6482 - val_accuracy: 0.6948
    Epoch 221/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5633 - accuracy: 0.7478 - val_loss: 0.6479 - val_accuracy: 0.6948
    Epoch 222/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5630 - accuracy: 0.7478 - val_loss: 0.6477 - val_accuracy: 0.6948
    Epoch 223/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5628 - accuracy: 0.7435 - val_loss: 0.6475 - val_accuracy: 0.6948
    Epoch 224/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5625 - accuracy: 0.7435 - val_loss: 0.6473 - val_accuracy: 0.6948
    Epoch 225/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5623 - accuracy: 0.7435 - val_loss: 0.6471 - val_accuracy: 0.6948
    Epoch 226/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.5621 - accuracy: 0.7413 - val_loss: 0.6470 - val_accuracy: 0.6948
    Epoch 227/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5618 - accuracy: 0.7413 - val_loss: 0.6469 - val_accuracy: 0.6948
    Epoch 228/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5616 - accuracy: 0.7391 - val_loss: 0.6469 - val_accuracy: 0.6948
    Epoch 229/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5613 - accuracy: 0.7391 - val_loss: 0.6468 - val_accuracy: 0.6948
    Epoch 230/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5611 - accuracy: 0.7391 - val_loss: 0.6468 - val_accuracy: 0.6948
    Epoch 231/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5609 - accuracy: 0.7391 - val_loss: 0.6467 - val_accuracy: 0.6948
    Epoch 232/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5606 - accuracy: 0.7413 - val_loss: 0.6466 - val_accuracy: 0.6948
    Epoch 233/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5604 - accuracy: 0.7413 - val_loss: 0.6465 - val_accuracy: 0.6948
    Epoch 234/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5601 - accuracy: 0.7413 - val_loss: 0.6465 - val_accuracy: 0.6883
    Epoch 235/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5599 - accuracy: 0.7435 - val_loss: 0.6464 - val_accuracy: 0.6883
    Epoch 236/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5597 - accuracy: 0.7435 - val_loss: 0.6464 - val_accuracy: 0.6883
    Epoch 237/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5595 - accuracy: 0.7435 - val_loss: 0.6464 - val_accuracy: 0.6883
    Epoch 238/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5593 - accuracy: 0.7435 - val_loss: 0.6463 - val_accuracy: 0.6883
    Epoch 239/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5590 - accuracy: 0.7435 - val_loss: 0.6462 - val_accuracy: 0.6883
    Epoch 240/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5588 - accuracy: 0.7435 - val_loss: 0.6461 - val_accuracy: 0.6883
    Epoch 241/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5586 - accuracy: 0.7435 - val_loss: 0.6460 - val_accuracy: 0.6883
    Epoch 242/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5584 - accuracy: 0.7435 - val_loss: 0.6458 - val_accuracy: 0.6883
    Epoch 243/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5581 - accuracy: 0.7435 - val_loss: 0.6457 - val_accuracy: 0.6883
    Epoch 244/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.5579 - accuracy: 0.7435 - val_loss: 0.6454 - val_accuracy: 0.6883
    Epoch 245/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5577 - accuracy: 0.7413 - val_loss: 0.6452 - val_accuracy: 0.6883
    Epoch 246/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5574 - accuracy: 0.7413 - val_loss: 0.6449 - val_accuracy: 0.6883
    Epoch 247/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5571 - accuracy: 0.7413 - val_loss: 0.6447 - val_accuracy: 0.6883
    Epoch 248/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5568 - accuracy: 0.7435 - val_loss: 0.6444 - val_accuracy: 0.6883
    Epoch 249/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5565 - accuracy: 0.7435 - val_loss: 0.6442 - val_accuracy: 0.6883
    Epoch 250/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5562 - accuracy: 0.7435 - val_loss: 0.6440 - val_accuracy: 0.6883
    Epoch 251/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5559 - accuracy: 0.7435 - val_loss: 0.6439 - val_accuracy: 0.6818
    Epoch 252/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5555 - accuracy: 0.7457 - val_loss: 0.6438 - val_accuracy: 0.6818
    Epoch 253/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5552 - accuracy: 0.7478 - val_loss: 0.6438 - val_accuracy: 0.6883
    Epoch 254/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5549 - accuracy: 0.7478 - val_loss: 0.6438 - val_accuracy: 0.6883
    Epoch 255/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5545 - accuracy: 0.7478 - val_loss: 0.6439 - val_accuracy: 0.6883
    Epoch 256/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5542 - accuracy: 0.7478 - val_loss: 0.6440 - val_accuracy: 0.6883
    Epoch 257/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5539 - accuracy: 0.7478 - val_loss: 0.6441 - val_accuracy: 0.6818
    Epoch 258/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5536 - accuracy: 0.7478 - val_loss: 0.6442 - val_accuracy: 0.6818
    Epoch 259/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5532 - accuracy: 0.7478 - val_loss: 0.6443 - val_accuracy: 0.6818
    Epoch 260/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5529 - accuracy: 0.7478 - val_loss: 0.6443 - val_accuracy: 0.6818
    Epoch 261/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5526 - accuracy: 0.7478 - val_loss: 0.6442 - val_accuracy: 0.6818
    Epoch 262/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5522 - accuracy: 0.7457 - val_loss: 0.6442 - val_accuracy: 0.6818
    Epoch 263/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5519 - accuracy: 0.7457 - val_loss: 0.6441 - val_accuracy: 0.6818
    Epoch 264/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5516 - accuracy: 0.7457 - val_loss: 0.6440 - val_accuracy: 0.6818
    Epoch 265/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5513 - accuracy: 0.7457 - val_loss: 0.6438 - val_accuracy: 0.6818
    Epoch 266/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5509 - accuracy: 0.7435 - val_loss: 0.6435 - val_accuracy: 0.6818
    Epoch 267/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5506 - accuracy: 0.7478 - val_loss: 0.6432 - val_accuracy: 0.6818
    Epoch 268/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.5502 - accuracy: 0.7478 - val_loss: 0.6429 - val_accuracy: 0.6818
    Epoch 269/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.5499 - accuracy: 0.7478 - val_loss: 0.6426 - val_accuracy: 0.6818
    Epoch 270/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5496 - accuracy: 0.7478 - val_loss: 0.6423 - val_accuracy: 0.6818
    Epoch 271/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5493 - accuracy: 0.7478 - val_loss: 0.6421 - val_accuracy: 0.6818
    Epoch 272/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5490 - accuracy: 0.7478 - val_loss: 0.6421 - val_accuracy: 0.6818
    Epoch 273/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5487 - accuracy: 0.7457 - val_loss: 0.6420 - val_accuracy: 0.6818
    Epoch 274/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5485 - accuracy: 0.7457 - val_loss: 0.6420 - val_accuracy: 0.6818
    Epoch 275/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5482 - accuracy: 0.7457 - val_loss: 0.6419 - val_accuracy: 0.6818
    Epoch 276/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5479 - accuracy: 0.7457 - val_loss: 0.6419 - val_accuracy: 0.6818
    Epoch 277/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5477 - accuracy: 0.7457 - val_loss: 0.6418 - val_accuracy: 0.6753
    Epoch 278/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5474 - accuracy: 0.7457 - val_loss: 0.6417 - val_accuracy: 0.6753
    Epoch 279/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5471 - accuracy: 0.7457 - val_loss: 0.6415 - val_accuracy: 0.6883
    Epoch 280/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5468 - accuracy: 0.7457 - val_loss: 0.6412 - val_accuracy: 0.6883
    Epoch 281/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5465 - accuracy: 0.7457 - val_loss: 0.6408 - val_accuracy: 0.6883
    Epoch 282/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5462 - accuracy: 0.7457 - val_loss: 0.6405 - val_accuracy: 0.6883
    Epoch 283/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5459 - accuracy: 0.7478 - val_loss: 0.6404 - val_accuracy: 0.6883
    Epoch 284/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5456 - accuracy: 0.7435 - val_loss: 0.6404 - val_accuracy: 0.6883
    Epoch 285/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5453 - accuracy: 0.7435 - val_loss: 0.6405 - val_accuracy: 0.6883
    Epoch 286/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5451 - accuracy: 0.7457 - val_loss: 0.6407 - val_accuracy: 0.6883
    Epoch 287/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5448 - accuracy: 0.7457 - val_loss: 0.6408 - val_accuracy: 0.6883
    Epoch 288/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5445 - accuracy: 0.7435 - val_loss: 0.6410 - val_accuracy: 0.6883
    Epoch 289/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.5442 - accuracy: 0.7413 - val_loss: 0.6411 - val_accuracy: 0.6883
    Epoch 290/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5440 - accuracy: 0.7413 - val_loss: 0.6412 - val_accuracy: 0.6818
    Epoch 291/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5437 - accuracy: 0.7413 - val_loss: 0.6413 - val_accuracy: 0.6818
    Epoch 292/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5434 - accuracy: 0.7413 - val_loss: 0.6413 - val_accuracy: 0.6818
    Epoch 293/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5431 - accuracy: 0.7413 - val_loss: 0.6412 - val_accuracy: 0.6818
    Epoch 294/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5429 - accuracy: 0.7391 - val_loss: 0.6410 - val_accuracy: 0.6818
    Epoch 295/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5426 - accuracy: 0.7391 - val_loss: 0.6408 - val_accuracy: 0.6818
    Epoch 296/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5423 - accuracy: 0.7370 - val_loss: 0.6406 - val_accuracy: 0.6818
    Epoch 297/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5420 - accuracy: 0.7370 - val_loss: 0.6404 - val_accuracy: 0.6818
    Epoch 298/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5417 - accuracy: 0.7391 - val_loss: 0.6401 - val_accuracy: 0.6818
    Epoch 299/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5414 - accuracy: 0.7391 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 300/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5411 - accuracy: 0.7391 - val_loss: 0.6397 - val_accuracy: 0.6818
    Epoch 301/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5408 - accuracy: 0.7391 - val_loss: 0.6397 - val_accuracy: 0.6818
    Epoch 302/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5405 - accuracy: 0.7435 - val_loss: 0.6397 - val_accuracy: 0.6883
    Epoch 303/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5402 - accuracy: 0.7435 - val_loss: 0.6397 - val_accuracy: 0.6883
    Epoch 304/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5398 - accuracy: 0.7435 - val_loss: 0.6398 - val_accuracy: 0.6883
    Epoch 305/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5394 - accuracy: 0.7435 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 306/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5390 - accuracy: 0.7413 - val_loss: 0.6401 - val_accuracy: 0.6818
    Epoch 307/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.5386 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 308/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5382 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 309/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5378 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 310/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5375 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 311/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5373 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 312/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5370 - accuracy: 0.7435 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 313/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5368 - accuracy: 0.7457 - val_loss: 0.6403 - val_accuracy: 0.6818
    Epoch 314/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5365 - accuracy: 0.7500 - val_loss: 0.6405 - val_accuracy: 0.6818
    Epoch 315/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5362 - accuracy: 0.7478 - val_loss: 0.6406 - val_accuracy: 0.6818
    Epoch 316/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5360 - accuracy: 0.7478 - val_loss: 0.6405 - val_accuracy: 0.6818
    Epoch 317/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5358 - accuracy: 0.7478 - val_loss: 0.6404 - val_accuracy: 0.6818
    Epoch 318/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5355 - accuracy: 0.7457 - val_loss: 0.6401 - val_accuracy: 0.6818
    Epoch 319/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5353 - accuracy: 0.7457 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 320/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5351 - accuracy: 0.7500 - val_loss: 0.6397 - val_accuracy: 0.6818
    Epoch 321/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5349 - accuracy: 0.7500 - val_loss: 0.6397 - val_accuracy: 0.6818
    Epoch 322/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5346 - accuracy: 0.7478 - val_loss: 0.6398 - val_accuracy: 0.6818
    Epoch 323/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.5344 - accuracy: 0.7478 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 324/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5341 - accuracy: 0.7478 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 325/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5339 - accuracy: 0.7478 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 326/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5337 - accuracy: 0.7478 - val_loss: 0.6398 - val_accuracy: 0.6818
    Epoch 327/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5335 - accuracy: 0.7500 - val_loss: 0.6398 - val_accuracy: 0.6818
    Epoch 328/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5332 - accuracy: 0.7500 - val_loss: 0.6399 - val_accuracy: 0.6818
    Epoch 329/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5330 - accuracy: 0.7478 - val_loss: 0.6400 - val_accuracy: 0.6818
    Epoch 330/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5328 - accuracy: 0.7478 - val_loss: 0.6402 - val_accuracy: 0.6818
    Epoch 331/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5326 - accuracy: 0.7478 - val_loss: 0.6404 - val_accuracy: 0.6818
    Epoch 332/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5324 - accuracy: 0.7478 - val_loss: 0.6407 - val_accuracy: 0.6818
    Epoch 333/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5322 - accuracy: 0.7478 - val_loss: 0.6410 - val_accuracy: 0.6818
    Epoch 334/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5319 - accuracy: 0.7478 - val_loss: 0.6412 - val_accuracy: 0.6818
    Epoch 335/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5317 - accuracy: 0.7500 - val_loss: 0.6414 - val_accuracy: 0.6818
    Epoch 336/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5315 - accuracy: 0.7500 - val_loss: 0.6416 - val_accuracy: 0.6818
    Epoch 337/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5312 - accuracy: 0.7500 - val_loss: 0.6418 - val_accuracy: 0.6818
    Epoch 338/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5310 - accuracy: 0.7500 - val_loss: 0.6420 - val_accuracy: 0.6818
    Epoch 339/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5308 - accuracy: 0.7500 - val_loss: 0.6422 - val_accuracy: 0.6753
    Epoch 340/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5305 - accuracy: 0.7500 - val_loss: 0.6425 - val_accuracy: 0.6753
    Epoch 341/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5302 - accuracy: 0.7500 - val_loss: 0.6427 - val_accuracy: 0.6753
    Epoch 342/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5300 - accuracy: 0.7500 - val_loss: 0.6428 - val_accuracy: 0.6753
    Epoch 343/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5297 - accuracy: 0.7500 - val_loss: 0.6427 - val_accuracy: 0.6753
    Epoch 344/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5295 - accuracy: 0.7500 - val_loss: 0.6425 - val_accuracy: 0.6753
    Epoch 345/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5292 - accuracy: 0.7522 - val_loss: 0.6423 - val_accuracy: 0.6753
    Epoch 346/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5290 - accuracy: 0.7500 - val_loss: 0.6421 - val_accuracy: 0.6818
    Epoch 347/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.5288 - accuracy: 0.7500 - val_loss: 0.6421 - val_accuracy: 0.6818
    Epoch 348/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5286 - accuracy: 0.7500 - val_loss: 0.6421 - val_accuracy: 0.6753
    Epoch 349/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5284 - accuracy: 0.7500 - val_loss: 0.6421 - val_accuracy: 0.6753
    Epoch 350/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5282 - accuracy: 0.7522 - val_loss: 0.6422 - val_accuracy: 0.6753
    Epoch 351/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5280 - accuracy: 0.7500 - val_loss: 0.6421 - val_accuracy: 0.6753
    Epoch 352/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5279 - accuracy: 0.7500 - val_loss: 0.6419 - val_accuracy: 0.6753
    Epoch 353/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5277 - accuracy: 0.7500 - val_loss: 0.6417 - val_accuracy: 0.6753
    Epoch 354/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5275 - accuracy: 0.7522 - val_loss: 0.6415 - val_accuracy: 0.6753
    Epoch 355/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5272 - accuracy: 0.7500 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 356/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5270 - accuracy: 0.7478 - val_loss: 0.6413 - val_accuracy: 0.6753
    Epoch 357/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5268 - accuracy: 0.7478 - val_loss: 0.6413 - val_accuracy: 0.6753
    Epoch 358/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5266 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 359/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5264 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 360/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5262 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 361/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.5259 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 362/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5257 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6753
    Epoch 363/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5255 - accuracy: 0.7478 - val_loss: 0.6415 - val_accuracy: 0.6818
    Epoch 364/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5252 - accuracy: 0.7478 - val_loss: 0.6415 - val_accuracy: 0.6818
    Epoch 365/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5250 - accuracy: 0.7478 - val_loss: 0.6415 - val_accuracy: 0.6818
    Epoch 366/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.5248 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6818
    Epoch 367/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5246 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6818
    Epoch 368/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5243 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6818
    Epoch 369/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5241 - accuracy: 0.7478 - val_loss: 0.6414 - val_accuracy: 0.6818
    Epoch 370/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.5239 - accuracy: 0.7478 - val_loss: 0.6412 - val_accuracy: 0.6883
    Epoch 371/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5237 - accuracy: 0.7478 - val_loss: 0.6411 - val_accuracy: 0.6883
    Epoch 372/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5235 - accuracy: 0.7478 - val_loss: 0.6410 - val_accuracy: 0.6883
    Epoch 373/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5233 - accuracy: 0.7478 - val_loss: 0.6409 - val_accuracy: 0.6883
    Epoch 374/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5231 - accuracy: 0.7478 - val_loss: 0.6408 - val_accuracy: 0.6883
    Epoch 375/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5228 - accuracy: 0.7500 - val_loss: 0.6407 - val_accuracy: 0.6883
    Epoch 376/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5226 - accuracy: 0.7500 - val_loss: 0.6405 - val_accuracy: 0.6883
    Epoch 377/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5224 - accuracy: 0.7478 - val_loss: 0.6404 - val_accuracy: 0.6883
    Epoch 378/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5222 - accuracy: 0.7478 - val_loss: 0.6403 - val_accuracy: 0.6883
    Epoch 379/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.5220 - accuracy: 0.7478 - val_loss: 0.6402 - val_accuracy: 0.6883
    Epoch 380/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5218 - accuracy: 0.7500 - val_loss: 0.6401 - val_accuracy: 0.6883
    Epoch 381/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5216 - accuracy: 0.7500 - val_loss: 0.6400 - val_accuracy: 0.6883
    Epoch 382/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5215 - accuracy: 0.7500 - val_loss: 0.6399 - val_accuracy: 0.6883
    Epoch 383/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5213 - accuracy: 0.7500 - val_loss: 0.6399 - val_accuracy: 0.6883
    Epoch 384/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5211 - accuracy: 0.7500 - val_loss: 0.6398 - val_accuracy: 0.6883
    Epoch 385/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5209 - accuracy: 0.7500 - val_loss: 0.6398 - val_accuracy: 0.6883
    Epoch 386/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5208 - accuracy: 0.7500 - val_loss: 0.6398 - val_accuracy: 0.6883
    Epoch 387/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5206 - accuracy: 0.7500 - val_loss: 0.6398 - val_accuracy: 0.6883
    Epoch 388/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5204 - accuracy: 0.7500 - val_loss: 0.6399 - val_accuracy: 0.6883
    Epoch 389/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5202 - accuracy: 0.7522 - val_loss: 0.6399 - val_accuracy: 0.6948
    Epoch 390/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5200 - accuracy: 0.7522 - val_loss: 0.6400 - val_accuracy: 0.6948
    Epoch 391/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5199 - accuracy: 0.7522 - val_loss: 0.6401 - val_accuracy: 0.6948
    Epoch 392/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5197 - accuracy: 0.7522 - val_loss: 0.6402 - val_accuracy: 0.6948
    Epoch 393/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5195 - accuracy: 0.7522 - val_loss: 0.6401 - val_accuracy: 0.6948
    Epoch 394/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5194 - accuracy: 0.7522 - val_loss: 0.6401 - val_accuracy: 0.6948
    Epoch 395/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5192 - accuracy: 0.7522 - val_loss: 0.6401 - val_accuracy: 0.6948
    Epoch 396/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5190 - accuracy: 0.7522 - val_loss: 0.6401 - val_accuracy: 0.6948
    Epoch 397/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5188 - accuracy: 0.7522 - val_loss: 0.6402 - val_accuracy: 0.6948
    Epoch 398/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5186 - accuracy: 0.7522 - val_loss: 0.6403 - val_accuracy: 0.6948
    Epoch 399/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5184 - accuracy: 0.7522 - val_loss: 0.6403 - val_accuracy: 0.6948
    Epoch 400/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5182 - accuracy: 0.7500 - val_loss: 0.6403 - val_accuracy: 0.7013
    Epoch 401/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5180 - accuracy: 0.7500 - val_loss: 0.6403 - val_accuracy: 0.7013
    Epoch 402/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5178 - accuracy: 0.7543 - val_loss: 0.6402 - val_accuracy: 0.7013
    Epoch 403/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5176 - accuracy: 0.7543 - val_loss: 0.6400 - val_accuracy: 0.6948
    Epoch 404/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5174 - accuracy: 0.7522 - val_loss: 0.6400 - val_accuracy: 0.6948
    Epoch 405/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5172 - accuracy: 0.7522 - val_loss: 0.6399 - val_accuracy: 0.7013
    Epoch 406/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5171 - accuracy: 0.7522 - val_loss: 0.6400 - val_accuracy: 0.7013
    Epoch 407/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.5169 - accuracy: 0.7522 - val_loss: 0.6400 - val_accuracy: 0.6948
    Epoch 408/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.5167 - accuracy: 0.7543 - val_loss: 0.6399 - val_accuracy: 0.6948
    Epoch 409/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5166 - accuracy: 0.7543 - val_loss: 0.6397 - val_accuracy: 0.6948
    Epoch 410/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5164 - accuracy: 0.7543 - val_loss: 0.6392 - val_accuracy: 0.7078
    Epoch 411/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5162 - accuracy: 0.7565 - val_loss: 0.6387 - val_accuracy: 0.7078
    Epoch 412/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5160 - accuracy: 0.7565 - val_loss: 0.6384 - val_accuracy: 0.7078
    Epoch 413/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5158 - accuracy: 0.7565 - val_loss: 0.6382 - val_accuracy: 0.7078
    Epoch 414/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5157 - accuracy: 0.7565 - val_loss: 0.6381 - val_accuracy: 0.7078
    Epoch 415/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5155 - accuracy: 0.7587 - val_loss: 0.6380 - val_accuracy: 0.7078
    Epoch 416/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5154 - accuracy: 0.7565 - val_loss: 0.6378 - val_accuracy: 0.7078
    Epoch 417/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5152 - accuracy: 0.7543 - val_loss: 0.6377 - val_accuracy: 0.7078
    Epoch 418/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.5151 - accuracy: 0.7565 - val_loss: 0.6375 - val_accuracy: 0.7078
    Epoch 419/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5149 - accuracy: 0.7565 - val_loss: 0.6372 - val_accuracy: 0.7078
    Epoch 420/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5148 - accuracy: 0.7565 - val_loss: 0.6370 - val_accuracy: 0.7078
    Epoch 421/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5146 - accuracy: 0.7565 - val_loss: 0.6369 - val_accuracy: 0.7078
    Epoch 422/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5145 - accuracy: 0.7587 - val_loss: 0.6368 - val_accuracy: 0.7078
    Epoch 423/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5143 - accuracy: 0.7587 - val_loss: 0.6366 - val_accuracy: 0.7078
    Epoch 424/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5142 - accuracy: 0.7565 - val_loss: 0.6363 - val_accuracy: 0.7078
    Epoch 425/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5140 - accuracy: 0.7565 - val_loss: 0.6360 - val_accuracy: 0.7078
    Epoch 426/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5139 - accuracy: 0.7565 - val_loss: 0.6356 - val_accuracy: 0.7078
    Epoch 427/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5138 - accuracy: 0.7587 - val_loss: 0.6353 - val_accuracy: 0.7078
    Epoch 428/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.5136 - accuracy: 0.7609 - val_loss: 0.6350 - val_accuracy: 0.7078
    Epoch 429/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5135 - accuracy: 0.7609 - val_loss: 0.6348 - val_accuracy: 0.7078
    Epoch 430/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5133 - accuracy: 0.7565 - val_loss: 0.6347 - val_accuracy: 0.7078
    Epoch 431/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5132 - accuracy: 0.7565 - val_loss: 0.6346 - val_accuracy: 0.7143
    Epoch 432/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5131 - accuracy: 0.7565 - val_loss: 0.6344 - val_accuracy: 0.7013
    Epoch 433/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.5129 - accuracy: 0.7565 - val_loss: 0.6342 - val_accuracy: 0.7013
    Epoch 434/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5128 - accuracy: 0.7565 - val_loss: 0.6341 - val_accuracy: 0.7013
    Epoch 435/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5127 - accuracy: 0.7587 - val_loss: 0.6341 - val_accuracy: 0.7013
    Epoch 436/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5125 - accuracy: 0.7587 - val_loss: 0.6342 - val_accuracy: 0.7013
    Epoch 437/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5124 - accuracy: 0.7587 - val_loss: 0.6343 - val_accuracy: 0.7078
    Epoch 438/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5123 - accuracy: 0.7565 - val_loss: 0.6343 - val_accuracy: 0.7078
    Epoch 439/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5122 - accuracy: 0.7587 - val_loss: 0.6343 - val_accuracy: 0.7078
    Epoch 440/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.5120 - accuracy: 0.7587 - val_loss: 0.6341 - val_accuracy: 0.7078
    Epoch 441/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5119 - accuracy: 0.7609 - val_loss: 0.6339 - val_accuracy: 0.7078
    Epoch 442/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5118 - accuracy: 0.7587 - val_loss: 0.6337 - val_accuracy: 0.7078
    Epoch 443/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5116 - accuracy: 0.7565 - val_loss: 0.6336 - val_accuracy: 0.7078
    Epoch 444/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5115 - accuracy: 0.7587 - val_loss: 0.6336 - val_accuracy: 0.7078
    Epoch 445/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5114 - accuracy: 0.7587 - val_loss: 0.6335 - val_accuracy: 0.7078
    Epoch 446/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5113 - accuracy: 0.7587 - val_loss: 0.6335 - val_accuracy: 0.7078
    Epoch 447/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5111 - accuracy: 0.7609 - val_loss: 0.6333 - val_accuracy: 0.7078
    Epoch 448/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5110 - accuracy: 0.7609 - val_loss: 0.6331 - val_accuracy: 0.7078
    Epoch 449/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.5109 - accuracy: 0.7609 - val_loss: 0.6329 - val_accuracy: 0.7078
    Epoch 450/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5108 - accuracy: 0.7609 - val_loss: 0.6328 - val_accuracy: 0.7078
    Epoch 451/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5106 - accuracy: 0.7609 - val_loss: 0.6327 - val_accuracy: 0.7078
    Epoch 452/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5105 - accuracy: 0.7609 - val_loss: 0.6327 - val_accuracy: 0.7078
    Epoch 453/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5104 - accuracy: 0.7609 - val_loss: 0.6326 - val_accuracy: 0.7078
    Epoch 454/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5103 - accuracy: 0.7609 - val_loss: 0.6324 - val_accuracy: 0.7078
    Epoch 455/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5102 - accuracy: 0.7587 - val_loss: 0.6322 - val_accuracy: 0.7078
    Epoch 456/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5101 - accuracy: 0.7587 - val_loss: 0.6321 - val_accuracy: 0.7078
    Epoch 457/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5099 - accuracy: 0.7587 - val_loss: 0.6320 - val_accuracy: 0.7078
    Epoch 458/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5098 - accuracy: 0.7609 - val_loss: 0.6318 - val_accuracy: 0.7078
    Epoch 459/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5097 - accuracy: 0.7609 - val_loss: 0.6316 - val_accuracy: 0.7078
    Epoch 460/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5096 - accuracy: 0.7609 - val_loss: 0.6316 - val_accuracy: 0.7078
    Epoch 461/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5095 - accuracy: 0.7609 - val_loss: 0.6314 - val_accuracy: 0.7078
    Epoch 462/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5093 - accuracy: 0.7609 - val_loss: 0.6315 - val_accuracy: 0.7078
    Epoch 463/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5092 - accuracy: 0.7565 - val_loss: 0.6314 - val_accuracy: 0.7078
    Epoch 464/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5091 - accuracy: 0.7565 - val_loss: 0.6311 - val_accuracy: 0.7078
    Epoch 465/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5090 - accuracy: 0.7587 - val_loss: 0.6310 - val_accuracy: 0.7078
    Epoch 466/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5089 - accuracy: 0.7609 - val_loss: 0.6311 - val_accuracy: 0.7078
    Epoch 467/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5087 - accuracy: 0.7609 - val_loss: 0.6312 - val_accuracy: 0.7078
    Epoch 468/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5086 - accuracy: 0.7609 - val_loss: 0.6311 - val_accuracy: 0.7078
    Epoch 469/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5085 - accuracy: 0.7609 - val_loss: 0.6310 - val_accuracy: 0.7078
    Epoch 470/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5084 - accuracy: 0.7609 - val_loss: 0.6310 - val_accuracy: 0.7078
    Epoch 471/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5083 - accuracy: 0.7609 - val_loss: 0.6311 - val_accuracy: 0.7078
    Epoch 472/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5081 - accuracy: 0.7609 - val_loss: 0.6310 - val_accuracy: 0.7078
    Epoch 473/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5080 - accuracy: 0.7609 - val_loss: 0.6308 - val_accuracy: 0.7078
    Epoch 474/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5079 - accuracy: 0.7609 - val_loss: 0.6306 - val_accuracy: 0.7078
    Epoch 475/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5078 - accuracy: 0.7609 - val_loss: 0.6304 - val_accuracy: 0.7078
    Epoch 476/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5076 - accuracy: 0.7609 - val_loss: 0.6302 - val_accuracy: 0.7078
    Epoch 477/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5075 - accuracy: 0.7609 - val_loss: 0.6299 - val_accuracy: 0.7078
    Epoch 478/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5074 - accuracy: 0.7609 - val_loss: 0.6297 - val_accuracy: 0.7078
    Epoch 479/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5073 - accuracy: 0.7609 - val_loss: 0.6293 - val_accuracy: 0.7078
    Epoch 480/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5072 - accuracy: 0.7609 - val_loss: 0.6290 - val_accuracy: 0.7078
    Epoch 481/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5071 - accuracy: 0.7609 - val_loss: 0.6287 - val_accuracy: 0.7078
    Epoch 482/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5070 - accuracy: 0.7609 - val_loss: 0.6285 - val_accuracy: 0.7078
    Epoch 483/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5069 - accuracy: 0.7587 - val_loss: 0.6282 - val_accuracy: 0.7078
    Epoch 484/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5068 - accuracy: 0.7587 - val_loss: 0.6278 - val_accuracy: 0.7078
    Epoch 485/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5066 - accuracy: 0.7609 - val_loss: 0.6274 - val_accuracy: 0.7078
    Epoch 486/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5065 - accuracy: 0.7609 - val_loss: 0.6272 - val_accuracy: 0.7078
    Epoch 487/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5064 - accuracy: 0.7609 - val_loss: 0.6271 - val_accuracy: 0.7078
    Epoch 488/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5063 - accuracy: 0.7609 - val_loss: 0.6272 - val_accuracy: 0.7078
    Epoch 489/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5062 - accuracy: 0.7630 - val_loss: 0.6273 - val_accuracy: 0.7078
    Epoch 490/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5061 - accuracy: 0.7630 - val_loss: 0.6272 - val_accuracy: 0.7078
    Epoch 491/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5060 - accuracy: 0.7630 - val_loss: 0.6272 - val_accuracy: 0.7078
    Epoch 492/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.5059 - accuracy: 0.7630 - val_loss: 0.6273 - val_accuracy: 0.7078
    Epoch 493/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.5058 - accuracy: 0.7630 - val_loss: 0.6274 - val_accuracy: 0.7078
    Epoch 494/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5057 - accuracy: 0.7630 - val_loss: 0.6273 - val_accuracy: 0.7078
    Epoch 495/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5056 - accuracy: 0.7630 - val_loss: 0.6271 - val_accuracy: 0.7078
    Epoch 496/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5055 - accuracy: 0.7630 - val_loss: 0.6268 - val_accuracy: 0.7078
    Epoch 497/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5054 - accuracy: 0.7630 - val_loss: 0.6267 - val_accuracy: 0.7078
    Epoch 498/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.5053 - accuracy: 0.7630 - val_loss: 0.6265 - val_accuracy: 0.7078
    Epoch 499/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5052 - accuracy: 0.7630 - val_loss: 0.6263 - val_accuracy: 0.7078
    Epoch 500/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5051 - accuracy: 0.7630 - val_loss: 0.6262 - val_accuracy: 0.7078
    Epoch 501/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5050 - accuracy: 0.7630 - val_loss: 0.6260 - val_accuracy: 0.7078
    Epoch 502/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.5049 - accuracy: 0.7630 - val_loss: 0.6260 - val_accuracy: 0.7078
    Epoch 503/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5047 - accuracy: 0.7630 - val_loss: 0.6261 - val_accuracy: 0.7078
    Epoch 504/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5046 - accuracy: 0.7652 - val_loss: 0.6262 - val_accuracy: 0.7078
    Epoch 505/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5045 - accuracy: 0.7630 - val_loss: 0.6262 - val_accuracy: 0.7078
    Epoch 506/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5044 - accuracy: 0.7630 - val_loss: 0.6261 - val_accuracy: 0.7078
    Epoch 507/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5043 - accuracy: 0.7652 - val_loss: 0.6260 - val_accuracy: 0.7078
    Epoch 508/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5042 - accuracy: 0.7630 - val_loss: 0.6258 - val_accuracy: 0.7078
    Epoch 509/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5041 - accuracy: 0.7630 - val_loss: 0.6255 - val_accuracy: 0.7078
    Epoch 510/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5040 - accuracy: 0.7652 - val_loss: 0.6252 - val_accuracy: 0.7078
    Epoch 511/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5039 - accuracy: 0.7652 - val_loss: 0.6251 - val_accuracy: 0.7078
    Epoch 512/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5038 - accuracy: 0.7652 - val_loss: 0.6251 - val_accuracy: 0.7078
    Epoch 513/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.5037 - accuracy: 0.7630 - val_loss: 0.6251 - val_accuracy: 0.7078
    Epoch 514/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5036 - accuracy: 0.7630 - val_loss: 0.6250 - val_accuracy: 0.7078
    Epoch 515/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5034 - accuracy: 0.7630 - val_loss: 0.6249 - val_accuracy: 0.7078
    Epoch 516/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5033 - accuracy: 0.7630 - val_loss: 0.6250 - val_accuracy: 0.7078
    Epoch 517/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5032 - accuracy: 0.7630 - val_loss: 0.6249 - val_accuracy: 0.7078
    Epoch 518/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5031 - accuracy: 0.7630 - val_loss: 0.6246 - val_accuracy: 0.7078
    Epoch 519/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5030 - accuracy: 0.7652 - val_loss: 0.6243 - val_accuracy: 0.7078
    Epoch 520/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5029 - accuracy: 0.7652 - val_loss: 0.6241 - val_accuracy: 0.7078
    Epoch 521/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.5028 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 522/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.5027 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 523/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5026 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 524/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5025 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 525/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5024 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 526/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5023 - accuracy: 0.7652 - val_loss: 0.6238 - val_accuracy: 0.7078
    Epoch 527/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5022 - accuracy: 0.7652 - val_loss: 0.6238 - val_accuracy: 0.7078
    Epoch 528/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.5021 - accuracy: 0.7652 - val_loss: 0.6238 - val_accuracy: 0.7078
    Epoch 529/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5020 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7078
    Epoch 530/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.5019 - accuracy: 0.7652 - val_loss: 0.6240 - val_accuracy: 0.7078
    Epoch 531/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5018 - accuracy: 0.7652 - val_loss: 0.6239 - val_accuracy: 0.7013
    Epoch 532/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5017 - accuracy: 0.7652 - val_loss: 0.6238 - val_accuracy: 0.7013
    Epoch 533/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5016 - accuracy: 0.7652 - val_loss: 0.6237 - val_accuracy: 0.7013
    Epoch 534/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.5015 - accuracy: 0.7652 - val_loss: 0.6234 - val_accuracy: 0.7078
    Epoch 535/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.5014 - accuracy: 0.7652 - val_loss: 0.6231 - val_accuracy: 0.7078
    Epoch 536/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.5013 - accuracy: 0.7652 - val_loss: 0.6230 - val_accuracy: 0.7078
    Epoch 537/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5012 - accuracy: 0.7652 - val_loss: 0.6231 - val_accuracy: 0.7013
    Epoch 538/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5011 - accuracy: 0.7652 - val_loss: 0.6232 - val_accuracy: 0.7013
    Epoch 539/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.5010 - accuracy: 0.7674 - val_loss: 0.6234 - val_accuracy: 0.7013
    Epoch 540/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.5009 - accuracy: 0.7674 - val_loss: 0.6234 - val_accuracy: 0.7013
    Epoch 541/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.5008 - accuracy: 0.7674 - val_loss: 0.6233 - val_accuracy: 0.7013
    Epoch 542/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5007 - accuracy: 0.7674 - val_loss: 0.6231 - val_accuracy: 0.7013
    Epoch 543/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5006 - accuracy: 0.7652 - val_loss: 0.6229 - val_accuracy: 0.7013
    Epoch 544/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.5005 - accuracy: 0.7652 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 545/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.5004 - accuracy: 0.7652 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 546/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.5003 - accuracy: 0.7674 - val_loss: 0.6227 - val_accuracy: 0.7013
    Epoch 547/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5002 - accuracy: 0.7652 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 548/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.5001 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 549/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.5000 - accuracy: 0.7674 - val_loss: 0.6226 - val_accuracy: 0.7013
    Epoch 550/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4999 - accuracy: 0.7674 - val_loss: 0.6224 - val_accuracy: 0.7013
    Epoch 551/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4998 - accuracy: 0.7652 - val_loss: 0.6224 - val_accuracy: 0.7013
    Epoch 552/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4997 - accuracy: 0.7674 - val_loss: 0.6223 - val_accuracy: 0.7013
    Epoch 553/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4996 - accuracy: 0.7674 - val_loss: 0.6222 - val_accuracy: 0.7013
    Epoch 554/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4995 - accuracy: 0.7674 - val_loss: 0.6222 - val_accuracy: 0.7013
    Epoch 555/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4994 - accuracy: 0.7652 - val_loss: 0.6224 - val_accuracy: 0.7013
    Epoch 556/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4992 - accuracy: 0.7630 - val_loss: 0.6225 - val_accuracy: 0.7013
    Epoch 557/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4991 - accuracy: 0.7674 - val_loss: 0.6227 - val_accuracy: 0.7013
    Epoch 558/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4990 - accuracy: 0.7674 - val_loss: 0.6227 - val_accuracy: 0.7013
    Epoch 559/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4989 - accuracy: 0.7696 - val_loss: 0.6225 - val_accuracy: 0.7013
    Epoch 560/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4988 - accuracy: 0.7674 - val_loss: 0.6226 - val_accuracy: 0.7013
    Epoch 561/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4987 - accuracy: 0.7652 - val_loss: 0.6227 - val_accuracy: 0.7013
    Epoch 562/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4986 - accuracy: 0.7674 - val_loss: 0.6229 - val_accuracy: 0.7013
    Epoch 563/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4985 - accuracy: 0.7674 - val_loss: 0.6229 - val_accuracy: 0.7013
    Epoch 564/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4983 - accuracy: 0.7674 - val_loss: 0.6229 - val_accuracy: 0.7013
    Epoch 565/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4982 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 566/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4981 - accuracy: 0.7674 - val_loss: 0.6229 - val_accuracy: 0.7013
    Epoch 567/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4980 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 568/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4979 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 569/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4978 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 570/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4977 - accuracy: 0.7674 - val_loss: 0.6228 - val_accuracy: 0.7013
    Epoch 571/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4976 - accuracy: 0.7696 - val_loss: 0.6227 - val_accuracy: 0.7013
    Epoch 572/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4975 - accuracy: 0.7674 - val_loss: 0.6227 - val_accuracy: 0.7078
    Epoch 573/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4974 - accuracy: 0.7674 - val_loss: 0.6225 - val_accuracy: 0.7013
    Epoch 574/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4973 - accuracy: 0.7674 - val_loss: 0.6225 - val_accuracy: 0.7078
    Epoch 575/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4972 - accuracy: 0.7674 - val_loss: 0.6226 - val_accuracy: 0.7078
    Epoch 576/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4971 - accuracy: 0.7696 - val_loss: 0.6225 - val_accuracy: 0.7078
    Epoch 577/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4970 - accuracy: 0.7674 - val_loss: 0.6224 - val_accuracy: 0.7013
    Epoch 578/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4969 - accuracy: 0.7674 - val_loss: 0.6224 - val_accuracy: 0.7013
    Epoch 579/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4968 - accuracy: 0.7674 - val_loss: 0.6223 - val_accuracy: 0.7013
    Epoch 580/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4968 - accuracy: 0.7674 - val_loss: 0.6225 - val_accuracy: 0.7078
    Epoch 581/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4966 - accuracy: 0.7696 - val_loss: 0.6224 - val_accuracy: 0.7078
    Epoch 582/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4965 - accuracy: 0.7696 - val_loss: 0.6221 - val_accuracy: 0.7013
    Epoch 583/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4965 - accuracy: 0.7674 - val_loss: 0.6220 - val_accuracy: 0.7078
    Epoch 584/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4964 - accuracy: 0.7674 - val_loss: 0.6220 - val_accuracy: 0.7078
    Epoch 585/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4963 - accuracy: 0.7674 - val_loss: 0.6219 - val_accuracy: 0.7143
    Epoch 586/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4962 - accuracy: 0.7674 - val_loss: 0.6218 - val_accuracy: 0.7143
    Epoch 587/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4961 - accuracy: 0.7674 - val_loss: 0.6217 - val_accuracy: 0.7143
    Epoch 588/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4960 - accuracy: 0.7674 - val_loss: 0.6216 - val_accuracy: 0.7143
    Epoch 589/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4959 - accuracy: 0.7674 - val_loss: 0.6215 - val_accuracy: 0.7078
    Epoch 590/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4958 - accuracy: 0.7674 - val_loss: 0.6216 - val_accuracy: 0.7078
    Epoch 591/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4957 - accuracy: 0.7696 - val_loss: 0.6215 - val_accuracy: 0.7078
    Epoch 592/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4956 - accuracy: 0.7674 - val_loss: 0.6215 - val_accuracy: 0.7078
    Epoch 593/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4955 - accuracy: 0.7696 - val_loss: 0.6216 - val_accuracy: 0.7078
    Epoch 594/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4954 - accuracy: 0.7696 - val_loss: 0.6217 - val_accuracy: 0.7143
    Epoch 595/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4953 - accuracy: 0.7696 - val_loss: 0.6218 - val_accuracy: 0.7143
    Epoch 596/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4952 - accuracy: 0.7696 - val_loss: 0.6219 - val_accuracy: 0.7143
    Epoch 597/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4951 - accuracy: 0.7674 - val_loss: 0.6218 - val_accuracy: 0.7143
    Epoch 598/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4950 - accuracy: 0.7717 - val_loss: 0.6215 - val_accuracy: 0.7078
    Epoch 599/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4949 - accuracy: 0.7739 - val_loss: 0.6215 - val_accuracy: 0.7143
    Epoch 600/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4948 - accuracy: 0.7717 - val_loss: 0.6216 - val_accuracy: 0.7143
    Epoch 601/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4947 - accuracy: 0.7696 - val_loss: 0.6216 - val_accuracy: 0.7143
    Epoch 602/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4946 - accuracy: 0.7696 - val_loss: 0.6215 - val_accuracy: 0.7143
    Epoch 603/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4945 - accuracy: 0.7696 - val_loss: 0.6214 - val_accuracy: 0.7143
    Epoch 604/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4944 - accuracy: 0.7717 - val_loss: 0.6215 - val_accuracy: 0.7143
    Epoch 605/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4943 - accuracy: 0.7717 - val_loss: 0.6215 - val_accuracy: 0.7143
    Epoch 606/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4942 - accuracy: 0.7739 - val_loss: 0.6214 - val_accuracy: 0.7143
    Epoch 607/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4941 - accuracy: 0.7717 - val_loss: 0.6212 - val_accuracy: 0.7143
    Epoch 608/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4940 - accuracy: 0.7717 - val_loss: 0.6209 - val_accuracy: 0.7143
    Epoch 609/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4939 - accuracy: 0.7717 - val_loss: 0.6208 - val_accuracy: 0.7143
    Epoch 610/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4938 - accuracy: 0.7717 - val_loss: 0.6208 - val_accuracy: 0.7143
    Epoch 611/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4937 - accuracy: 0.7717 - val_loss: 0.6206 - val_accuracy: 0.7143
    Epoch 612/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4936 - accuracy: 0.7717 - val_loss: 0.6205 - val_accuracy: 0.7078
    Epoch 613/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4935 - accuracy: 0.7717 - val_loss: 0.6204 - val_accuracy: 0.7078
    Epoch 614/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4934 - accuracy: 0.7717 - val_loss: 0.6204 - val_accuracy: 0.7078
    Epoch 615/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4933 - accuracy: 0.7739 - val_loss: 0.6204 - val_accuracy: 0.7078
    Epoch 616/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4932 - accuracy: 0.7739 - val_loss: 0.6204 - val_accuracy: 0.7078
    Epoch 617/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4931 - accuracy: 0.7739 - val_loss: 0.6205 - val_accuracy: 0.7078
    Epoch 618/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4930 - accuracy: 0.7717 - val_loss: 0.6205 - val_accuracy: 0.7078
    Epoch 619/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4929 - accuracy: 0.7717 - val_loss: 0.6205 - val_accuracy: 0.7078
    Epoch 620/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4928 - accuracy: 0.7717 - val_loss: 0.6205 - val_accuracy: 0.7078
    Epoch 621/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4927 - accuracy: 0.7739 - val_loss: 0.6206 - val_accuracy: 0.7078
    Epoch 622/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4926 - accuracy: 0.7739 - val_loss: 0.6207 - val_accuracy: 0.7143
    Epoch 623/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4925 - accuracy: 0.7739 - val_loss: 0.6208 - val_accuracy: 0.7143
    Epoch 624/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4924 - accuracy: 0.7761 - val_loss: 0.6211 - val_accuracy: 0.7143
    Epoch 625/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4923 - accuracy: 0.7739 - val_loss: 0.6212 - val_accuracy: 0.7143
    Epoch 626/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4922 - accuracy: 0.7739 - val_loss: 0.6212 - val_accuracy: 0.7208
    Epoch 627/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4921 - accuracy: 0.7739 - val_loss: 0.6212 - val_accuracy: 0.7208
    Epoch 628/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4920 - accuracy: 0.7739 - val_loss: 0.6211 - val_accuracy: 0.7208
    Epoch 629/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4919 - accuracy: 0.7739 - val_loss: 0.6212 - val_accuracy: 0.7208
    Epoch 630/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4918 - accuracy: 0.7739 - val_loss: 0.6211 - val_accuracy: 0.7208
    Epoch 631/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4917 - accuracy: 0.7739 - val_loss: 0.6208 - val_accuracy: 0.7208
    Epoch 632/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4916 - accuracy: 0.7739 - val_loss: 0.6207 - val_accuracy: 0.7143
    Epoch 633/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4915 - accuracy: 0.7739 - val_loss: 0.6206 - val_accuracy: 0.7143
    Epoch 634/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4914 - accuracy: 0.7761 - val_loss: 0.6208 - val_accuracy: 0.7143
    Epoch 635/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4913 - accuracy: 0.7739 - val_loss: 0.6209 - val_accuracy: 0.7143
    Epoch 636/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4912 - accuracy: 0.7739 - val_loss: 0.6208 - val_accuracy: 0.7143
    Epoch 637/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4911 - accuracy: 0.7739 - val_loss: 0.6206 - val_accuracy: 0.7143
    Epoch 638/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4910 - accuracy: 0.7739 - val_loss: 0.6204 - val_accuracy: 0.7143
    Epoch 639/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4909 - accuracy: 0.7739 - val_loss: 0.6203 - val_accuracy: 0.7143
    Epoch 640/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4908 - accuracy: 0.7761 - val_loss: 0.6202 - val_accuracy: 0.7143
    Epoch 641/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4907 - accuracy: 0.7761 - val_loss: 0.6201 - val_accuracy: 0.7143
    Epoch 642/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4906 - accuracy: 0.7761 - val_loss: 0.6201 - val_accuracy: 0.7143
    Epoch 643/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4905 - accuracy: 0.7761 - val_loss: 0.6202 - val_accuracy: 0.7143
    Epoch 644/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4904 - accuracy: 0.7761 - val_loss: 0.6203 - val_accuracy: 0.7143
    Epoch 645/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4903 - accuracy: 0.7761 - val_loss: 0.6203 - val_accuracy: 0.7143
    Epoch 646/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4902 - accuracy: 0.7761 - val_loss: 0.6203 - val_accuracy: 0.7143
    Epoch 647/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4901 - accuracy: 0.7761 - val_loss: 0.6203 - val_accuracy: 0.7143
    Epoch 648/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4900 - accuracy: 0.7739 - val_loss: 0.6204 - val_accuracy: 0.7208
    Epoch 649/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4899 - accuracy: 0.7739 - val_loss: 0.6205 - val_accuracy: 0.7208
    Epoch 650/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4898 - accuracy: 0.7717 - val_loss: 0.6205 - val_accuracy: 0.7208
    Epoch 651/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4897 - accuracy: 0.7739 - val_loss: 0.6204 - val_accuracy: 0.7143
    Epoch 652/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.4896 - accuracy: 0.7761 - val_loss: 0.6202 - val_accuracy: 0.7143
    Epoch 653/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4895 - accuracy: 0.7761 - val_loss: 0.6201 - val_accuracy: 0.7143
    Epoch 654/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4894 - accuracy: 0.7739 - val_loss: 0.6200 - val_accuracy: 0.7208
    Epoch 655/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4893 - accuracy: 0.7739 - val_loss: 0.6199 - val_accuracy: 0.7208
    Epoch 656/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4892 - accuracy: 0.7739 - val_loss: 0.6199 - val_accuracy: 0.7143
    Epoch 657/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4891 - accuracy: 0.7739 - val_loss: 0.6198 - val_accuracy: 0.7143
    Epoch 658/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4890 - accuracy: 0.7761 - val_loss: 0.6199 - val_accuracy: 0.7143
    Epoch 659/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4889 - accuracy: 0.7761 - val_loss: 0.6198 - val_accuracy: 0.7208
    Epoch 660/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4888 - accuracy: 0.7783 - val_loss: 0.6194 - val_accuracy: 0.7143
    Epoch 661/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4887 - accuracy: 0.7783 - val_loss: 0.6193 - val_accuracy: 0.7143
    Epoch 662/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4887 - accuracy: 0.7761 - val_loss: 0.6193 - val_accuracy: 0.7143
    Epoch 663/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4886 - accuracy: 0.7761 - val_loss: 0.6195 - val_accuracy: 0.7208
    Epoch 664/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4885 - accuracy: 0.7783 - val_loss: 0.6194 - val_accuracy: 0.7143
    Epoch 665/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4884 - accuracy: 0.7761 - val_loss: 0.6194 - val_accuracy: 0.7143
    Epoch 666/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4882 - accuracy: 0.7783 - val_loss: 0.6194 - val_accuracy: 0.7143
    Epoch 667/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4882 - accuracy: 0.7761 - val_loss: 0.6193 - val_accuracy: 0.7143
    Epoch 668/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4881 - accuracy: 0.7761 - val_loss: 0.6195 - val_accuracy: 0.7208
    Epoch 669/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4880 - accuracy: 0.7804 - val_loss: 0.6195 - val_accuracy: 0.7208
    Epoch 670/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4879 - accuracy: 0.7804 - val_loss: 0.6194 - val_accuracy: 0.7208
    Epoch 671/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4878 - accuracy: 0.7804 - val_loss: 0.6191 - val_accuracy: 0.7143
    Epoch 672/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4876 - accuracy: 0.7783 - val_loss: 0.6190 - val_accuracy: 0.7143
    Epoch 673/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4875 - accuracy: 0.7783 - val_loss: 0.6191 - val_accuracy: 0.7208
    Epoch 674/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4874 - accuracy: 0.7804 - val_loss: 0.6190 - val_accuracy: 0.7208
    Epoch 675/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4873 - accuracy: 0.7804 - val_loss: 0.6187 - val_accuracy: 0.7143
    Epoch 676/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4872 - accuracy: 0.7804 - val_loss: 0.6184 - val_accuracy: 0.7143
    Epoch 677/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4871 - accuracy: 0.7783 - val_loss: 0.6184 - val_accuracy: 0.7143
    Epoch 678/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4870 - accuracy: 0.7804 - val_loss: 0.6183 - val_accuracy: 0.7208
    Epoch 679/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.4870 - accuracy: 0.7804 - val_loss: 0.6181 - val_accuracy: 0.7143
    Epoch 680/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4868 - accuracy: 0.7783 - val_loss: 0.6180 - val_accuracy: 0.7143
    Epoch 681/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4868 - accuracy: 0.7783 - val_loss: 0.6181 - val_accuracy: 0.7143
    Epoch 682/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4866 - accuracy: 0.7804 - val_loss: 0.6183 - val_accuracy: 0.7208
    Epoch 683/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4866 - accuracy: 0.7804 - val_loss: 0.6184 - val_accuracy: 0.7208
    Epoch 684/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4865 - accuracy: 0.7804 - val_loss: 0.6183 - val_accuracy: 0.7143
    Epoch 685/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4864 - accuracy: 0.7804 - val_loss: 0.6183 - val_accuracy: 0.7143
    Epoch 686/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4863 - accuracy: 0.7804 - val_loss: 0.6185 - val_accuracy: 0.7208
    Epoch 687/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4862 - accuracy: 0.7804 - val_loss: 0.6184 - val_accuracy: 0.7208
    Epoch 688/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4861 - accuracy: 0.7804 - val_loss: 0.6183 - val_accuracy: 0.7143
    Epoch 689/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4860 - accuracy: 0.7804 - val_loss: 0.6182 - val_accuracy: 0.7143
    Epoch 690/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4859 - accuracy: 0.7826 - val_loss: 0.6183 - val_accuracy: 0.7208
    Epoch 691/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4858 - accuracy: 0.7826 - val_loss: 0.6182 - val_accuracy: 0.7208
    Epoch 692/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4857 - accuracy: 0.7826 - val_loss: 0.6179 - val_accuracy: 0.7143
    Epoch 693/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4856 - accuracy: 0.7826 - val_loss: 0.6178 - val_accuracy: 0.7078
    Epoch 694/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4855 - accuracy: 0.7826 - val_loss: 0.6180 - val_accuracy: 0.7143
    Epoch 695/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4854 - accuracy: 0.7826 - val_loss: 0.6179 - val_accuracy: 0.7143
    Epoch 696/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4853 - accuracy: 0.7826 - val_loss: 0.6176 - val_accuracy: 0.7078
    Epoch 697/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4852 - accuracy: 0.7826 - val_loss: 0.6172 - val_accuracy: 0.7078
    Epoch 698/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4851 - accuracy: 0.7804 - val_loss: 0.6170 - val_accuracy: 0.7078
    Epoch 699/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4850 - accuracy: 0.7826 - val_loss: 0.6172 - val_accuracy: 0.7143
    Epoch 700/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4849 - accuracy: 0.7826 - val_loss: 0.6172 - val_accuracy: 0.7143
    Epoch 701/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4848 - accuracy: 0.7826 - val_loss: 0.6172 - val_accuracy: 0.7208
    Epoch 702/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4846 - accuracy: 0.7826 - val_loss: 0.6171 - val_accuracy: 0.7143
    Epoch 703/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4845 - accuracy: 0.7848 - val_loss: 0.6171 - val_accuracy: 0.7143
    Epoch 704/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4844 - accuracy: 0.7848 - val_loss: 0.6174 - val_accuracy: 0.7143
    Epoch 705/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4843 - accuracy: 0.7848 - val_loss: 0.6178 - val_accuracy: 0.7208
    Epoch 706/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4842 - accuracy: 0.7848 - val_loss: 0.6183 - val_accuracy: 0.7208
    Epoch 707/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4841 - accuracy: 0.7826 - val_loss: 0.6186 - val_accuracy: 0.7208
    Epoch 708/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4840 - accuracy: 0.7826 - val_loss: 0.6185 - val_accuracy: 0.7208
    Epoch 709/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4839 - accuracy: 0.7826 - val_loss: 0.6181 - val_accuracy: 0.7208
    Epoch 710/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4838 - accuracy: 0.7826 - val_loss: 0.6179 - val_accuracy: 0.7208
    Epoch 711/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4837 - accuracy: 0.7848 - val_loss: 0.6180 - val_accuracy: 0.7208
    Epoch 712/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4836 - accuracy: 0.7826 - val_loss: 0.6180 - val_accuracy: 0.7208
    Epoch 713/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4835 - accuracy: 0.7848 - val_loss: 0.6179 - val_accuracy: 0.7208
    Epoch 714/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4834 - accuracy: 0.7848 - val_loss: 0.6177 - val_accuracy: 0.7208
    Epoch 715/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4833 - accuracy: 0.7848 - val_loss: 0.6175 - val_accuracy: 0.7208
    Epoch 716/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4833 - accuracy: 0.7848 - val_loss: 0.6171 - val_accuracy: 0.7208
    Epoch 717/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4832 - accuracy: 0.7848 - val_loss: 0.6169 - val_accuracy: 0.7143
    Epoch 718/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4831 - accuracy: 0.7870 - val_loss: 0.6170 - val_accuracy: 0.7143
    Epoch 719/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4830 - accuracy: 0.7870 - val_loss: 0.6173 - val_accuracy: 0.7208
    Epoch 720/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4829 - accuracy: 0.7848 - val_loss: 0.6178 - val_accuracy: 0.7208
    Epoch 721/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4828 - accuracy: 0.7826 - val_loss: 0.6179 - val_accuracy: 0.7208
    Epoch 722/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4827 - accuracy: 0.7848 - val_loss: 0.6176 - val_accuracy: 0.7208
    Epoch 723/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4826 - accuracy: 0.7848 - val_loss: 0.6174 - val_accuracy: 0.7143
    Epoch 724/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4825 - accuracy: 0.7891 - val_loss: 0.6176 - val_accuracy: 0.7208
    Epoch 725/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4824 - accuracy: 0.7891 - val_loss: 0.6177 - val_accuracy: 0.7208
    Epoch 726/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4823 - accuracy: 0.7848 - val_loss: 0.6176 - val_accuracy: 0.7208
    Epoch 727/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4822 - accuracy: 0.7848 - val_loss: 0.6173 - val_accuracy: 0.7208
    Epoch 728/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4822 - accuracy: 0.7891 - val_loss: 0.6173 - val_accuracy: 0.7273
    Epoch 729/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4821 - accuracy: 0.7891 - val_loss: 0.6175 - val_accuracy: 0.7273
    Epoch 730/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4820 - accuracy: 0.7848 - val_loss: 0.6176 - val_accuracy: 0.7273
    Epoch 731/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4819 - accuracy: 0.7870 - val_loss: 0.6176 - val_accuracy: 0.7208
    Epoch 732/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4818 - accuracy: 0.7891 - val_loss: 0.6175 - val_accuracy: 0.7273
    Epoch 733/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4818 - accuracy: 0.7891 - val_loss: 0.6175 - val_accuracy: 0.7273
    Epoch 734/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4816 - accuracy: 0.7891 - val_loss: 0.6177 - val_accuracy: 0.7273
    Epoch 735/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.4816 - accuracy: 0.7870 - val_loss: 0.6177 - val_accuracy: 0.7273
    Epoch 736/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4815 - accuracy: 0.7870 - val_loss: 0.6176 - val_accuracy: 0.7273
    Epoch 737/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4814 - accuracy: 0.7870 - val_loss: 0.6175 - val_accuracy: 0.7208
    Epoch 738/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4813 - accuracy: 0.7891 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 739/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4812 - accuracy: 0.7891 - val_loss: 0.6173 - val_accuracy: 0.7208
    Epoch 740/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4811 - accuracy: 0.7891 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 741/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4810 - accuracy: 0.7891 - val_loss: 0.6177 - val_accuracy: 0.7208
    Epoch 742/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4810 - accuracy: 0.7870 - val_loss: 0.6178 - val_accuracy: 0.7208
    Epoch 743/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4809 - accuracy: 0.7848 - val_loss: 0.6176 - val_accuracy: 0.7208
    Epoch 744/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4808 - accuracy: 0.7891 - val_loss: 0.6175 - val_accuracy: 0.7208
    Epoch 745/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4807 - accuracy: 0.7891 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 746/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4806 - accuracy: 0.7848 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 747/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4805 - accuracy: 0.7848 - val_loss: 0.6175 - val_accuracy: 0.7208
    Epoch 748/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4804 - accuracy: 0.7870 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 749/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4803 - accuracy: 0.7891 - val_loss: 0.6173 - val_accuracy: 0.7208
    Epoch 750/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4803 - accuracy: 0.7891 - val_loss: 0.6172 - val_accuracy: 0.7208
    Epoch 751/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4802 - accuracy: 0.7870 - val_loss: 0.6172 - val_accuracy: 0.7208
    Epoch 752/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4801 - accuracy: 0.7848 - val_loss: 0.6169 - val_accuracy: 0.7208
    Epoch 753/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4800 - accuracy: 0.7891 - val_loss: 0.6166 - val_accuracy: 0.7208
    Epoch 754/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4799 - accuracy: 0.7870 - val_loss: 0.6167 - val_accuracy: 0.7208
    Epoch 755/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4798 - accuracy: 0.7870 - val_loss: 0.6170 - val_accuracy: 0.7208
    Epoch 756/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4797 - accuracy: 0.7848 - val_loss: 0.6171 - val_accuracy: 0.7208
    Epoch 757/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4797 - accuracy: 0.7870 - val_loss: 0.6172 - val_accuracy: 0.7208
    Epoch 758/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4796 - accuracy: 0.7870 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 759/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4795 - accuracy: 0.7848 - val_loss: 0.6174 - val_accuracy: 0.7208
    Epoch 760/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4794 - accuracy: 0.7848 - val_loss: 0.6171 - val_accuracy: 0.7208
    Epoch 761/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4793 - accuracy: 0.7848 - val_loss: 0.6170 - val_accuracy: 0.7208
    Epoch 762/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4792 - accuracy: 0.7870 - val_loss: 0.6171 - val_accuracy: 0.7208
    Epoch 763/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4792 - accuracy: 0.7870 - val_loss: 0.6172 - val_accuracy: 0.7208
    Epoch 764/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4791 - accuracy: 0.7826 - val_loss: 0.6173 - val_accuracy: 0.7208
    Epoch 765/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4790 - accuracy: 0.7826 - val_loss: 0.6170 - val_accuracy: 0.7208
    Epoch 766/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4789 - accuracy: 0.7826 - val_loss: 0.6166 - val_accuracy: 0.7208
    Epoch 767/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4788 - accuracy: 0.7870 - val_loss: 0.6167 - val_accuracy: 0.7208
    Epoch 768/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4787 - accuracy: 0.7870 - val_loss: 0.6168 - val_accuracy: 0.7208
    Epoch 769/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4787 - accuracy: 0.7870 - val_loss: 0.6169 - val_accuracy: 0.7208
    Epoch 770/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4786 - accuracy: 0.7870 - val_loss: 0.6168 - val_accuracy: 0.7208
    Epoch 771/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4785 - accuracy: 0.7848 - val_loss: 0.6168 - val_accuracy: 0.7208
    Epoch 772/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4784 - accuracy: 0.7826 - val_loss: 0.6166 - val_accuracy: 0.7208
    Epoch 773/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4784 - accuracy: 0.7848 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 774/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4782 - accuracy: 0.7848 - val_loss: 0.6160 - val_accuracy: 0.7143
    Epoch 775/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4782 - accuracy: 0.7891 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 776/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4781 - accuracy: 0.7848 - val_loss: 0.6165 - val_accuracy: 0.7208
    Epoch 777/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4780 - accuracy: 0.7826 - val_loss: 0.6165 - val_accuracy: 0.7208
    Epoch 778/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4779 - accuracy: 0.7870 - val_loss: 0.6164 - val_accuracy: 0.7208
    Epoch 779/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4779 - accuracy: 0.7913 - val_loss: 0.6165 - val_accuracy: 0.7143
    Epoch 780/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.4778 - accuracy: 0.7891 - val_loss: 0.6167 - val_accuracy: 0.7208
    Epoch 781/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4777 - accuracy: 0.7870 - val_loss: 0.6166 - val_accuracy: 0.7208
    Epoch 782/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4776 - accuracy: 0.7891 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 783/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4775 - accuracy: 0.7870 - val_loss: 0.6159 - val_accuracy: 0.7143
    Epoch 784/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4774 - accuracy: 0.7891 - val_loss: 0.6158 - val_accuracy: 0.7208
    Epoch 785/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4774 - accuracy: 0.7913 - val_loss: 0.6160 - val_accuracy: 0.7208
    Epoch 786/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4773 - accuracy: 0.7913 - val_loss: 0.6161 - val_accuracy: 0.7208
    Epoch 787/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4772 - accuracy: 0.7913 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 788/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4771 - accuracy: 0.7891 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 789/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4770 - accuracy: 0.7891 - val_loss: 0.6163 - val_accuracy: 0.7208
    Epoch 790/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4769 - accuracy: 0.7870 - val_loss: 0.6164 - val_accuracy: 0.7208
    Epoch 791/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4768 - accuracy: 0.7870 - val_loss: 0.6162 - val_accuracy: 0.7143
    Epoch 792/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4768 - accuracy: 0.7913 - val_loss: 0.6162 - val_accuracy: 0.7143
    Epoch 793/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4767 - accuracy: 0.7935 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 794/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4766 - accuracy: 0.7913 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 795/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4765 - accuracy: 0.7913 - val_loss: 0.6158 - val_accuracy: 0.7208
    Epoch 796/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4765 - accuracy: 0.7913 - val_loss: 0.6154 - val_accuracy: 0.7143
    Epoch 797/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4764 - accuracy: 0.7913 - val_loss: 0.6153 - val_accuracy: 0.7143
    Epoch 798/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4763 - accuracy: 0.7935 - val_loss: 0.6154 - val_accuracy: 0.7208
    Epoch 799/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4762 - accuracy: 0.7913 - val_loss: 0.6153 - val_accuracy: 0.7208
    Epoch 800/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4762 - accuracy: 0.7891 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 801/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4761 - accuracy: 0.7891 - val_loss: 0.6149 - val_accuracy: 0.7143
    Epoch 802/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4760 - accuracy: 0.7891 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 803/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4759 - accuracy: 0.7935 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 804/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4758 - accuracy: 0.7935 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 805/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4757 - accuracy: 0.7913 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 806/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4756 - accuracy: 0.7913 - val_loss: 0.6153 - val_accuracy: 0.7208
    Epoch 807/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4755 - accuracy: 0.7891 - val_loss: 0.6154 - val_accuracy: 0.7208
    Epoch 808/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4754 - accuracy: 0.7913 - val_loss: 0.6152 - val_accuracy: 0.7143
    Epoch 809/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4754 - accuracy: 0.7913 - val_loss: 0.6151 - val_accuracy: 0.7143
    Epoch 810/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4753 - accuracy: 0.7913 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 811/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4752 - accuracy: 0.7935 - val_loss: 0.6150 - val_accuracy: 0.7208
    Epoch 812/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4751 - accuracy: 0.7913 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 813/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4750 - accuracy: 0.7913 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 814/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4749 - accuracy: 0.7913 - val_loss: 0.6149 - val_accuracy: 0.7208
    Epoch 815/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4748 - accuracy: 0.7935 - val_loss: 0.6151 - val_accuracy: 0.7208
    Epoch 816/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4747 - accuracy: 0.7935 - val_loss: 0.6150 - val_accuracy: 0.7208
    Epoch 817/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.4746 - accuracy: 0.7935 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 818/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4745 - accuracy: 0.7913 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 819/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4744 - accuracy: 0.7913 - val_loss: 0.6148 - val_accuracy: 0.7208
    Epoch 820/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4743 - accuracy: 0.7935 - val_loss: 0.6149 - val_accuracy: 0.7208
    Epoch 821/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4742 - accuracy: 0.7935 - val_loss: 0.6149 - val_accuracy: 0.7208
    Epoch 822/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4741 - accuracy: 0.7935 - val_loss: 0.6148 - val_accuracy: 0.7208
    Epoch 823/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4740 - accuracy: 0.7935 - val_loss: 0.6148 - val_accuracy: 0.7208
    Epoch 824/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4739 - accuracy: 0.7935 - val_loss: 0.6145 - val_accuracy: 0.7143
    Epoch 825/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4738 - accuracy: 0.7957 - val_loss: 0.6144 - val_accuracy: 0.7143
    Epoch 826/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4737 - accuracy: 0.7935 - val_loss: 0.6145 - val_accuracy: 0.7143
    Epoch 827/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4736 - accuracy: 0.7935 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 828/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4736 - accuracy: 0.7957 - val_loss: 0.6147 - val_accuracy: 0.7143
    Epoch 829/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4734 - accuracy: 0.7957 - val_loss: 0.6144 - val_accuracy: 0.7078
    Epoch 830/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4733 - accuracy: 0.7935 - val_loss: 0.6141 - val_accuracy: 0.7078
    Epoch 831/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4732 - accuracy: 0.7957 - val_loss: 0.6140 - val_accuracy: 0.7078
    Epoch 832/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4731 - accuracy: 0.8000 - val_loss: 0.6137 - val_accuracy: 0.7078
    Epoch 833/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4729 - accuracy: 0.8000 - val_loss: 0.6133 - val_accuracy: 0.7078
    Epoch 834/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4728 - accuracy: 0.7978 - val_loss: 0.6131 - val_accuracy: 0.7078
    Epoch 835/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4727 - accuracy: 0.7978 - val_loss: 0.6133 - val_accuracy: 0.7143
    Epoch 836/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4726 - accuracy: 0.8000 - val_loss: 0.6135 - val_accuracy: 0.7078
    Epoch 837/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4724 - accuracy: 0.8000 - val_loss: 0.6138 - val_accuracy: 0.7078
    Epoch 838/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4723 - accuracy: 0.7978 - val_loss: 0.6140 - val_accuracy: 0.6948
    Epoch 839/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4722 - accuracy: 0.7978 - val_loss: 0.6144 - val_accuracy: 0.6948
    Epoch 840/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4721 - accuracy: 0.8000 - val_loss: 0.6147 - val_accuracy: 0.7013
    Epoch 841/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4719 - accuracy: 0.8000 - val_loss: 0.6148 - val_accuracy: 0.7078
    Epoch 842/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4718 - accuracy: 0.8000 - val_loss: 0.6146 - val_accuracy: 0.7013
    Epoch 843/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4717 - accuracy: 0.8000 - val_loss: 0.6142 - val_accuracy: 0.6948
    Epoch 844/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4716 - accuracy: 0.7978 - val_loss: 0.6140 - val_accuracy: 0.6948
    Epoch 845/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4714 - accuracy: 0.7978 - val_loss: 0.6141 - val_accuracy: 0.7013
    Epoch 846/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4713 - accuracy: 0.8000 - val_loss: 0.6143 - val_accuracy: 0.7013
    Epoch 847/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4712 - accuracy: 0.8022 - val_loss: 0.6143 - val_accuracy: 0.7013
    Epoch 848/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4711 - accuracy: 0.8000 - val_loss: 0.6144 - val_accuracy: 0.6948
    Epoch 849/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4710 - accuracy: 0.8000 - val_loss: 0.6146 - val_accuracy: 0.7013
    Epoch 850/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4708 - accuracy: 0.8000 - val_loss: 0.6147 - val_accuracy: 0.7013
    Epoch 851/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4707 - accuracy: 0.8022 - val_loss: 0.6147 - val_accuracy: 0.7013
    Epoch 852/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4706 - accuracy: 0.8022 - val_loss: 0.6145 - val_accuracy: 0.6948
    Epoch 853/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4705 - accuracy: 0.8022 - val_loss: 0.6145 - val_accuracy: 0.6948
    Epoch 854/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4704 - accuracy: 0.8022 - val_loss: 0.6146 - val_accuracy: 0.6948
    Epoch 855/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4703 - accuracy: 0.8022 - val_loss: 0.6148 - val_accuracy: 0.7013
    Epoch 856/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4701 - accuracy: 0.8022 - val_loss: 0.6148 - val_accuracy: 0.7078
    Epoch 857/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4700 - accuracy: 0.8022 - val_loss: 0.6147 - val_accuracy: 0.7013
    Epoch 858/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4699 - accuracy: 0.8022 - val_loss: 0.6144 - val_accuracy: 0.6948
    Epoch 859/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4698 - accuracy: 0.8022 - val_loss: 0.6142 - val_accuracy: 0.6948
    Epoch 860/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4697 - accuracy: 0.8022 - val_loss: 0.6141 - val_accuracy: 0.6948
    Epoch 861/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4696 - accuracy: 0.8022 - val_loss: 0.6140 - val_accuracy: 0.7013
    Epoch 862/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4695 - accuracy: 0.8022 - val_loss: 0.6137 - val_accuracy: 0.6948
    Epoch 863/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4693 - accuracy: 0.8022 - val_loss: 0.6136 - val_accuracy: 0.7013
    Epoch 864/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4692 - accuracy: 0.8022 - val_loss: 0.6135 - val_accuracy: 0.7013
    Epoch 865/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4691 - accuracy: 0.8022 - val_loss: 0.6134 - val_accuracy: 0.7013
    Epoch 866/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4690 - accuracy: 0.8022 - val_loss: 0.6132 - val_accuracy: 0.6883
    Epoch 867/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4689 - accuracy: 0.8000 - val_loss: 0.6133 - val_accuracy: 0.6883
    Epoch 868/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4688 - accuracy: 0.8000 - val_loss: 0.6136 - val_accuracy: 0.7013
    Epoch 869/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4687 - accuracy: 0.8022 - val_loss: 0.6137 - val_accuracy: 0.7013
    Epoch 870/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4686 - accuracy: 0.8022 - val_loss: 0.6134 - val_accuracy: 0.7013
    Epoch 871/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4685 - accuracy: 0.8022 - val_loss: 0.6131 - val_accuracy: 0.6883
    Epoch 872/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4684 - accuracy: 0.8000 - val_loss: 0.6130 - val_accuracy: 0.6948
    Epoch 873/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4683 - accuracy: 0.8022 - val_loss: 0.6129 - val_accuracy: 0.7013
    Epoch 874/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4682 - accuracy: 0.8022 - val_loss: 0.6127 - val_accuracy: 0.6948
    Epoch 875/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4681 - accuracy: 0.8022 - val_loss: 0.6126 - val_accuracy: 0.6948
    Epoch 876/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4680 - accuracy: 0.8022 - val_loss: 0.6127 - val_accuracy: 0.6948
    Epoch 877/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4679 - accuracy: 0.8022 - val_loss: 0.6128 - val_accuracy: 0.6948
    Epoch 878/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4678 - accuracy: 0.8022 - val_loss: 0.6129 - val_accuracy: 0.6948
    Epoch 879/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4677 - accuracy: 0.8022 - val_loss: 0.6130 - val_accuracy: 0.6948
    Epoch 880/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4676 - accuracy: 0.8022 - val_loss: 0.6133 - val_accuracy: 0.6948
    Epoch 881/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4675 - accuracy: 0.8022 - val_loss: 0.6135 - val_accuracy: 0.7013
    Epoch 882/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4674 - accuracy: 0.8022 - val_loss: 0.6134 - val_accuracy: 0.6948
    Epoch 883/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4673 - accuracy: 0.8022 - val_loss: 0.6131 - val_accuracy: 0.6948
    Epoch 884/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4672 - accuracy: 0.8022 - val_loss: 0.6129 - val_accuracy: 0.6948
    Epoch 885/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4671 - accuracy: 0.8022 - val_loss: 0.6127 - val_accuracy: 0.6948
    Epoch 886/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4670 - accuracy: 0.8022 - val_loss: 0.6126 - val_accuracy: 0.6948
    Epoch 887/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4669 - accuracy: 0.8022 - val_loss: 0.6125 - val_accuracy: 0.6948
    Epoch 888/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4668 - accuracy: 0.8022 - val_loss: 0.6127 - val_accuracy: 0.6948
    Epoch 889/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4667 - accuracy: 0.8022 - val_loss: 0.6128 - val_accuracy: 0.6948
    Epoch 890/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4666 - accuracy: 0.8022 - val_loss: 0.6128 - val_accuracy: 0.6948
    Epoch 891/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4666 - accuracy: 0.8022 - val_loss: 0.6123 - val_accuracy: 0.6948
    Epoch 892/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4664 - accuracy: 0.8022 - val_loss: 0.6117 - val_accuracy: 0.6883
    Epoch 893/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4664 - accuracy: 0.8022 - val_loss: 0.6116 - val_accuracy: 0.6948
    Epoch 894/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4663 - accuracy: 0.8022 - val_loss: 0.6118 - val_accuracy: 0.6948
    Epoch 895/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4662 - accuracy: 0.8022 - val_loss: 0.6117 - val_accuracy: 0.6948
    Epoch 896/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4661 - accuracy: 0.8022 - val_loss: 0.6117 - val_accuracy: 0.6948
    Epoch 897/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4660 - accuracy: 0.8022 - val_loss: 0.6116 - val_accuracy: 0.6948
    Epoch 898/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4659 - accuracy: 0.8022 - val_loss: 0.6116 - val_accuracy: 0.6948
    Epoch 899/2000
    1/1 [==============================] - 0s 89ms/step - loss: 0.4658 - accuracy: 0.8022 - val_loss: 0.6114 - val_accuracy: 0.6948
    Epoch 900/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4657 - accuracy: 0.8022 - val_loss: 0.6112 - val_accuracy: 0.6883
    Epoch 901/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4656 - accuracy: 0.8022 - val_loss: 0.6114 - val_accuracy: 0.6948
    Epoch 902/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4655 - accuracy: 0.8022 - val_loss: 0.6114 - val_accuracy: 0.6948
    Epoch 903/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4654 - accuracy: 0.8022 - val_loss: 0.6113 - val_accuracy: 0.6948
    Epoch 904/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4654 - accuracy: 0.8022 - val_loss: 0.6110 - val_accuracy: 0.6948
    Epoch 905/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4653 - accuracy: 0.8022 - val_loss: 0.6109 - val_accuracy: 0.6948
    Epoch 906/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4652 - accuracy: 0.8022 - val_loss: 0.6108 - val_accuracy: 0.6948
    Epoch 907/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4651 - accuracy: 0.8022 - val_loss: 0.6105 - val_accuracy: 0.6883
    Epoch 908/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4650 - accuracy: 0.8022 - val_loss: 0.6106 - val_accuracy: 0.6883
    Epoch 909/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4649 - accuracy: 0.8022 - val_loss: 0.6109 - val_accuracy: 0.6948
    Epoch 910/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4648 - accuracy: 0.8022 - val_loss: 0.6111 - val_accuracy: 0.6948
    Epoch 911/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4648 - accuracy: 0.8000 - val_loss: 0.6105 - val_accuracy: 0.6948
    Epoch 912/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4646 - accuracy: 0.8022 - val_loss: 0.6102 - val_accuracy: 0.6883
    Epoch 913/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4646 - accuracy: 0.8043 - val_loss: 0.6103 - val_accuracy: 0.6948
    Epoch 914/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4645 - accuracy: 0.8022 - val_loss: 0.6104 - val_accuracy: 0.6948
    Epoch 915/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4644 - accuracy: 0.8022 - val_loss: 0.6103 - val_accuracy: 0.6948
    Epoch 916/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.4643 - accuracy: 0.8043 - val_loss: 0.6104 - val_accuracy: 0.6948
    Epoch 917/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4642 - accuracy: 0.8043 - val_loss: 0.6106 - val_accuracy: 0.6948
    Epoch 918/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4641 - accuracy: 0.8022 - val_loss: 0.6105 - val_accuracy: 0.6948
    Epoch 919/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4640 - accuracy: 0.8022 - val_loss: 0.6100 - val_accuracy: 0.6883
    Epoch 920/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4640 - accuracy: 0.8043 - val_loss: 0.6098 - val_accuracy: 0.6883
    Epoch 921/2000
    1/1 [==============================] - 0s 72ms/step - loss: 0.4639 - accuracy: 0.8043 - val_loss: 0.6098 - val_accuracy: 0.6948
    Epoch 922/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4638 - accuracy: 0.8022 - val_loss: 0.6096 - val_accuracy: 0.6948
    Epoch 923/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4637 - accuracy: 0.8000 - val_loss: 0.6089 - val_accuracy: 0.6883
    Epoch 924/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4636 - accuracy: 0.8043 - val_loss: 0.6086 - val_accuracy: 0.6883
    Epoch 925/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4635 - accuracy: 0.8043 - val_loss: 0.6089 - val_accuracy: 0.6948
    Epoch 926/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4634 - accuracy: 0.8043 - val_loss: 0.6093 - val_accuracy: 0.6948
    Epoch 927/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4634 - accuracy: 0.8022 - val_loss: 0.6092 - val_accuracy: 0.6948
    Epoch 928/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4633 - accuracy: 0.8043 - val_loss: 0.6089 - val_accuracy: 0.6948
    Epoch 929/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4632 - accuracy: 0.8043 - val_loss: 0.6086 - val_accuracy: 0.6948
    Epoch 930/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4631 - accuracy: 0.8043 - val_loss: 0.6087 - val_accuracy: 0.6948
    Epoch 931/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4630 - accuracy: 0.8043 - val_loss: 0.6086 - val_accuracy: 0.6948
    Epoch 932/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4629 - accuracy: 0.8043 - val_loss: 0.6085 - val_accuracy: 0.6948
    Epoch 933/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4629 - accuracy: 0.8043 - val_loss: 0.6086 - val_accuracy: 0.6948
    Epoch 934/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4628 - accuracy: 0.8043 - val_loss: 0.6086 - val_accuracy: 0.6948
    Epoch 935/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4627 - accuracy: 0.8022 - val_loss: 0.6080 - val_accuracy: 0.6948
    Epoch 936/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4626 - accuracy: 0.8043 - val_loss: 0.6077 - val_accuracy: 0.6948
    Epoch 937/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4625 - accuracy: 0.8043 - val_loss: 0.6078 - val_accuracy: 0.6948
    Epoch 938/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4624 - accuracy: 0.8043 - val_loss: 0.6078 - val_accuracy: 0.7013
    Epoch 939/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4623 - accuracy: 0.8043 - val_loss: 0.6077 - val_accuracy: 0.7013
    Epoch 940/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4622 - accuracy: 0.8043 - val_loss: 0.6078 - val_accuracy: 0.7013
    Epoch 941/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4621 - accuracy: 0.8043 - val_loss: 0.6079 - val_accuracy: 0.7013
    Epoch 942/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4621 - accuracy: 0.8022 - val_loss: 0.6077 - val_accuracy: 0.6948
    Epoch 943/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4620 - accuracy: 0.8043 - val_loss: 0.6074 - val_accuracy: 0.6948
    Epoch 944/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4619 - accuracy: 0.8043 - val_loss: 0.6074 - val_accuracy: 0.6948
    Epoch 945/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4618 - accuracy: 0.8043 - val_loss: 0.6072 - val_accuracy: 0.6948
    Epoch 946/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4617 - accuracy: 0.8043 - val_loss: 0.6070 - val_accuracy: 0.7013
    Epoch 947/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4616 - accuracy: 0.8043 - val_loss: 0.6069 - val_accuracy: 0.7013
    Epoch 948/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4616 - accuracy: 0.8043 - val_loss: 0.6071 - val_accuracy: 0.7013
    Epoch 949/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4615 - accuracy: 0.8043 - val_loss: 0.6066 - val_accuracy: 0.7013
    Epoch 950/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4614 - accuracy: 0.8043 - val_loss: 0.6060 - val_accuracy: 0.7013
    Epoch 951/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4613 - accuracy: 0.8043 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 952/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4612 - accuracy: 0.8043 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 953/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4611 - accuracy: 0.8043 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 954/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4609 - accuracy: 0.8043 - val_loss: 0.6055 - val_accuracy: 0.7013
    Epoch 955/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4607 - accuracy: 0.8043 - val_loss: 0.6054 - val_accuracy: 0.7013
    Epoch 956/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4606 - accuracy: 0.8022 - val_loss: 0.6055 - val_accuracy: 0.7013
    Epoch 957/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.4605 - accuracy: 0.8022 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 958/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4603 - accuracy: 0.8043 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 959/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4602 - accuracy: 0.8065 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 960/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4601 - accuracy: 0.8065 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 961/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4600 - accuracy: 0.8065 - val_loss: 0.6062 - val_accuracy: 0.7013
    Epoch 962/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4599 - accuracy: 0.8043 - val_loss: 0.6069 - val_accuracy: 0.7013
    Epoch 963/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4598 - accuracy: 0.8022 - val_loss: 0.6073 - val_accuracy: 0.6948
    Epoch 964/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4597 - accuracy: 0.8022 - val_loss: 0.6072 - val_accuracy: 0.7013
    Epoch 965/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4596 - accuracy: 0.8022 - val_loss: 0.6071 - val_accuracy: 0.7013
    Epoch 966/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4595 - accuracy: 0.8043 - val_loss: 0.6071 - val_accuracy: 0.6948
    Epoch 967/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4594 - accuracy: 0.8043 - val_loss: 0.6074 - val_accuracy: 0.7013
    Epoch 968/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4593 - accuracy: 0.8043 - val_loss: 0.6077 - val_accuracy: 0.7013
    Epoch 969/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4592 - accuracy: 0.8043 - val_loss: 0.6076 - val_accuracy: 0.6948
    Epoch 970/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4591 - accuracy: 0.8043 - val_loss: 0.6070 - val_accuracy: 0.7013
    Epoch 971/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4590 - accuracy: 0.8043 - val_loss: 0.6065 - val_accuracy: 0.7013
    Epoch 972/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4589 - accuracy: 0.8043 - val_loss: 0.6061 - val_accuracy: 0.7013
    Epoch 973/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4588 - accuracy: 0.8043 - val_loss: 0.6055 - val_accuracy: 0.6948
    Epoch 974/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4587 - accuracy: 0.8065 - val_loss: 0.6053 - val_accuracy: 0.6883
    Epoch 975/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4586 - accuracy: 0.8043 - val_loss: 0.6057 - val_accuracy: 0.6883
    Epoch 976/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4585 - accuracy: 0.8043 - val_loss: 0.6064 - val_accuracy: 0.7013
    Epoch 977/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4584 - accuracy: 0.8022 - val_loss: 0.6067 - val_accuracy: 0.7013
    Epoch 978/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4584 - accuracy: 0.8022 - val_loss: 0.6063 - val_accuracy: 0.6948
    Epoch 979/2000
    1/1 [==============================] - 0s 72ms/step - loss: 0.4582 - accuracy: 0.8043 - val_loss: 0.6061 - val_accuracy: 0.6883
    Epoch 980/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4582 - accuracy: 0.8065 - val_loss: 0.6066 - val_accuracy: 0.6948
    Epoch 981/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4580 - accuracy: 0.8043 - val_loss: 0.6068 - val_accuracy: 0.7013
    Epoch 982/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4579 - accuracy: 0.8043 - val_loss: 0.6066 - val_accuracy: 0.6948
    Epoch 983/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4578 - accuracy: 0.8043 - val_loss: 0.6064 - val_accuracy: 0.7013
    Epoch 984/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4577 - accuracy: 0.8043 - val_loss: 0.6065 - val_accuracy: 0.7078
    Epoch 985/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4576 - accuracy: 0.8043 - val_loss: 0.6061 - val_accuracy: 0.7078
    Epoch 986/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4575 - accuracy: 0.8043 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 987/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4575 - accuracy: 0.8043 - val_loss: 0.6057 - val_accuracy: 0.7078
    Epoch 988/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4573 - accuracy: 0.8043 - val_loss: 0.6059 - val_accuracy: 0.7078
    Epoch 989/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4573 - accuracy: 0.8043 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 990/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4572 - accuracy: 0.8065 - val_loss: 0.6057 - val_accuracy: 0.7013
    Epoch 991/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4571 - accuracy: 0.8065 - val_loss: 0.6060 - val_accuracy: 0.7143
    Epoch 992/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4570 - accuracy: 0.8043 - val_loss: 0.6060 - val_accuracy: 0.7078
    Epoch 993/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.4569 - accuracy: 0.8043 - val_loss: 0.6057 - val_accuracy: 0.7013
    Epoch 994/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4569 - accuracy: 0.8065 - val_loss: 0.6058 - val_accuracy: 0.7078
    Epoch 995/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4567 - accuracy: 0.8065 - val_loss: 0.6061 - val_accuracy: 0.7143
    Epoch 996/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.4567 - accuracy: 0.8022 - val_loss: 0.6059 - val_accuracy: 0.7078
    Epoch 997/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4565 - accuracy: 0.8065 - val_loss: 0.6058 - val_accuracy: 0.7078
    Epoch 998/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4565 - accuracy: 0.8065 - val_loss: 0.6058 - val_accuracy: 0.7013
    Epoch 999/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4564 - accuracy: 0.8065 - val_loss: 0.6055 - val_accuracy: 0.7013
    Epoch 1000/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4563 - accuracy: 0.8065 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 1001/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4562 - accuracy: 0.8065 - val_loss: 0.6056 - val_accuracy: 0.7013
    Epoch 1002/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4561 - accuracy: 0.8065 - val_loss: 0.6053 - val_accuracy: 0.7013
    Epoch 1003/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4560 - accuracy: 0.8065 - val_loss: 0.6055 - val_accuracy: 0.7013
    Epoch 1004/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4559 - accuracy: 0.8065 - val_loss: 0.6057 - val_accuracy: 0.7143
    Epoch 1005/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4559 - accuracy: 0.8022 - val_loss: 0.6053 - val_accuracy: 0.7013
    Epoch 1006/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4558 - accuracy: 0.8065 - val_loss: 0.6049 - val_accuracy: 0.7013
    Epoch 1007/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4557 - accuracy: 0.8065 - val_loss: 0.6052 - val_accuracy: 0.7013
    Epoch 1008/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4556 - accuracy: 0.8065 - val_loss: 0.6057 - val_accuracy: 0.7143
    Epoch 1009/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4555 - accuracy: 0.8043 - val_loss: 0.6053 - val_accuracy: 0.7078
    Epoch 1010/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4554 - accuracy: 0.8043 - val_loss: 0.6048 - val_accuracy: 0.6948
    Epoch 1011/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4553 - accuracy: 0.8065 - val_loss: 0.6046 - val_accuracy: 0.6948
    Epoch 1012/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4552 - accuracy: 0.8065 - val_loss: 0.6047 - val_accuracy: 0.7013
    Epoch 1013/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4551 - accuracy: 0.8065 - val_loss: 0.6047 - val_accuracy: 0.7013
    Epoch 1014/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4550 - accuracy: 0.8065 - val_loss: 0.6048 - val_accuracy: 0.7013
    Epoch 1015/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4549 - accuracy: 0.8065 - val_loss: 0.6052 - val_accuracy: 0.7078
    Epoch 1016/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4548 - accuracy: 0.8043 - val_loss: 0.6050 - val_accuracy: 0.7013
    Epoch 1017/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4547 - accuracy: 0.8087 - val_loss: 0.6049 - val_accuracy: 0.7013
    Epoch 1018/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4546 - accuracy: 0.8065 - val_loss: 0.6047 - val_accuracy: 0.6948
    Epoch 1019/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4545 - accuracy: 0.8065 - val_loss: 0.6045 - val_accuracy: 0.7013
    Epoch 1020/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4545 - accuracy: 0.8065 - val_loss: 0.6044 - val_accuracy: 0.7013
    Epoch 1021/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4544 - accuracy: 0.8065 - val_loss: 0.6047 - val_accuracy: 0.7078
    Epoch 1022/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4543 - accuracy: 0.8043 - val_loss: 0.6043 - val_accuracy: 0.7013
    Epoch 1023/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4542 - accuracy: 0.8065 - val_loss: 0.6040 - val_accuracy: 0.7013
    Epoch 1024/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4541 - accuracy: 0.8065 - val_loss: 0.6041 - val_accuracy: 0.7013
    Epoch 1025/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4540 - accuracy: 0.8043 - val_loss: 0.6042 - val_accuracy: 0.6948
    Epoch 1026/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4539 - accuracy: 0.8043 - val_loss: 0.6038 - val_accuracy: 0.7013
    Epoch 1027/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4538 - accuracy: 0.8087 - val_loss: 0.6040 - val_accuracy: 0.7013
    Epoch 1028/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4537 - accuracy: 0.8043 - val_loss: 0.6043 - val_accuracy: 0.7143
    Epoch 1029/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4537 - accuracy: 0.8043 - val_loss: 0.6037 - val_accuracy: 0.7013
    Epoch 1030/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4536 - accuracy: 0.8065 - val_loss: 0.6034 - val_accuracy: 0.7013
    Epoch 1031/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4535 - accuracy: 0.8065 - val_loss: 0.6037 - val_accuracy: 0.7013
    Epoch 1032/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4534 - accuracy: 0.8043 - val_loss: 0.6040 - val_accuracy: 0.7143
    Epoch 1033/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4533 - accuracy: 0.8043 - val_loss: 0.6034 - val_accuracy: 0.7013
    Epoch 1034/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4532 - accuracy: 0.8043 - val_loss: 0.6030 - val_accuracy: 0.7013
    Epoch 1035/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4531 - accuracy: 0.8065 - val_loss: 0.6032 - val_accuracy: 0.7013
    Epoch 1036/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4530 - accuracy: 0.8043 - val_loss: 0.6030 - val_accuracy: 0.7013
    Epoch 1037/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4529 - accuracy: 0.8043 - val_loss: 0.6030 - val_accuracy: 0.6948
    Epoch 1038/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4528 - accuracy: 0.8043 - val_loss: 0.6033 - val_accuracy: 0.6948
    Epoch 1039/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4527 - accuracy: 0.8043 - val_loss: 0.6036 - val_accuracy: 0.6948
    Epoch 1040/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4527 - accuracy: 0.8043 - val_loss: 0.6032 - val_accuracy: 0.6948
    Epoch 1041/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4526 - accuracy: 0.8087 - val_loss: 0.6032 - val_accuracy: 0.6948
    Epoch 1042/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4525 - accuracy: 0.8065 - val_loss: 0.6035 - val_accuracy: 0.7143
    Epoch 1043/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4524 - accuracy: 0.8043 - val_loss: 0.6029 - val_accuracy: 0.7078
    Epoch 1044/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4523 - accuracy: 0.8043 - val_loss: 0.6024 - val_accuracy: 0.7078
    Epoch 1045/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.4522 - accuracy: 0.8087 - val_loss: 0.6025 - val_accuracy: 0.7078
    Epoch 1046/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4521 - accuracy: 0.8043 - val_loss: 0.6024 - val_accuracy: 0.7078
    Epoch 1047/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4520 - accuracy: 0.8043 - val_loss: 0.6018 - val_accuracy: 0.7078
    Epoch 1048/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4519 - accuracy: 0.8065 - val_loss: 0.6018 - val_accuracy: 0.7078
    Epoch 1049/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4518 - accuracy: 0.8065 - val_loss: 0.6020 - val_accuracy: 0.7078
    Epoch 1050/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4517 - accuracy: 0.8043 - val_loss: 0.6019 - val_accuracy: 0.7078
    Epoch 1051/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4516 - accuracy: 0.8065 - val_loss: 0.6016 - val_accuracy: 0.7078
    Epoch 1052/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4515 - accuracy: 0.8065 - val_loss: 0.6015 - val_accuracy: 0.7078
    Epoch 1053/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4515 - accuracy: 0.8065 - val_loss: 0.6016 - val_accuracy: 0.7078
    Epoch 1054/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4514 - accuracy: 0.8043 - val_loss: 0.6014 - val_accuracy: 0.7078
    Epoch 1055/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4513 - accuracy: 0.8043 - val_loss: 0.6013 - val_accuracy: 0.7078
    Epoch 1056/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4512 - accuracy: 0.8043 - val_loss: 0.6014 - val_accuracy: 0.7078
    Epoch 1057/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4511 - accuracy: 0.8043 - val_loss: 0.6014 - val_accuracy: 0.7078
    Epoch 1058/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4510 - accuracy: 0.8043 - val_loss: 0.6014 - val_accuracy: 0.7078
    Epoch 1059/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4509 - accuracy: 0.8043 - val_loss: 0.6013 - val_accuracy: 0.7078
    Epoch 1060/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4509 - accuracy: 0.8043 - val_loss: 0.6013 - val_accuracy: 0.7078
    Epoch 1061/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.4507 - accuracy: 0.8043 - val_loss: 0.6010 - val_accuracy: 0.7078
    Epoch 1062/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4507 - accuracy: 0.8043 - val_loss: 0.6007 - val_accuracy: 0.7013
    Epoch 1063/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4506 - accuracy: 0.8043 - val_loss: 0.6002 - val_accuracy: 0.7013
    Epoch 1064/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4505 - accuracy: 0.8043 - val_loss: 0.6004 - val_accuracy: 0.7078
    Epoch 1065/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4504 - accuracy: 0.8043 - val_loss: 0.6003 - val_accuracy: 0.7013
    Epoch 1066/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4503 - accuracy: 0.8043 - val_loss: 0.6001 - val_accuracy: 0.7078
    Epoch 1067/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4503 - accuracy: 0.8065 - val_loss: 0.6004 - val_accuracy: 0.7078
    Epoch 1068/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4502 - accuracy: 0.8043 - val_loss: 0.6006 - val_accuracy: 0.7078
    Epoch 1069/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4501 - accuracy: 0.8022 - val_loss: 0.6001 - val_accuracy: 0.7078
    Epoch 1070/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4500 - accuracy: 0.8022 - val_loss: 0.6000 - val_accuracy: 0.7078
    Epoch 1071/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4500 - accuracy: 0.8022 - val_loss: 0.6005 - val_accuracy: 0.7078
    Epoch 1072/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.4499 - accuracy: 0.8022 - val_loss: 0.6002 - val_accuracy: 0.7078
    Epoch 1073/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.4498 - accuracy: 0.8022 - val_loss: 0.5997 - val_accuracy: 0.7078
    Epoch 1074/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4497 - accuracy: 0.8043 - val_loss: 0.5998 - val_accuracy: 0.7078
    Epoch 1075/2000
    1/1 [==============================] - 0s 91ms/step - loss: 0.4496 - accuracy: 0.8043 - val_loss: 0.6000 - val_accuracy: 0.7078
    Epoch 1076/2000
    1/1 [==============================] - 0s 82ms/step - loss: 0.4495 - accuracy: 0.8022 - val_loss: 0.5997 - val_accuracy: 0.7078
    Epoch 1077/2000
    1/1 [==============================] - 0s 84ms/step - loss: 0.4494 - accuracy: 0.8000 - val_loss: 0.5997 - val_accuracy: 0.7013
    Epoch 1078/2000
    1/1 [==============================] - 0s 72ms/step - loss: 0.4493 - accuracy: 0.8022 - val_loss: 0.6001 - val_accuracy: 0.7013
    Epoch 1079/2000
    1/1 [==============================] - 0s 82ms/step - loss: 0.4493 - accuracy: 0.8022 - val_loss: 0.5999 - val_accuracy: 0.7078
    Epoch 1080/2000
    1/1 [==============================] - 0s 81ms/step - loss: 0.4492 - accuracy: 0.8043 - val_loss: 0.6002 - val_accuracy: 0.7078
    Epoch 1081/2000
    1/1 [==============================] - 0s 75ms/step - loss: 0.4491 - accuracy: 0.8043 - val_loss: 0.6002 - val_accuracy: 0.7078
    Epoch 1082/2000
    1/1 [==============================] - 0s 76ms/step - loss: 0.4490 - accuracy: 0.8043 - val_loss: 0.5997 - val_accuracy: 0.7078
    Epoch 1083/2000
    1/1 [==============================] - 0s 86ms/step - loss: 0.4490 - accuracy: 0.8043 - val_loss: 0.5998 - val_accuracy: 0.7078
    Epoch 1084/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4489 - accuracy: 0.8043 - val_loss: 0.5997 - val_accuracy: 0.7078
    Epoch 1085/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4488 - accuracy: 0.8022 - val_loss: 0.5994 - val_accuracy: 0.7013
    Epoch 1086/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4487 - accuracy: 0.8022 - val_loss: 0.5991 - val_accuracy: 0.7078
    Epoch 1087/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.4486 - accuracy: 0.8022 - val_loss: 0.5994 - val_accuracy: 0.7078
    Epoch 1088/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4485 - accuracy: 0.8022 - val_loss: 0.5994 - val_accuracy: 0.7078
    Epoch 1089/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4485 - accuracy: 0.8022 - val_loss: 0.5992 - val_accuracy: 0.7078
    Epoch 1090/2000
    1/1 [==============================] - 0s 87ms/step - loss: 0.4484 - accuracy: 0.8043 - val_loss: 0.5994 - val_accuracy: 0.7078
    Epoch 1091/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4483 - accuracy: 0.8043 - val_loss: 0.5995 - val_accuracy: 0.7078
    Epoch 1092/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4483 - accuracy: 0.8022 - val_loss: 0.5985 - val_accuracy: 0.7013
    Epoch 1093/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4482 - accuracy: 0.8043 - val_loss: 0.5984 - val_accuracy: 0.7013
    Epoch 1094/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4481 - accuracy: 0.8065 - val_loss: 0.5990 - val_accuracy: 0.7013
    Epoch 1095/2000
    1/1 [==============================] - 0s 87ms/step - loss: 0.4480 - accuracy: 0.8043 - val_loss: 0.5987 - val_accuracy: 0.7013
    Epoch 1096/2000
    1/1 [==============================] - 0s 78ms/step - loss: 0.4479 - accuracy: 0.8043 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1097/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4479 - accuracy: 0.8043 - val_loss: 0.5988 - val_accuracy: 0.7013
    Epoch 1098/2000
    1/1 [==============================] - 0s 79ms/step - loss: 0.4478 - accuracy: 0.8043 - val_loss: 0.5997 - val_accuracy: 0.7013
    Epoch 1099/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.4477 - accuracy: 0.8043 - val_loss: 0.5993 - val_accuracy: 0.7078
    Epoch 1100/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4476 - accuracy: 0.8043 - val_loss: 0.5989 - val_accuracy: 0.7078
    Epoch 1101/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4475 - accuracy: 0.8043 - val_loss: 0.5991 - val_accuracy: 0.7013
    Epoch 1102/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.4474 - accuracy: 0.8043 - val_loss: 0.5987 - val_accuracy: 0.7078
    Epoch 1103/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4473 - accuracy: 0.8043 - val_loss: 0.5985 - val_accuracy: 0.7078
    Epoch 1104/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4473 - accuracy: 0.8065 - val_loss: 0.5985 - val_accuracy: 0.7013
    Epoch 1105/2000
    1/1 [==============================] - 0s 91ms/step - loss: 0.4472 - accuracy: 0.8065 - val_loss: 0.5982 - val_accuracy: 0.7013
    Epoch 1106/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4471 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7013
    Epoch 1107/2000
    1/1 [==============================] - 0s 100ms/step - loss: 0.4470 - accuracy: 0.8065 - val_loss: 0.5985 - val_accuracy: 0.7013
    Epoch 1108/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.4469 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1109/2000
    1/1 [==============================] - 0s 91ms/step - loss: 0.4468 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1110/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4468 - accuracy: 0.8065 - val_loss: 0.5984 - val_accuracy: 0.7013
    Epoch 1111/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4467 - accuracy: 0.8065 - val_loss: 0.5984 - val_accuracy: 0.7013
    Epoch 1112/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4466 - accuracy: 0.8065 - val_loss: 0.5982 - val_accuracy: 0.7013
    Epoch 1113/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4465 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1114/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4464 - accuracy: 0.8065 - val_loss: 0.5977 - val_accuracy: 0.7013
    Epoch 1115/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4464 - accuracy: 0.8087 - val_loss: 0.5979 - val_accuracy: 0.7013
    Epoch 1116/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4463 - accuracy: 0.8065 - val_loss: 0.5982 - val_accuracy: 0.7013
    Epoch 1117/2000
    1/1 [==============================] - 0s 73ms/step - loss: 0.4462 - accuracy: 0.8065 - val_loss: 0.5980 - val_accuracy: 0.7013
    Epoch 1118/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4461 - accuracy: 0.8065 - val_loss: 0.5979 - val_accuracy: 0.7013
    Epoch 1119/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4460 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1120/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4459 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7078
    Epoch 1121/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4459 - accuracy: 0.8043 - val_loss: 0.5986 - val_accuracy: 0.7078
    Epoch 1122/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4458 - accuracy: 0.8043 - val_loss: 0.5984 - val_accuracy: 0.7013
    Epoch 1123/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4457 - accuracy: 0.8065 - val_loss: 0.5983 - val_accuracy: 0.7013
    Epoch 1124/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4456 - accuracy: 0.8087 - val_loss: 0.5980 - val_accuracy: 0.7013
    Epoch 1125/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4455 - accuracy: 0.8087 - val_loss: 0.5975 - val_accuracy: 0.7013
    Epoch 1126/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4455 - accuracy: 0.8087 - val_loss: 0.5978 - val_accuracy: 0.7013
    Epoch 1127/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4454 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7013
    Epoch 1128/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4453 - accuracy: 0.8087 - val_loss: 0.5980 - val_accuracy: 0.7013
    Epoch 1129/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4452 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7078
    Epoch 1130/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4451 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7013
    Epoch 1131/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4450 - accuracy: 0.8087 - val_loss: 0.5975 - val_accuracy: 0.7013
    Epoch 1132/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4450 - accuracy: 0.8109 - val_loss: 0.5978 - val_accuracy: 0.7013
    Epoch 1133/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4449 - accuracy: 0.8087 - val_loss: 0.5982 - val_accuracy: 0.7078
    Epoch 1134/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4448 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7013
    Epoch 1135/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4447 - accuracy: 0.8087 - val_loss: 0.5982 - val_accuracy: 0.7013
    Epoch 1136/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4446 - accuracy: 0.8087 - val_loss: 0.5981 - val_accuracy: 0.7013
    Epoch 1137/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4445 - accuracy: 0.8109 - val_loss: 0.5973 - val_accuracy: 0.7013
    Epoch 1138/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4445 - accuracy: 0.8109 - val_loss: 0.5972 - val_accuracy: 0.7078
    Epoch 1139/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4444 - accuracy: 0.8109 - val_loss: 0.5978 - val_accuracy: 0.7013
    Epoch 1140/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4443 - accuracy: 0.8130 - val_loss: 0.5974 - val_accuracy: 0.7013
    Epoch 1141/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4442 - accuracy: 0.8109 - val_loss: 0.5970 - val_accuracy: 0.7013
    Epoch 1142/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4442 - accuracy: 0.8130 - val_loss: 0.5973 - val_accuracy: 0.7013
    Epoch 1143/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4440 - accuracy: 0.8109 - val_loss: 0.5978 - val_accuracy: 0.7078
    Epoch 1144/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4440 - accuracy: 0.8130 - val_loss: 0.5969 - val_accuracy: 0.7078
    Epoch 1145/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4439 - accuracy: 0.8130 - val_loss: 0.5967 - val_accuracy: 0.7013
    Epoch 1146/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4438 - accuracy: 0.8130 - val_loss: 0.5973 - val_accuracy: 0.7013
    Epoch 1147/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4438 - accuracy: 0.8109 - val_loss: 0.5970 - val_accuracy: 0.7078
    Epoch 1148/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4436 - accuracy: 0.8130 - val_loss: 0.5970 - val_accuracy: 0.7078
    Epoch 1149/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4435 - accuracy: 0.8130 - val_loss: 0.5978 - val_accuracy: 0.7013
    Epoch 1150/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4435 - accuracy: 0.8130 - val_loss: 0.5975 - val_accuracy: 0.7013
    Epoch 1151/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4433 - accuracy: 0.8130 - val_loss: 0.5972 - val_accuracy: 0.7078
    Epoch 1152/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4433 - accuracy: 0.8109 - val_loss: 0.5974 - val_accuracy: 0.7078
    Epoch 1153/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4432 - accuracy: 0.8130 - val_loss: 0.5972 - val_accuracy: 0.7013
    Epoch 1154/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4431 - accuracy: 0.8130 - val_loss: 0.5960 - val_accuracy: 0.7013
    Epoch 1155/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4430 - accuracy: 0.8130 - val_loss: 0.5959 - val_accuracy: 0.7013
    Epoch 1156/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4429 - accuracy: 0.8130 - val_loss: 0.5967 - val_accuracy: 0.7013
    Epoch 1157/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4429 - accuracy: 0.8152 - val_loss: 0.5968 - val_accuracy: 0.7013
    Epoch 1158/2000
    1/1 [==============================] - 0s 75ms/step - loss: 0.4427 - accuracy: 0.8130 - val_loss: 0.5966 - val_accuracy: 0.7078
    Epoch 1159/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4427 - accuracy: 0.8130 - val_loss: 0.5970 - val_accuracy: 0.7078
    Epoch 1160/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4425 - accuracy: 0.8152 - val_loss: 0.5973 - val_accuracy: 0.7078
    Epoch 1161/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4425 - accuracy: 0.8152 - val_loss: 0.5967 - val_accuracy: 0.7078
    Epoch 1162/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4424 - accuracy: 0.8152 - val_loss: 0.5963 - val_accuracy: 0.7078
    Epoch 1163/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4423 - accuracy: 0.8130 - val_loss: 0.5967 - val_accuracy: 0.7078
    Epoch 1164/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4422 - accuracy: 0.8130 - val_loss: 0.5962 - val_accuracy: 0.7078
    Epoch 1165/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4422 - accuracy: 0.8130 - val_loss: 0.5962 - val_accuracy: 0.7078
    Epoch 1166/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4421 - accuracy: 0.8152 - val_loss: 0.5964 - val_accuracy: 0.7078
    Epoch 1167/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4420 - accuracy: 0.8152 - val_loss: 0.5960 - val_accuracy: 0.7078
    Epoch 1168/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4419 - accuracy: 0.8152 - val_loss: 0.5958 - val_accuracy: 0.7078
    Epoch 1169/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4419 - accuracy: 0.8130 - val_loss: 0.5965 - val_accuracy: 0.7078
    Epoch 1170/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4417 - accuracy: 0.8174 - val_loss: 0.5971 - val_accuracy: 0.7078
    Epoch 1171/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4417 - accuracy: 0.8152 - val_loss: 0.5965 - val_accuracy: 0.7078
    Epoch 1172/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4416 - accuracy: 0.8130 - val_loss: 0.5963 - val_accuracy: 0.7078
    Epoch 1173/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4415 - accuracy: 0.8152 - val_loss: 0.5962 - val_accuracy: 0.7078
    Epoch 1174/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4414 - accuracy: 0.8174 - val_loss: 0.5959 - val_accuracy: 0.7078
    Epoch 1175/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4413 - accuracy: 0.8152 - val_loss: 0.5955 - val_accuracy: 0.7078
    Epoch 1176/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4413 - accuracy: 0.8152 - val_loss: 0.5959 - val_accuracy: 0.7013
    Epoch 1177/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4412 - accuracy: 0.8152 - val_loss: 0.5961 - val_accuracy: 0.7013
    Epoch 1178/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4411 - accuracy: 0.8174 - val_loss: 0.5962 - val_accuracy: 0.7078
    Epoch 1179/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4410 - accuracy: 0.8152 - val_loss: 0.5965 - val_accuracy: 0.7078
    Epoch 1180/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4410 - accuracy: 0.8174 - val_loss: 0.5968 - val_accuracy: 0.7078
    Epoch 1181/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4409 - accuracy: 0.8174 - val_loss: 0.5960 - val_accuracy: 0.7013
    Epoch 1182/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4408 - accuracy: 0.8174 - val_loss: 0.5956 - val_accuracy: 0.7013
    Epoch 1183/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4407 - accuracy: 0.8152 - val_loss: 0.5958 - val_accuracy: 0.7078
    Epoch 1184/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4406 - accuracy: 0.8152 - val_loss: 0.5956 - val_accuracy: 0.7078
    Epoch 1185/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4405 - accuracy: 0.8174 - val_loss: 0.5960 - val_accuracy: 0.7078
    Epoch 1186/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4404 - accuracy: 0.8174 - val_loss: 0.5958 - val_accuracy: 0.7078
    Epoch 1187/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4404 - accuracy: 0.8174 - val_loss: 0.5958 - val_accuracy: 0.7078
    Epoch 1188/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4403 - accuracy: 0.8174 - val_loss: 0.5953 - val_accuracy: 0.7078
    Epoch 1189/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4402 - accuracy: 0.8152 - val_loss: 0.5955 - val_accuracy: 0.7078
    Epoch 1190/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4401 - accuracy: 0.8174 - val_loss: 0.5957 - val_accuracy: 0.7078
    Epoch 1191/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4400 - accuracy: 0.8174 - val_loss: 0.5955 - val_accuracy: 0.7078
    Epoch 1192/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4400 - accuracy: 0.8152 - val_loss: 0.5960 - val_accuracy: 0.7078
    Epoch 1193/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4399 - accuracy: 0.8174 - val_loss: 0.5959 - val_accuracy: 0.7078
    Epoch 1194/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4398 - accuracy: 0.8174 - val_loss: 0.5953 - val_accuracy: 0.7078
    Epoch 1195/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4397 - accuracy: 0.8174 - val_loss: 0.5951 - val_accuracy: 0.7078
    Epoch 1196/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4396 - accuracy: 0.8174 - val_loss: 0.5953 - val_accuracy: 0.7013
    Epoch 1197/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4396 - accuracy: 0.8130 - val_loss: 0.5946 - val_accuracy: 0.7078
    Epoch 1198/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4394 - accuracy: 0.8174 - val_loss: 0.5949 - val_accuracy: 0.7078
    Epoch 1199/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4393 - accuracy: 0.8174 - val_loss: 0.5949 - val_accuracy: 0.7013
    Epoch 1200/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4392 - accuracy: 0.8174 - val_loss: 0.5943 - val_accuracy: 0.7013
    Epoch 1201/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4391 - accuracy: 0.8174 - val_loss: 0.5945 - val_accuracy: 0.7078
    Epoch 1202/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4390 - accuracy: 0.8152 - val_loss: 0.5949 - val_accuracy: 0.7013
    Epoch 1203/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4390 - accuracy: 0.8152 - val_loss: 0.5939 - val_accuracy: 0.7013
    Epoch 1204/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4389 - accuracy: 0.8174 - val_loss: 0.5937 - val_accuracy: 0.7013
    Epoch 1205/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4387 - accuracy: 0.8174 - val_loss: 0.5942 - val_accuracy: 0.7013
    Epoch 1206/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4387 - accuracy: 0.8174 - val_loss: 0.5940 - val_accuracy: 0.7078
    Epoch 1207/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4386 - accuracy: 0.8196 - val_loss: 0.5943 - val_accuracy: 0.7078
    Epoch 1208/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4385 - accuracy: 0.8196 - val_loss: 0.5944 - val_accuracy: 0.7078
    Epoch 1209/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4384 - accuracy: 0.8174 - val_loss: 0.5943 - val_accuracy: 0.7078
    Epoch 1210/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4383 - accuracy: 0.8174 - val_loss: 0.5943 - val_accuracy: 0.7078
    Epoch 1211/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4382 - accuracy: 0.8174 - val_loss: 0.5944 - val_accuracy: 0.7078
    Epoch 1212/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4381 - accuracy: 0.8174 - val_loss: 0.5942 - val_accuracy: 0.7078
    Epoch 1213/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4380 - accuracy: 0.8174 - val_loss: 0.5947 - val_accuracy: 0.7013
    Epoch 1214/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4379 - accuracy: 0.8152 - val_loss: 0.5939 - val_accuracy: 0.7013
    Epoch 1215/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4378 - accuracy: 0.8196 - val_loss: 0.5939 - val_accuracy: 0.7078
    Epoch 1216/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4377 - accuracy: 0.8174 - val_loss: 0.5945 - val_accuracy: 0.7078
    Epoch 1217/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4377 - accuracy: 0.8152 - val_loss: 0.5939 - val_accuracy: 0.7078
    Epoch 1218/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4376 - accuracy: 0.8217 - val_loss: 0.5942 - val_accuracy: 0.7013
    Epoch 1219/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4375 - accuracy: 0.8174 - val_loss: 0.5948 - val_accuracy: 0.7078
    Epoch 1220/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4374 - accuracy: 0.8109 - val_loss: 0.5942 - val_accuracy: 0.7078
    Epoch 1221/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4373 - accuracy: 0.8196 - val_loss: 0.5939 - val_accuracy: 0.7078
    Epoch 1222/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4372 - accuracy: 0.8196 - val_loss: 0.5942 - val_accuracy: 0.7078
    Epoch 1223/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4371 - accuracy: 0.8174 - val_loss: 0.5943 - val_accuracy: 0.7078
    Epoch 1224/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4370 - accuracy: 0.8130 - val_loss: 0.5936 - val_accuracy: 0.7078
    Epoch 1225/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4369 - accuracy: 0.8152 - val_loss: 0.5937 - val_accuracy: 0.7078
    Epoch 1226/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4368 - accuracy: 0.8174 - val_loss: 0.5942 - val_accuracy: 0.7078
    Epoch 1227/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4368 - accuracy: 0.8174 - val_loss: 0.5942 - val_accuracy: 0.7078
    Epoch 1228/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4367 - accuracy: 0.8174 - val_loss: 0.5950 - val_accuracy: 0.7078
    Epoch 1229/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4365 - accuracy: 0.8174 - val_loss: 0.5948 - val_accuracy: 0.7078
    Epoch 1230/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.4365 - accuracy: 0.8174 - val_loss: 0.5947 - val_accuracy: 0.7078
    Epoch 1231/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4364 - accuracy: 0.8152 - val_loss: 0.5936 - val_accuracy: 0.7078
    Epoch 1232/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4363 - accuracy: 0.8196 - val_loss: 0.5939 - val_accuracy: 0.7143
    Epoch 1233/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4362 - accuracy: 0.8196 - val_loss: 0.5941 - val_accuracy: 0.7143
    Epoch 1234/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4362 - accuracy: 0.8152 - val_loss: 0.5934 - val_accuracy: 0.7078
    Epoch 1235/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4360 - accuracy: 0.8196 - val_loss: 0.5937 - val_accuracy: 0.7078
    Epoch 1236/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4359 - accuracy: 0.8174 - val_loss: 0.5946 - val_accuracy: 0.7013
    Epoch 1237/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4359 - accuracy: 0.8152 - val_loss: 0.5941 - val_accuracy: 0.7078
    Epoch 1238/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4358 - accuracy: 0.8174 - val_loss: 0.5943 - val_accuracy: 0.7078
    Epoch 1239/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.4356 - accuracy: 0.8196 - val_loss: 0.5948 - val_accuracy: 0.7143
    Epoch 1240/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4357 - accuracy: 0.8130 - val_loss: 0.5936 - val_accuracy: 0.7143
    Epoch 1241/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4355 - accuracy: 0.8196 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1242/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4355 - accuracy: 0.8196 - val_loss: 0.5938 - val_accuracy: 0.7143
    Epoch 1243/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4353 - accuracy: 0.8152 - val_loss: 0.5946 - val_accuracy: 0.7143
    Epoch 1244/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4353 - accuracy: 0.8109 - val_loss: 0.5939 - val_accuracy: 0.7078
    Epoch 1245/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4352 - accuracy: 0.8174 - val_loss: 0.5938 - val_accuracy: 0.7078
    Epoch 1246/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4351 - accuracy: 0.8196 - val_loss: 0.5944 - val_accuracy: 0.7143
    Epoch 1247/2000
    1/1 [==============================] - 0s 76ms/step - loss: 0.4350 - accuracy: 0.8152 - val_loss: 0.5937 - val_accuracy: 0.7143
    Epoch 1248/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4349 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1249/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4348 - accuracy: 0.8196 - val_loss: 0.5935 - val_accuracy: 0.7143
    Epoch 1250/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4347 - accuracy: 0.8196 - val_loss: 0.5941 - val_accuracy: 0.7078
    Epoch 1251/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4346 - accuracy: 0.8152 - val_loss: 0.5931 - val_accuracy: 0.7013
    Epoch 1252/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4346 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1253/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4344 - accuracy: 0.8196 - val_loss: 0.5940 - val_accuracy: 0.7143
    Epoch 1254/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4344 - accuracy: 0.8130 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1255/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4342 - accuracy: 0.8196 - val_loss: 0.5926 - val_accuracy: 0.7143
    Epoch 1256/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4342 - accuracy: 0.8174 - val_loss: 0.5937 - val_accuracy: 0.7143
    Epoch 1257/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4340 - accuracy: 0.8174 - val_loss: 0.5939 - val_accuracy: 0.7143
    Epoch 1258/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4339 - accuracy: 0.8152 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1259/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4339 - accuracy: 0.8196 - val_loss: 0.5936 - val_accuracy: 0.7143
    Epoch 1260/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4337 - accuracy: 0.8174 - val_loss: 0.5939 - val_accuracy: 0.7143
    Epoch 1261/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4337 - accuracy: 0.8130 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1262/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4336 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1263/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4335 - accuracy: 0.8196 - val_loss: 0.5945 - val_accuracy: 0.7143
    Epoch 1264/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4335 - accuracy: 0.8152 - val_loss: 0.5937 - val_accuracy: 0.7143
    Epoch 1265/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4333 - accuracy: 0.8174 - val_loss: 0.5933 - val_accuracy: 0.7143
    Epoch 1266/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4333 - accuracy: 0.8196 - val_loss: 0.5944 - val_accuracy: 0.7143
    Epoch 1267/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4332 - accuracy: 0.8174 - val_loss: 0.5939 - val_accuracy: 0.7143
    Epoch 1268/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4330 - accuracy: 0.8174 - val_loss: 0.5929 - val_accuracy: 0.7143
    Epoch 1269/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4330 - accuracy: 0.8196 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1270/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4329 - accuracy: 0.8196 - val_loss: 0.5938 - val_accuracy: 0.7143
    Epoch 1271/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4329 - accuracy: 0.8130 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1272/2000
    1/1 [==============================] - 0s 79ms/step - loss: 0.4327 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1273/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4326 - accuracy: 0.8196 - val_loss: 0.5936 - val_accuracy: 0.7143
    Epoch 1274/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4326 - accuracy: 0.8152 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1275/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4324 - accuracy: 0.8174 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1276/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4324 - accuracy: 0.8196 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1277/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4322 - accuracy: 0.8174 - val_loss: 0.5933 - val_accuracy: 0.7143
    Epoch 1278/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4322 - accuracy: 0.8152 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1279/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4321 - accuracy: 0.8196 - val_loss: 0.5932 - val_accuracy: 0.7143
    Epoch 1280/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4320 - accuracy: 0.8174 - val_loss: 0.5942 - val_accuracy: 0.7143
    Epoch 1281/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4321 - accuracy: 0.8130 - val_loss: 0.5932 - val_accuracy: 0.7143
    Epoch 1282/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4318 - accuracy: 0.8196 - val_loss: 0.5932 - val_accuracy: 0.7143
    Epoch 1283/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4317 - accuracy: 0.8196 - val_loss: 0.5942 - val_accuracy: 0.7143
    Epoch 1284/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4318 - accuracy: 0.8130 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1285/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4316 - accuracy: 0.8196 - val_loss: 0.5933 - val_accuracy: 0.7143
    Epoch 1286/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4315 - accuracy: 0.8196 - val_loss: 0.5938 - val_accuracy: 0.7143
    Epoch 1287/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4315 - accuracy: 0.8152 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1288/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4313 - accuracy: 0.8196 - val_loss: 0.5927 - val_accuracy: 0.7143
    Epoch 1289/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4312 - accuracy: 0.8196 - val_loss: 0.5936 - val_accuracy: 0.7143
    Epoch 1290/2000
    1/1 [==============================] - 0s 92ms/step - loss: 0.4312 - accuracy: 0.8152 - val_loss: 0.5927 - val_accuracy: 0.7143
    Epoch 1291/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4310 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7143
    Epoch 1292/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4311 - accuracy: 0.8174 - val_loss: 0.5927 - val_accuracy: 0.7143
    Epoch 1293/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4309 - accuracy: 0.8196 - val_loss: 0.5929 - val_accuracy: 0.7143
    Epoch 1294/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4308 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1295/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4309 - accuracy: 0.8196 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1296/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4307 - accuracy: 0.8196 - val_loss: 0.5934 - val_accuracy: 0.7078
    Epoch 1297/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4307 - accuracy: 0.8130 - val_loss: 0.5926 - val_accuracy: 0.7143
    Epoch 1298/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4305 - accuracy: 0.8196 - val_loss: 0.5930 - val_accuracy: 0.7143
    Epoch 1299/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4304 - accuracy: 0.8196 - val_loss: 0.5939 - val_accuracy: 0.7143
    Epoch 1300/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4304 - accuracy: 0.8130 - val_loss: 0.5927 - val_accuracy: 0.7143
    Epoch 1301/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4302 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1302/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4301 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1303/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4301 - accuracy: 0.8109 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1304/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4299 - accuracy: 0.8196 - val_loss: 0.5922 - val_accuracy: 0.7143
    Epoch 1305/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4300 - accuracy: 0.8196 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1306/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4298 - accuracy: 0.8152 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1307/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4296 - accuracy: 0.8152 - val_loss: 0.5929 - val_accuracy: 0.7143
    Epoch 1308/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4296 - accuracy: 0.8196 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1309/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4294 - accuracy: 0.8174 - val_loss: 0.5937 - val_accuracy: 0.7143
    Epoch 1310/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4295 - accuracy: 0.8130 - val_loss: 0.5922 - val_accuracy: 0.7143
    Epoch 1311/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4294 - accuracy: 0.8196 - val_loss: 0.5922 - val_accuracy: 0.7143
    Epoch 1312/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4292 - accuracy: 0.8196 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1313/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4293 - accuracy: 0.8130 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1314/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4291 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7143
    Epoch 1315/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4290 - accuracy: 0.8196 - val_loss: 0.5934 - val_accuracy: 0.7143
    Epoch 1316/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.4289 - accuracy: 0.8152 - val_loss: 0.5930 - val_accuracy: 0.7143
    Epoch 1317/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4288 - accuracy: 0.8152 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1318/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4288 - accuracy: 0.8196 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1319/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.4286 - accuracy: 0.8152 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1320/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4285 - accuracy: 0.8152 - val_loss: 0.5921 - val_accuracy: 0.7143
    Epoch 1321/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4284 - accuracy: 0.8196 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1322/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4283 - accuracy: 0.8174 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1323/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4282 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1324/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4281 - accuracy: 0.8174 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1325/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4280 - accuracy: 0.8130 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1326/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4280 - accuracy: 0.8196 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1327/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4279 - accuracy: 0.8196 - val_loss: 0.5928 - val_accuracy: 0.7143
    Epoch 1328/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4278 - accuracy: 0.8152 - val_loss: 0.5926 - val_accuracy: 0.7143
    Epoch 1329/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4276 - accuracy: 0.8196 - val_loss: 0.5929 - val_accuracy: 0.7143
    Epoch 1330/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4275 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7143
    Epoch 1331/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4275 - accuracy: 0.8196 - val_loss: 0.5927 - val_accuracy: 0.7143
    Epoch 1332/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4274 - accuracy: 0.8174 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1333/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4273 - accuracy: 0.8174 - val_loss: 0.5914 - val_accuracy: 0.7143
    Epoch 1334/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4272 - accuracy: 0.8152 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1335/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4272 - accuracy: 0.8130 - val_loss: 0.5914 - val_accuracy: 0.7143
    Epoch 1336/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4271 - accuracy: 0.8196 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1337/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4269 - accuracy: 0.8174 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1338/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4268 - accuracy: 0.8130 - val_loss: 0.5926 - val_accuracy: 0.7143
    Epoch 1339/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4267 - accuracy: 0.8196 - val_loss: 0.5933 - val_accuracy: 0.7143
    Epoch 1340/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4267 - accuracy: 0.8152 - val_loss: 0.5932 - val_accuracy: 0.7143
    Epoch 1341/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4265 - accuracy: 0.8130 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1342/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4265 - accuracy: 0.8196 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1343/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4264 - accuracy: 0.8152 - val_loss: 0.5915 - val_accuracy: 0.7143
    Epoch 1344/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4263 - accuracy: 0.8152 - val_loss: 0.5911 - val_accuracy: 0.7143
    Epoch 1345/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4262 - accuracy: 0.8174 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1346/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4261 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1347/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4260 - accuracy: 0.8152 - val_loss: 0.5914 - val_accuracy: 0.7143
    Epoch 1348/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4259 - accuracy: 0.8174 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1349/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4258 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1350/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4257 - accuracy: 0.8174 - val_loss: 0.5922 - val_accuracy: 0.7143
    Epoch 1351/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4256 - accuracy: 0.8152 - val_loss: 0.5921 - val_accuracy: 0.7143
    Epoch 1352/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4256 - accuracy: 0.8174 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1353/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4255 - accuracy: 0.8174 - val_loss: 0.5912 - val_accuracy: 0.7143
    Epoch 1354/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4254 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7143
    Epoch 1355/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4253 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1356/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4252 - accuracy: 0.8174 - val_loss: 0.5913 - val_accuracy: 0.7143
    Epoch 1357/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4251 - accuracy: 0.8174 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1358/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4250 - accuracy: 0.8174 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1359/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4249 - accuracy: 0.8152 - val_loss: 0.5912 - val_accuracy: 0.7143
    Epoch 1360/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4248 - accuracy: 0.8174 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1361/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4247 - accuracy: 0.8152 - val_loss: 0.5915 - val_accuracy: 0.7143
    Epoch 1362/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4246 - accuracy: 0.8152 - val_loss: 0.5915 - val_accuracy: 0.7143
    Epoch 1363/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4245 - accuracy: 0.8196 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1364/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4245 - accuracy: 0.8152 - val_loss: 0.5913 - val_accuracy: 0.7143
    Epoch 1365/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4244 - accuracy: 0.8196 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1366/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4243 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7143
    Epoch 1367/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4242 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1368/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4241 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1369/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4240 - accuracy: 0.8196 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1370/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4239 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7078
    Epoch 1371/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4238 - accuracy: 0.8196 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1372/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.4237 - accuracy: 0.8152 - val_loss: 0.5919 - val_accuracy: 0.7078
    Epoch 1373/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4236 - accuracy: 0.8152 - val_loss: 0.5914 - val_accuracy: 0.7143
    Epoch 1374/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4236 - accuracy: 0.8174 - val_loss: 0.5922 - val_accuracy: 0.7143
    Epoch 1375/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4235 - accuracy: 0.8174 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1376/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4234 - accuracy: 0.8152 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1377/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4233 - accuracy: 0.8196 - val_loss: 0.5925 - val_accuracy: 0.7078
    Epoch 1378/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4232 - accuracy: 0.8152 - val_loss: 0.5930 - val_accuracy: 0.7078
    Epoch 1379/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4231 - accuracy: 0.8130 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1380/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4230 - accuracy: 0.8174 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1381/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4229 - accuracy: 0.8152 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1382/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4227 - accuracy: 0.8174 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1383/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4227 - accuracy: 0.8174 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1384/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4225 - accuracy: 0.8152 - val_loss: 0.5917 - val_accuracy: 0.7078
    Epoch 1385/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4224 - accuracy: 0.8152 - val_loss: 0.5926 - val_accuracy: 0.7078
    Epoch 1386/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4224 - accuracy: 0.8152 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1387/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4224 - accuracy: 0.8196 - val_loss: 0.5931 - val_accuracy: 0.7078
    Epoch 1388/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4222 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7078
    Epoch 1389/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4221 - accuracy: 0.8152 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1390/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4220 - accuracy: 0.8217 - val_loss: 0.5925 - val_accuracy: 0.7078
    Epoch 1391/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4219 - accuracy: 0.8152 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1392/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4217 - accuracy: 0.8174 - val_loss: 0.5915 - val_accuracy: 0.7078
    Epoch 1393/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4217 - accuracy: 0.8196 - val_loss: 0.5925 - val_accuracy: 0.7078
    Epoch 1394/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4216 - accuracy: 0.8152 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1395/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4214 - accuracy: 0.8174 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1396/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4213 - accuracy: 0.8174 - val_loss: 0.5935 - val_accuracy: 0.7078
    Epoch 1397/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4213 - accuracy: 0.8152 - val_loss: 0.5929 - val_accuracy: 0.7078
    Epoch 1398/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4212 - accuracy: 0.8196 - val_loss: 0.5932 - val_accuracy: 0.7078
    Epoch 1399/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4210 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7078
    Epoch 1400/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4210 - accuracy: 0.8130 - val_loss: 0.5917 - val_accuracy: 0.7078
    Epoch 1401/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4210 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7078
    Epoch 1402/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4208 - accuracy: 0.8196 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1403/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4207 - accuracy: 0.8174 - val_loss: 0.5913 - val_accuracy: 0.7078
    Epoch 1404/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4207 - accuracy: 0.8196 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1405/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4206 - accuracy: 0.8130 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1406/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4205 - accuracy: 0.8174 - val_loss: 0.5931 - val_accuracy: 0.7078
    Epoch 1407/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4203 - accuracy: 0.8196 - val_loss: 0.5941 - val_accuracy: 0.7078
    Epoch 1408/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4203 - accuracy: 0.8152 - val_loss: 0.5935 - val_accuracy: 0.7143
    Epoch 1409/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.4202 - accuracy: 0.8196 - val_loss: 0.5940 - val_accuracy: 0.7078
    Epoch 1410/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4201 - accuracy: 0.8152 - val_loss: 0.5931 - val_accuracy: 0.7078
    Epoch 1411/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4200 - accuracy: 0.8174 - val_loss: 0.5928 - val_accuracy: 0.7078
    Epoch 1412/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4199 - accuracy: 0.8174 - val_loss: 0.5919 - val_accuracy: 0.7078
    Epoch 1413/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4198 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1414/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4197 - accuracy: 0.8174 - val_loss: 0.5917 - val_accuracy: 0.7078
    Epoch 1415/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4196 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1416/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4195 - accuracy: 0.8174 - val_loss: 0.5926 - val_accuracy: 0.7078
    Epoch 1417/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4194 - accuracy: 0.8174 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1418/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4193 - accuracy: 0.8196 - val_loss: 0.5927 - val_accuracy: 0.7078
    Epoch 1419/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4192 - accuracy: 0.8174 - val_loss: 0.5928 - val_accuracy: 0.7078
    Epoch 1420/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4192 - accuracy: 0.8174 - val_loss: 0.5918 - val_accuracy: 0.7078
    Epoch 1421/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4191 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7078
    Epoch 1422/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4190 - accuracy: 0.8174 - val_loss: 0.5914 - val_accuracy: 0.7078
    Epoch 1423/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4189 - accuracy: 0.8152 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1424/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4188 - accuracy: 0.8174 - val_loss: 0.5930 - val_accuracy: 0.7078
    Epoch 1425/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4187 - accuracy: 0.8152 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1426/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4186 - accuracy: 0.8196 - val_loss: 0.5930 - val_accuracy: 0.7078
    Epoch 1427/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4186 - accuracy: 0.8196 - val_loss: 0.5928 - val_accuracy: 0.7078
    Epoch 1428/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4184 - accuracy: 0.8174 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1429/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4184 - accuracy: 0.8196 - val_loss: 0.5927 - val_accuracy: 0.7078
    Epoch 1430/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4183 - accuracy: 0.8196 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1431/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4181 - accuracy: 0.8152 - val_loss: 0.5918 - val_accuracy: 0.7078
    Epoch 1432/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4181 - accuracy: 0.8196 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1433/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4180 - accuracy: 0.8196 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1434/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4179 - accuracy: 0.8217 - val_loss: 0.5914 - val_accuracy: 0.7078
    Epoch 1435/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4178 - accuracy: 0.8217 - val_loss: 0.5913 - val_accuracy: 0.7078
    Epoch 1436/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4177 - accuracy: 0.8196 - val_loss: 0.5928 - val_accuracy: 0.7078
    Epoch 1437/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.4178 - accuracy: 0.8196 - val_loss: 0.5923 - val_accuracy: 0.7143
    Epoch 1438/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4176 - accuracy: 0.8196 - val_loss: 0.5933 - val_accuracy: 0.7078
    Epoch 1439/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4174 - accuracy: 0.8174 - val_loss: 0.5939 - val_accuracy: 0.7078
    Epoch 1440/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4175 - accuracy: 0.8152 - val_loss: 0.5921 - val_accuracy: 0.7143
    Epoch 1441/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4173 - accuracy: 0.8196 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1442/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4172 - accuracy: 0.8217 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1443/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4171 - accuracy: 0.8196 - val_loss: 0.5909 - val_accuracy: 0.7078
    Epoch 1444/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4171 - accuracy: 0.8217 - val_loss: 0.5911 - val_accuracy: 0.7078
    Epoch 1445/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4169 - accuracy: 0.8196 - val_loss: 0.5915 - val_accuracy: 0.7078
    Epoch 1446/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4168 - accuracy: 0.8174 - val_loss: 0.5909 - val_accuracy: 0.7143
    Epoch 1447/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4169 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1448/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4166 - accuracy: 0.8174 - val_loss: 0.5934 - val_accuracy: 0.7078
    Epoch 1449/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4166 - accuracy: 0.8152 - val_loss: 0.5920 - val_accuracy: 0.7143
    Epoch 1450/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4166 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1451/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4163 - accuracy: 0.8196 - val_loss: 0.5933 - val_accuracy: 0.7078
    Epoch 1452/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4164 - accuracy: 0.8196 - val_loss: 0.5915 - val_accuracy: 0.7143
    Epoch 1453/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4162 - accuracy: 0.8196 - val_loss: 0.5913 - val_accuracy: 0.7013
    Epoch 1454/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4161 - accuracy: 0.8174 - val_loss: 0.5927 - val_accuracy: 0.7013
    Epoch 1455/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4162 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1456/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4159 - accuracy: 0.8217 - val_loss: 0.5918 - val_accuracy: 0.7143
    Epoch 1457/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4160 - accuracy: 0.8152 - val_loss: 0.5933 - val_accuracy: 0.7013
    Epoch 1458/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4158 - accuracy: 0.8196 - val_loss: 0.5925 - val_accuracy: 0.7078
    Epoch 1459/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4156 - accuracy: 0.8196 - val_loss: 0.5913 - val_accuracy: 0.7143
    Epoch 1460/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4157 - accuracy: 0.8152 - val_loss: 0.5923 - val_accuracy: 0.7078
    Epoch 1461/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4154 - accuracy: 0.8196 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1462/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4154 - accuracy: 0.8174 - val_loss: 0.5920 - val_accuracy: 0.7078
    Epoch 1463/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.4153 - accuracy: 0.8174 - val_loss: 0.5926 - val_accuracy: 0.7078
    Epoch 1464/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4152 - accuracy: 0.8174 - val_loss: 0.5921 - val_accuracy: 0.7143
    Epoch 1465/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4151 - accuracy: 0.8174 - val_loss: 0.5924 - val_accuracy: 0.7143
    Epoch 1466/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4150 - accuracy: 0.8196 - val_loss: 0.5929 - val_accuracy: 0.7078
    Epoch 1467/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4149 - accuracy: 0.8196 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1468/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4148 - accuracy: 0.8196 - val_loss: 0.5918 - val_accuracy: 0.7078
    Epoch 1469/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4147 - accuracy: 0.8217 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1470/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4146 - accuracy: 0.8196 - val_loss: 0.5914 - val_accuracy: 0.7078
    Epoch 1471/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4146 - accuracy: 0.8217 - val_loss: 0.5921 - val_accuracy: 0.7078
    Epoch 1472/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4144 - accuracy: 0.8217 - val_loss: 0.5932 - val_accuracy: 0.7013
    Epoch 1473/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4145 - accuracy: 0.8174 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1474/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4143 - accuracy: 0.8217 - val_loss: 0.5925 - val_accuracy: 0.7143
    Epoch 1475/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4143 - accuracy: 0.8196 - val_loss: 0.5945 - val_accuracy: 0.7013
    Epoch 1476/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4143 - accuracy: 0.8217 - val_loss: 0.5935 - val_accuracy: 0.7078
    Epoch 1477/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4140 - accuracy: 0.8239 - val_loss: 0.5916 - val_accuracy: 0.7143
    Epoch 1478/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4140 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7013
    Epoch 1479/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4138 - accuracy: 0.8217 - val_loss: 0.5912 - val_accuracy: 0.7078
    Epoch 1480/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4137 - accuracy: 0.8239 - val_loss: 0.5903 - val_accuracy: 0.7143
    Epoch 1481/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4138 - accuracy: 0.8196 - val_loss: 0.5915 - val_accuracy: 0.7078
    Epoch 1482/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4135 - accuracy: 0.8239 - val_loss: 0.5924 - val_accuracy: 0.7078
    Epoch 1483/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4135 - accuracy: 0.8196 - val_loss: 0.5919 - val_accuracy: 0.7143
    Epoch 1484/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4134 - accuracy: 0.8196 - val_loss: 0.5933 - val_accuracy: 0.7013
    Epoch 1485/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4132 - accuracy: 0.8217 - val_loss: 0.5940 - val_accuracy: 0.7013
    Epoch 1486/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4132 - accuracy: 0.8217 - val_loss: 0.5922 - val_accuracy: 0.7078
    Epoch 1487/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4130 - accuracy: 0.8217 - val_loss: 0.5917 - val_accuracy: 0.7013
    Epoch 1488/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4128 - accuracy: 0.8196 - val_loss: 0.5917 - val_accuracy: 0.7013
    Epoch 1489/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4128 - accuracy: 0.8217 - val_loss: 0.5912 - val_accuracy: 0.7013
    Epoch 1490/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4126 - accuracy: 0.8196 - val_loss: 0.5915 - val_accuracy: 0.7013
    Epoch 1491/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4125 - accuracy: 0.8152 - val_loss: 0.5920 - val_accuracy: 0.7013
    Epoch 1492/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4124 - accuracy: 0.8239 - val_loss: 0.5919 - val_accuracy: 0.7013
    Epoch 1493/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4122 - accuracy: 0.8196 - val_loss: 0.5917 - val_accuracy: 0.7078
    Epoch 1494/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4121 - accuracy: 0.8196 - val_loss: 0.5914 - val_accuracy: 0.7078
    Epoch 1495/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4120 - accuracy: 0.8196 - val_loss: 0.5924 - val_accuracy: 0.7013
    Epoch 1496/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4120 - accuracy: 0.8217 - val_loss: 0.5918 - val_accuracy: 0.7013
    Epoch 1497/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.4117 - accuracy: 0.8261 - val_loss: 0.5916 - val_accuracy: 0.7078
    Epoch 1498/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4117 - accuracy: 0.8217 - val_loss: 0.5928 - val_accuracy: 0.7013
    Epoch 1499/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4115 - accuracy: 0.8239 - val_loss: 0.5936 - val_accuracy: 0.7013
    Epoch 1500/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4114 - accuracy: 0.8239 - val_loss: 0.5930 - val_accuracy: 0.7078
    Epoch 1501/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4113 - accuracy: 0.8196 - val_loss: 0.5934 - val_accuracy: 0.7013
    Epoch 1502/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4111 - accuracy: 0.8217 - val_loss: 0.5927 - val_accuracy: 0.7078
    Epoch 1503/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4110 - accuracy: 0.8239 - val_loss: 0.5928 - val_accuracy: 0.7013
    Epoch 1504/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4108 - accuracy: 0.8239 - val_loss: 0.5932 - val_accuracy: 0.6948
    Epoch 1505/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4107 - accuracy: 0.8217 - val_loss: 0.5924 - val_accuracy: 0.6948
    Epoch 1506/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4106 - accuracy: 0.8239 - val_loss: 0.5924 - val_accuracy: 0.6948
    Epoch 1507/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4105 - accuracy: 0.8261 - val_loss: 0.5927 - val_accuracy: 0.7013
    Epoch 1508/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4104 - accuracy: 0.8239 - val_loss: 0.5919 - val_accuracy: 0.7013
    Epoch 1509/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4102 - accuracy: 0.8283 - val_loss: 0.5926 - val_accuracy: 0.6948
    Epoch 1510/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4101 - accuracy: 0.8261 - val_loss: 0.5937 - val_accuracy: 0.6948
    Epoch 1511/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4101 - accuracy: 0.8217 - val_loss: 0.5923 - val_accuracy: 0.7013
    Epoch 1512/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4100 - accuracy: 0.8239 - val_loss: 0.5927 - val_accuracy: 0.6948
    Epoch 1513/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4098 - accuracy: 0.8261 - val_loss: 0.5941 - val_accuracy: 0.6948
    Epoch 1514/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4099 - accuracy: 0.8217 - val_loss: 0.5918 - val_accuracy: 0.7013
    Epoch 1515/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4096 - accuracy: 0.8261 - val_loss: 0.5920 - val_accuracy: 0.7013
    Epoch 1516/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4094 - accuracy: 0.8283 - val_loss: 0.5942 - val_accuracy: 0.6948
    Epoch 1517/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4094 - accuracy: 0.8217 - val_loss: 0.5936 - val_accuracy: 0.6948
    Epoch 1518/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4091 - accuracy: 0.8304 - val_loss: 0.5935 - val_accuracy: 0.7013
    Epoch 1519/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4091 - accuracy: 0.8283 - val_loss: 0.5952 - val_accuracy: 0.6883
    Epoch 1520/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4090 - accuracy: 0.8217 - val_loss: 0.5952 - val_accuracy: 0.6883
    Epoch 1521/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4088 - accuracy: 0.8283 - val_loss: 0.5938 - val_accuracy: 0.7013
    Epoch 1522/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4087 - accuracy: 0.8261 - val_loss: 0.5939 - val_accuracy: 0.6948
    Epoch 1523/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4085 - accuracy: 0.8261 - val_loss: 0.5947 - val_accuracy: 0.6883
    Epoch 1524/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4086 - accuracy: 0.8239 - val_loss: 0.5935 - val_accuracy: 0.7013
    Epoch 1525/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.4083 - accuracy: 0.8304 - val_loss: 0.5939 - val_accuracy: 0.7013
    Epoch 1526/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4082 - accuracy: 0.8283 - val_loss: 0.5953 - val_accuracy: 0.6883
    Epoch 1527/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4081 - accuracy: 0.8261 - val_loss: 0.5953 - val_accuracy: 0.6883
    Epoch 1528/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4080 - accuracy: 0.8261 - val_loss: 0.5948 - val_accuracy: 0.6948
    Epoch 1529/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.4079 - accuracy: 0.8283 - val_loss: 0.5953 - val_accuracy: 0.6883
    Epoch 1530/2000
    1/1 [==============================] - 0s 75ms/step - loss: 0.4077 - accuracy: 0.8261 - val_loss: 0.5956 - val_accuracy: 0.6883
    Epoch 1531/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4077 - accuracy: 0.8239 - val_loss: 0.5943 - val_accuracy: 0.7013
    Epoch 1532/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4076 - accuracy: 0.8283 - val_loss: 0.5948 - val_accuracy: 0.6883
    Epoch 1533/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4074 - accuracy: 0.8283 - val_loss: 0.5962 - val_accuracy: 0.6883
    Epoch 1534/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4075 - accuracy: 0.8239 - val_loss: 0.5949 - val_accuracy: 0.7013
    Epoch 1535/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.4073 - accuracy: 0.8283 - val_loss: 0.5951 - val_accuracy: 0.7013
    Epoch 1536/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4071 - accuracy: 0.8304 - val_loss: 0.5966 - val_accuracy: 0.6883
    Epoch 1537/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4071 - accuracy: 0.8196 - val_loss: 0.5945 - val_accuracy: 0.7013
    Epoch 1538/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4069 - accuracy: 0.8283 - val_loss: 0.5941 - val_accuracy: 0.7013
    Epoch 1539/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4068 - accuracy: 0.8283 - val_loss: 0.5955 - val_accuracy: 0.6883
    Epoch 1540/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4067 - accuracy: 0.8196 - val_loss: 0.5954 - val_accuracy: 0.6883
    Epoch 1541/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4066 - accuracy: 0.8261 - val_loss: 0.5958 - val_accuracy: 0.6948
    Epoch 1542/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4065 - accuracy: 0.8283 - val_loss: 0.5969 - val_accuracy: 0.6883
    Epoch 1543/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4063 - accuracy: 0.8261 - val_loss: 0.5969 - val_accuracy: 0.6883
    Epoch 1544/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4061 - accuracy: 0.8261 - val_loss: 0.5961 - val_accuracy: 0.6883
    Epoch 1545/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4060 - accuracy: 0.8283 - val_loss: 0.5959 - val_accuracy: 0.6883
    Epoch 1546/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4059 - accuracy: 0.8261 - val_loss: 0.5959 - val_accuracy: 0.6883
    Epoch 1547/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4058 - accuracy: 0.8239 - val_loss: 0.5952 - val_accuracy: 0.6948
    Epoch 1548/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4057 - accuracy: 0.8261 - val_loss: 0.5957 - val_accuracy: 0.6883
    Epoch 1549/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4056 - accuracy: 0.8261 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1550/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.4055 - accuracy: 0.8261 - val_loss: 0.5961 - val_accuracy: 0.6948
    Epoch 1551/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4054 - accuracy: 0.8261 - val_loss: 0.5962 - val_accuracy: 0.6883
    Epoch 1552/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.4052 - accuracy: 0.8261 - val_loss: 0.5963 - val_accuracy: 0.6883
    Epoch 1553/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4052 - accuracy: 0.8261 - val_loss: 0.5958 - val_accuracy: 0.6948
    Epoch 1554/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4050 - accuracy: 0.8261 - val_loss: 0.5960 - val_accuracy: 0.6948
    Epoch 1555/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4049 - accuracy: 0.8283 - val_loss: 0.5969 - val_accuracy: 0.6883
    Epoch 1556/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.4048 - accuracy: 0.8261 - val_loss: 0.5964 - val_accuracy: 0.6883
    Epoch 1557/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4047 - accuracy: 0.8261 - val_loss: 0.5964 - val_accuracy: 0.6883
    Epoch 1558/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4046 - accuracy: 0.8239 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1559/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4045 - accuracy: 0.8261 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1560/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.4044 - accuracy: 0.8239 - val_loss: 0.5964 - val_accuracy: 0.6948
    Epoch 1561/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4042 - accuracy: 0.8261 - val_loss: 0.5961 - val_accuracy: 0.6883
    Epoch 1562/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4041 - accuracy: 0.8261 - val_loss: 0.5957 - val_accuracy: 0.6883
    Epoch 1563/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4040 - accuracy: 0.8239 - val_loss: 0.5960 - val_accuracy: 0.6883
    Epoch 1564/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4039 - accuracy: 0.8261 - val_loss: 0.5962 - val_accuracy: 0.6883
    Epoch 1565/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4038 - accuracy: 0.8239 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1566/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4037 - accuracy: 0.8261 - val_loss: 0.5970 - val_accuracy: 0.6883
    Epoch 1567/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4036 - accuracy: 0.8261 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1568/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.4035 - accuracy: 0.8261 - val_loss: 0.5970 - val_accuracy: 0.6883
    Epoch 1569/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4034 - accuracy: 0.8261 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1570/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.4032 - accuracy: 0.8261 - val_loss: 0.5963 - val_accuracy: 0.6948
    Epoch 1571/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.4031 - accuracy: 0.8261 - val_loss: 0.5962 - val_accuracy: 0.6948
    Epoch 1572/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.4030 - accuracy: 0.8261 - val_loss: 0.5965 - val_accuracy: 0.6883
    Epoch 1573/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4029 - accuracy: 0.8261 - val_loss: 0.5963 - val_accuracy: 0.6883
    Epoch 1574/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4027 - accuracy: 0.8261 - val_loss: 0.5960 - val_accuracy: 0.6948
    Epoch 1575/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4027 - accuracy: 0.8239 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1576/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4025 - accuracy: 0.8261 - val_loss: 0.5963 - val_accuracy: 0.6948
    Epoch 1577/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4024 - accuracy: 0.8261 - val_loss: 0.5969 - val_accuracy: 0.6883
    Epoch 1578/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4022 - accuracy: 0.8283 - val_loss: 0.5971 - val_accuracy: 0.6883
    Epoch 1579/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4022 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6948
    Epoch 1580/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4020 - accuracy: 0.8283 - val_loss: 0.5963 - val_accuracy: 0.6883
    Epoch 1581/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.4019 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6883
    Epoch 1582/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.4018 - accuracy: 0.8283 - val_loss: 0.5961 - val_accuracy: 0.6948
    Epoch 1583/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.4017 - accuracy: 0.8283 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1584/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.4016 - accuracy: 0.8283 - val_loss: 0.5968 - val_accuracy: 0.6948
    Epoch 1585/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4015 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6948
    Epoch 1586/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4013 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6883
    Epoch 1587/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.4012 - accuracy: 0.8283 - val_loss: 0.5957 - val_accuracy: 0.6883
    Epoch 1588/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.4011 - accuracy: 0.8261 - val_loss: 0.5957 - val_accuracy: 0.6948
    Epoch 1589/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.4010 - accuracy: 0.8261 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1590/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4009 - accuracy: 0.8283 - val_loss: 0.5969 - val_accuracy: 0.6948
    Epoch 1591/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4008 - accuracy: 0.8283 - val_loss: 0.5960 - val_accuracy: 0.6948
    Epoch 1592/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.4007 - accuracy: 0.8283 - val_loss: 0.5963 - val_accuracy: 0.6883
    Epoch 1593/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4005 - accuracy: 0.8283 - val_loss: 0.5966 - val_accuracy: 0.6883
    Epoch 1594/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.4004 - accuracy: 0.8283 - val_loss: 0.5958 - val_accuracy: 0.6948
    Epoch 1595/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.4004 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6883
    Epoch 1596/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.4002 - accuracy: 0.8283 - val_loss: 0.5970 - val_accuracy: 0.6883
    Epoch 1597/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.4001 - accuracy: 0.8283 - val_loss: 0.5962 - val_accuracy: 0.6948
    Epoch 1598/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.4001 - accuracy: 0.8261 - val_loss: 0.5970 - val_accuracy: 0.6883
    Epoch 1599/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3999 - accuracy: 0.8304 - val_loss: 0.5966 - val_accuracy: 0.6883
    Epoch 1600/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3997 - accuracy: 0.8283 - val_loss: 0.5961 - val_accuracy: 0.6948
    Epoch 1601/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3996 - accuracy: 0.8283 - val_loss: 0.5961 - val_accuracy: 0.6948
    Epoch 1602/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.3995 - accuracy: 0.8326 - val_loss: 0.5958 - val_accuracy: 0.6948
    Epoch 1603/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3994 - accuracy: 0.8283 - val_loss: 0.5958 - val_accuracy: 0.6883
    Epoch 1604/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3993 - accuracy: 0.8283 - val_loss: 0.5963 - val_accuracy: 0.6883
    Epoch 1605/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3992 - accuracy: 0.8304 - val_loss: 0.5963 - val_accuracy: 0.6948
    Epoch 1606/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3990 - accuracy: 0.8283 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1607/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3990 - accuracy: 0.8326 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1608/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3988 - accuracy: 0.8304 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1609/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.3988 - accuracy: 0.8283 - val_loss: 0.5969 - val_accuracy: 0.6883
    Epoch 1610/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.3986 - accuracy: 0.8304 - val_loss: 0.5957 - val_accuracy: 0.6883
    Epoch 1611/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3985 - accuracy: 0.8304 - val_loss: 0.5950 - val_accuracy: 0.6948
    Epoch 1612/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3984 - accuracy: 0.8283 - val_loss: 0.5962 - val_accuracy: 0.6883
    Epoch 1613/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3982 - accuracy: 0.8304 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1614/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3981 - accuracy: 0.8304 - val_loss: 0.5962 - val_accuracy: 0.6948
    Epoch 1615/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.3981 - accuracy: 0.8283 - val_loss: 0.5971 - val_accuracy: 0.6948
    Epoch 1616/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3979 - accuracy: 0.8304 - val_loss: 0.5972 - val_accuracy: 0.6883
    Epoch 1617/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3978 - accuracy: 0.8326 - val_loss: 0.5953 - val_accuracy: 0.7013
    Epoch 1618/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3978 - accuracy: 0.8283 - val_loss: 0.5961 - val_accuracy: 0.6883
    Epoch 1619/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3975 - accuracy: 0.8326 - val_loss: 0.5967 - val_accuracy: 0.6883
    Epoch 1620/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3975 - accuracy: 0.8283 - val_loss: 0.5952 - val_accuracy: 0.7013
    Epoch 1621/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3975 - accuracy: 0.8283 - val_loss: 0.5964 - val_accuracy: 0.6948
    Epoch 1622/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3972 - accuracy: 0.8304 - val_loss: 0.5984 - val_accuracy: 0.6883
    Epoch 1623/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3973 - accuracy: 0.8283 - val_loss: 0.5967 - val_accuracy: 0.6948
    Epoch 1624/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3971 - accuracy: 0.8326 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1625/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3968 - accuracy: 0.8283 - val_loss: 0.5982 - val_accuracy: 0.6883
    Epoch 1626/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3970 - accuracy: 0.8283 - val_loss: 0.5958 - val_accuracy: 0.7013
    Epoch 1627/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3967 - accuracy: 0.8304 - val_loss: 0.5962 - val_accuracy: 0.7013
    Epoch 1628/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3966 - accuracy: 0.8304 - val_loss: 0.5984 - val_accuracy: 0.6883
    Epoch 1629/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3966 - accuracy: 0.8283 - val_loss: 0.5965 - val_accuracy: 0.6948
    Epoch 1630/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3962 - accuracy: 0.8304 - val_loss: 0.5956 - val_accuracy: 0.6948
    Epoch 1631/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3962 - accuracy: 0.8326 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1632/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3962 - accuracy: 0.8304 - val_loss: 0.5957 - val_accuracy: 0.6948
    Epoch 1633/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3959 - accuracy: 0.8304 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1634/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3958 - accuracy: 0.8304 - val_loss: 0.5980 - val_accuracy: 0.6948
    Epoch 1635/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3957 - accuracy: 0.8304 - val_loss: 0.5978 - val_accuracy: 0.6948
    Epoch 1636/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3955 - accuracy: 0.8304 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1637/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3954 - accuracy: 0.8304 - val_loss: 0.5968 - val_accuracy: 0.6883
    Epoch 1638/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3953 - accuracy: 0.8304 - val_loss: 0.5963 - val_accuracy: 0.6948
    Epoch 1639/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3952 - accuracy: 0.8326 - val_loss: 0.5958 - val_accuracy: 0.6948
    Epoch 1640/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3951 - accuracy: 0.8326 - val_loss: 0.5973 - val_accuracy: 0.6883
    Epoch 1641/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3950 - accuracy: 0.8326 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1642/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3948 - accuracy: 0.8326 - val_loss: 0.5961 - val_accuracy: 0.7013
    Epoch 1643/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3948 - accuracy: 0.8326 - val_loss: 0.5981 - val_accuracy: 0.6883
    Epoch 1644/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3947 - accuracy: 0.8326 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1645/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3944 - accuracy: 0.8326 - val_loss: 0.5957 - val_accuracy: 0.7013
    Epoch 1646/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3946 - accuracy: 0.8304 - val_loss: 0.5981 - val_accuracy: 0.6818
    Epoch 1647/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3945 - accuracy: 0.8304 - val_loss: 0.5971 - val_accuracy: 0.6948
    Epoch 1648/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3941 - accuracy: 0.8326 - val_loss: 0.5962 - val_accuracy: 0.7013
    Epoch 1649/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3943 - accuracy: 0.8326 - val_loss: 0.5985 - val_accuracy: 0.6883
    Epoch 1650/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3941 - accuracy: 0.8326 - val_loss: 0.5974 - val_accuracy: 0.6948
    Epoch 1651/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.3937 - accuracy: 0.8348 - val_loss: 0.5954 - val_accuracy: 0.7013
    Epoch 1652/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3939 - accuracy: 0.8304 - val_loss: 0.5970 - val_accuracy: 0.6883
    Epoch 1653/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3936 - accuracy: 0.8304 - val_loss: 0.5990 - val_accuracy: 0.6883
    Epoch 1654/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3935 - accuracy: 0.8283 - val_loss: 0.5979 - val_accuracy: 0.7013
    Epoch 1655/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3937 - accuracy: 0.8304 - val_loss: 0.5988 - val_accuracy: 0.6948
    Epoch 1656/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3931 - accuracy: 0.8326 - val_loss: 0.5988 - val_accuracy: 0.6883
    Epoch 1657/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3932 - accuracy: 0.8304 - val_loss: 0.5951 - val_accuracy: 0.7013
    Epoch 1658/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3932 - accuracy: 0.8304 - val_loss: 0.5957 - val_accuracy: 0.6948
    Epoch 1659/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3928 - accuracy: 0.8348 - val_loss: 0.5971 - val_accuracy: 0.6883
    Epoch 1660/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.3928 - accuracy: 0.8304 - val_loss: 0.5958 - val_accuracy: 0.7013
    Epoch 1661/2000
    1/1 [==============================] - 0s 85ms/step - loss: 0.3927 - accuracy: 0.8304 - val_loss: 0.5963 - val_accuracy: 0.6948
    Epoch 1662/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3923 - accuracy: 0.8326 - val_loss: 0.5977 - val_accuracy: 0.6883
    Epoch 1663/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3924 - accuracy: 0.8304 - val_loss: 0.5962 - val_accuracy: 0.6948
    Epoch 1664/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3922 - accuracy: 0.8326 - val_loss: 0.5974 - val_accuracy: 0.6948
    Epoch 1665/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3920 - accuracy: 0.8326 - val_loss: 0.5998 - val_accuracy: 0.6948
    Epoch 1666/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3920 - accuracy: 0.8326 - val_loss: 0.5989 - val_accuracy: 0.6948
    Epoch 1667/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.3918 - accuracy: 0.8370 - val_loss: 0.5978 - val_accuracy: 0.6948
    Epoch 1668/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3917 - accuracy: 0.8348 - val_loss: 0.5980 - val_accuracy: 0.6883
    Epoch 1669/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3916 - accuracy: 0.8304 - val_loss: 0.5962 - val_accuracy: 0.6948
    Epoch 1670/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.3914 - accuracy: 0.8348 - val_loss: 0.5954 - val_accuracy: 0.7013
    Epoch 1671/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3914 - accuracy: 0.8348 - val_loss: 0.5970 - val_accuracy: 0.6948
    Epoch 1672/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3911 - accuracy: 0.8348 - val_loss: 0.5974 - val_accuracy: 0.6948
    Epoch 1673/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3911 - accuracy: 0.8348 - val_loss: 0.5966 - val_accuracy: 0.6948
    Epoch 1674/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3911 - accuracy: 0.8304 - val_loss: 0.5981 - val_accuracy: 0.6948
    Epoch 1675/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3908 - accuracy: 0.8304 - val_loss: 0.5989 - val_accuracy: 0.6948
    Epoch 1676/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3907 - accuracy: 0.8304 - val_loss: 0.5975 - val_accuracy: 0.7013
    Epoch 1677/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3907 - accuracy: 0.8304 - val_loss: 0.5984 - val_accuracy: 0.6948
    Epoch 1678/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3903 - accuracy: 0.8348 - val_loss: 0.5985 - val_accuracy: 0.6948
    Epoch 1679/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3903 - accuracy: 0.8326 - val_loss: 0.5965 - val_accuracy: 0.6948
    Epoch 1680/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3902 - accuracy: 0.8348 - val_loss: 0.5972 - val_accuracy: 0.6948
    Epoch 1681/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3899 - accuracy: 0.8348 - val_loss: 0.5988 - val_accuracy: 0.6883
    Epoch 1682/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3900 - accuracy: 0.8326 - val_loss: 0.5974 - val_accuracy: 0.6948
    Epoch 1683/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3898 - accuracy: 0.8348 - val_loss: 0.5974 - val_accuracy: 0.6948
    Epoch 1684/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3896 - accuracy: 0.8348 - val_loss: 0.5980 - val_accuracy: 0.6883
    Epoch 1685/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3896 - accuracy: 0.8326 - val_loss: 0.5972 - val_accuracy: 0.6948
    Epoch 1686/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3894 - accuracy: 0.8326 - val_loss: 0.5980 - val_accuracy: 0.6948
    Epoch 1687/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3893 - accuracy: 0.8348 - val_loss: 0.5991 - val_accuracy: 0.6948
    Epoch 1688/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3892 - accuracy: 0.8370 - val_loss: 0.5977 - val_accuracy: 0.6948
    Epoch 1689/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3890 - accuracy: 0.8370 - val_loss: 0.5982 - val_accuracy: 0.6948
    Epoch 1690/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3888 - accuracy: 0.8348 - val_loss: 0.5987 - val_accuracy: 0.6948
    Epoch 1691/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3888 - accuracy: 0.8370 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1692/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3887 - accuracy: 0.8348 - val_loss: 0.5983 - val_accuracy: 0.6948
    Epoch 1693/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3885 - accuracy: 0.8370 - val_loss: 0.5985 - val_accuracy: 0.6948
    Epoch 1694/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3884 - accuracy: 0.8348 - val_loss: 0.5980 - val_accuracy: 0.6948
    Epoch 1695/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3883 - accuracy: 0.8348 - val_loss: 0.5981 - val_accuracy: 0.6948
    Epoch 1696/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3881 - accuracy: 0.8370 - val_loss: 0.5982 - val_accuracy: 0.6948
    Epoch 1697/2000
    1/1 [==============================] - 0s 64ms/step - loss: 0.3880 - accuracy: 0.8370 - val_loss: 0.5973 - val_accuracy: 0.6883
    Epoch 1698/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3880 - accuracy: 0.8348 - val_loss: 0.5981 - val_accuracy: 0.6883
    Epoch 1699/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3878 - accuracy: 0.8348 - val_loss: 0.5997 - val_accuracy: 0.6948
    Epoch 1700/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3878 - accuracy: 0.8391 - val_loss: 0.5982 - val_accuracy: 0.7078
    Epoch 1701/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3878 - accuracy: 0.8348 - val_loss: 0.5994 - val_accuracy: 0.6948
    Epoch 1702/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3874 - accuracy: 0.8370 - val_loss: 0.5993 - val_accuracy: 0.6948
    Epoch 1703/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.3873 - accuracy: 0.8370 - val_loss: 0.5979 - val_accuracy: 0.6948
    Epoch 1704/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3873 - accuracy: 0.8370 - val_loss: 0.5990 - val_accuracy: 0.6883
    Epoch 1705/2000
    1/1 [==============================] - 0s 75ms/step - loss: 0.3871 - accuracy: 0.8370 - val_loss: 0.5994 - val_accuracy: 0.6883
    Epoch 1706/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3869 - accuracy: 0.8370 - val_loss: 0.5985 - val_accuracy: 0.6883
    Epoch 1707/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3869 - accuracy: 0.8370 - val_loss: 0.5991 - val_accuracy: 0.6948
    Epoch 1708/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3867 - accuracy: 0.8370 - val_loss: 0.5991 - val_accuracy: 0.6948
    Epoch 1709/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3866 - accuracy: 0.8370 - val_loss: 0.5973 - val_accuracy: 0.7078
    Epoch 1710/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3867 - accuracy: 0.8370 - val_loss: 0.5988 - val_accuracy: 0.6883
    Epoch 1711/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3864 - accuracy: 0.8391 - val_loss: 0.5997 - val_accuracy: 0.6883
    Epoch 1712/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3863 - accuracy: 0.8391 - val_loss: 0.5999 - val_accuracy: 0.6883
    Epoch 1713/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3862 - accuracy: 0.8370 - val_loss: 0.6001 - val_accuracy: 0.6948
    Epoch 1714/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3860 - accuracy: 0.8370 - val_loss: 0.5995 - val_accuracy: 0.6948
    Epoch 1715/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3859 - accuracy: 0.8391 - val_loss: 0.5974 - val_accuracy: 0.6883
    Epoch 1716/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3859 - accuracy: 0.8370 - val_loss: 0.5981 - val_accuracy: 0.6883
    Epoch 1717/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3857 - accuracy: 0.8370 - val_loss: 0.5995 - val_accuracy: 0.6948
    Epoch 1718/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3855 - accuracy: 0.8391 - val_loss: 0.6000 - val_accuracy: 0.6948
    Epoch 1719/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.3855 - accuracy: 0.8370 - val_loss: 0.6000 - val_accuracy: 0.6883
    Epoch 1720/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3853 - accuracy: 0.8370 - val_loss: 0.5999 - val_accuracy: 0.6883
    Epoch 1721/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3853 - accuracy: 0.8391 - val_loss: 0.5973 - val_accuracy: 0.6948
    Epoch 1722/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3852 - accuracy: 0.8370 - val_loss: 0.5982 - val_accuracy: 0.6948
    Epoch 1723/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3849 - accuracy: 0.8413 - val_loss: 0.5998 - val_accuracy: 0.6948
    Epoch 1724/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3850 - accuracy: 0.8413 - val_loss: 0.5991 - val_accuracy: 0.7078
    Epoch 1725/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3849 - accuracy: 0.8370 - val_loss: 0.6005 - val_accuracy: 0.6883
    Epoch 1726/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3845 - accuracy: 0.8413 - val_loss: 0.5998 - val_accuracy: 0.6818
    Epoch 1727/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3845 - accuracy: 0.8413 - val_loss: 0.5983 - val_accuracy: 0.6818
    Epoch 1728/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3844 - accuracy: 0.8391 - val_loss: 0.5991 - val_accuracy: 0.6818
    Epoch 1729/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3842 - accuracy: 0.8413 - val_loss: 0.5998 - val_accuracy: 0.6948
    Epoch 1730/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.3841 - accuracy: 0.8391 - val_loss: 0.5992 - val_accuracy: 0.6948
    Epoch 1731/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3839 - accuracy: 0.8413 - val_loss: 0.5997 - val_accuracy: 0.6948
    Epoch 1732/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.3838 - accuracy: 0.8413 - val_loss: 0.5998 - val_accuracy: 0.6883
    Epoch 1733/2000
    1/1 [==============================] - 0s 57ms/step - loss: 0.3837 - accuracy: 0.8413 - val_loss: 0.5995 - val_accuracy: 0.6883
    Epoch 1734/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3835 - accuracy: 0.8413 - val_loss: 0.6001 - val_accuracy: 0.6883
    Epoch 1735/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3834 - accuracy: 0.8391 - val_loss: 0.5994 - val_accuracy: 0.6883
    Epoch 1736/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3833 - accuracy: 0.8391 - val_loss: 0.5998 - val_accuracy: 0.6818
    Epoch 1737/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3831 - accuracy: 0.8391 - val_loss: 0.5996 - val_accuracy: 0.6818
    Epoch 1738/2000
    1/1 [==============================] - 0s 69ms/step - loss: 0.3830 - accuracy: 0.8413 - val_loss: 0.5998 - val_accuracy: 0.6818
    Epoch 1739/2000
    1/1 [==============================] - 0s 63ms/step - loss: 0.3829 - accuracy: 0.8391 - val_loss: 0.5996 - val_accuracy: 0.6883
    Epoch 1740/2000
    1/1 [==============================] - 0s 53ms/step - loss: 0.3828 - accuracy: 0.8413 - val_loss: 0.6003 - val_accuracy: 0.6818
    Epoch 1741/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3827 - accuracy: 0.8435 - val_loss: 0.6000 - val_accuracy: 0.6818
    Epoch 1742/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3826 - accuracy: 0.8413 - val_loss: 0.6009 - val_accuracy: 0.6818
    Epoch 1743/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3824 - accuracy: 0.8457 - val_loss: 0.6012 - val_accuracy: 0.6818
    Epoch 1744/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3823 - accuracy: 0.8435 - val_loss: 0.6001 - val_accuracy: 0.6883
    Epoch 1745/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3822 - accuracy: 0.8413 - val_loss: 0.6000 - val_accuracy: 0.6818
    Epoch 1746/2000
    1/1 [==============================] - 0s 79ms/step - loss: 0.3820 - accuracy: 0.8413 - val_loss: 0.6000 - val_accuracy: 0.6818
    Epoch 1747/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3820 - accuracy: 0.8413 - val_loss: 0.5994 - val_accuracy: 0.7013
    Epoch 1748/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.3819 - accuracy: 0.8413 - val_loss: 0.6012 - val_accuracy: 0.6818
    Epoch 1749/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3818 - accuracy: 0.8435 - val_loss: 0.6007 - val_accuracy: 0.6883
    Epoch 1750/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3816 - accuracy: 0.8413 - val_loss: 0.6000 - val_accuracy: 0.6883
    Epoch 1751/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3815 - accuracy: 0.8413 - val_loss: 0.6016 - val_accuracy: 0.6818
    Epoch 1752/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3815 - accuracy: 0.8413 - val_loss: 0.6001 - val_accuracy: 0.6883
    Epoch 1753/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3812 - accuracy: 0.8413 - val_loss: 0.6006 - val_accuracy: 0.6883
    Epoch 1754/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3811 - accuracy: 0.8413 - val_loss: 0.6005 - val_accuracy: 0.6818
    Epoch 1755/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3809 - accuracy: 0.8435 - val_loss: 0.5999 - val_accuracy: 0.6883
    Epoch 1756/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3808 - accuracy: 0.8413 - val_loss: 0.6005 - val_accuracy: 0.6883
    Epoch 1757/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3807 - accuracy: 0.8413 - val_loss: 0.6006 - val_accuracy: 0.6883
    Epoch 1758/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3806 - accuracy: 0.8435 - val_loss: 0.6024 - val_accuracy: 0.6818
    Epoch 1759/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3805 - accuracy: 0.8457 - val_loss: 0.6014 - val_accuracy: 0.6883
    Epoch 1760/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.3804 - accuracy: 0.8435 - val_loss: 0.6012 - val_accuracy: 0.6883
    Epoch 1761/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3802 - accuracy: 0.8457 - val_loss: 0.6015 - val_accuracy: 0.6883
    Epoch 1762/2000
    1/1 [==============================] - 0s 66ms/step - loss: 0.3801 - accuracy: 0.8413 - val_loss: 0.6000 - val_accuracy: 0.6948
    Epoch 1763/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3801 - accuracy: 0.8391 - val_loss: 0.6015 - val_accuracy: 0.6818
    Epoch 1764/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3800 - accuracy: 0.8413 - val_loss: 0.6010 - val_accuracy: 0.6883
    Epoch 1765/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3797 - accuracy: 0.8435 - val_loss: 0.6013 - val_accuracy: 0.7013
    Epoch 1766/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3797 - accuracy: 0.8457 - val_loss: 0.6028 - val_accuracy: 0.6948
    Epoch 1767/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3796 - accuracy: 0.8457 - val_loss: 0.6016 - val_accuracy: 0.6883
    Epoch 1768/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3794 - accuracy: 0.8413 - val_loss: 0.6007 - val_accuracy: 0.6883
    Epoch 1769/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.3793 - accuracy: 0.8413 - val_loss: 0.6018 - val_accuracy: 0.6883
    Epoch 1770/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3793 - accuracy: 0.8391 - val_loss: 0.6003 - val_accuracy: 0.6948
    Epoch 1771/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3791 - accuracy: 0.8413 - val_loss: 0.6017 - val_accuracy: 0.6883
    Epoch 1772/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3789 - accuracy: 0.8435 - val_loss: 0.6033 - val_accuracy: 0.6818
    Epoch 1773/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3789 - accuracy: 0.8457 - val_loss: 0.6016 - val_accuracy: 0.6948
    Epoch 1774/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3788 - accuracy: 0.8413 - val_loss: 0.6026 - val_accuracy: 0.6818
    Epoch 1775/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3786 - accuracy: 0.8457 - val_loss: 0.6026 - val_accuracy: 0.6818
    Epoch 1776/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3785 - accuracy: 0.8435 - val_loss: 0.6004 - val_accuracy: 0.6883
    Epoch 1777/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3786 - accuracy: 0.8348 - val_loss: 0.6022 - val_accuracy: 0.6818
    Epoch 1778/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.3784 - accuracy: 0.8391 - val_loss: 0.6008 - val_accuracy: 0.6883
    Epoch 1779/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3780 - accuracy: 0.8413 - val_loss: 0.6010 - val_accuracy: 0.6883
    Epoch 1780/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3780 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6818
    Epoch 1781/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3779 - accuracy: 0.8435 - val_loss: 0.6030 - val_accuracy: 0.7013
    Epoch 1782/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3777 - accuracy: 0.8457 - val_loss: 0.6024 - val_accuracy: 0.7013
    Epoch 1783/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3776 - accuracy: 0.8457 - val_loss: 0.6024 - val_accuracy: 0.6818
    Epoch 1784/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3775 - accuracy: 0.8435 - val_loss: 0.6008 - val_accuracy: 0.6883
    Epoch 1785/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3774 - accuracy: 0.8435 - val_loss: 0.6009 - val_accuracy: 0.6883
    Epoch 1786/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3772 - accuracy: 0.8435 - val_loss: 0.6024 - val_accuracy: 0.6883
    Epoch 1787/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3772 - accuracy: 0.8435 - val_loss: 0.6014 - val_accuracy: 0.6883
    Epoch 1788/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3770 - accuracy: 0.8435 - val_loss: 0.6016 - val_accuracy: 0.6883
    Epoch 1789/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3768 - accuracy: 0.8435 - val_loss: 0.6015 - val_accuracy: 0.6883
    Epoch 1790/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3767 - accuracy: 0.8435 - val_loss: 0.6016 - val_accuracy: 0.6883
    Epoch 1791/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3766 - accuracy: 0.8435 - val_loss: 0.6029 - val_accuracy: 0.6948
    Epoch 1792/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3765 - accuracy: 0.8457 - val_loss: 0.6029 - val_accuracy: 0.6948
    Epoch 1793/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3764 - accuracy: 0.8457 - val_loss: 0.6021 - val_accuracy: 0.6883
    Epoch 1794/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3763 - accuracy: 0.8457 - val_loss: 0.6031 - val_accuracy: 0.6753
    Epoch 1795/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3763 - accuracy: 0.8413 - val_loss: 0.6013 - val_accuracy: 0.6883
    Epoch 1796/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3762 - accuracy: 0.8413 - val_loss: 0.6026 - val_accuracy: 0.6948
    Epoch 1797/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3759 - accuracy: 0.8413 - val_loss: 0.6032 - val_accuracy: 0.7013
    Epoch 1798/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3758 - accuracy: 0.8413 - val_loss: 0.6018 - val_accuracy: 0.6948
    Epoch 1799/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3757 - accuracy: 0.8435 - val_loss: 0.6022 - val_accuracy: 0.6883
    Epoch 1800/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3755 - accuracy: 0.8435 - val_loss: 0.6022 - val_accuracy: 0.6883
    Epoch 1801/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3754 - accuracy: 0.8435 - val_loss: 0.6014 - val_accuracy: 0.6948
    Epoch 1802/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3754 - accuracy: 0.8413 - val_loss: 0.6032 - val_accuracy: 0.6883
    Epoch 1803/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3752 - accuracy: 0.8435 - val_loss: 0.6031 - val_accuracy: 0.6948
    Epoch 1804/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3750 - accuracy: 0.8457 - val_loss: 0.6037 - val_accuracy: 0.6883
    Epoch 1805/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3749 - accuracy: 0.8435 - val_loss: 0.6038 - val_accuracy: 0.6818
    Epoch 1806/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3747 - accuracy: 0.8435 - val_loss: 0.6023 - val_accuracy: 0.6883
    Epoch 1807/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3746 - accuracy: 0.8435 - val_loss: 0.6016 - val_accuracy: 0.6948
    Epoch 1808/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3745 - accuracy: 0.8435 - val_loss: 0.6015 - val_accuracy: 0.6883
    Epoch 1809/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3743 - accuracy: 0.8435 - val_loss: 0.6022 - val_accuracy: 0.6883
    Epoch 1810/2000
    1/1 [==============================] - 0s 80ms/step - loss: 0.3742 - accuracy: 0.8435 - val_loss: 0.6025 - val_accuracy: 0.6883
    Epoch 1811/2000
    1/1 [==============================] - 0s 77ms/step - loss: 0.3741 - accuracy: 0.8435 - val_loss: 0.6038 - val_accuracy: 0.6818
    Epoch 1812/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.3740 - accuracy: 0.8435 - val_loss: 0.6029 - val_accuracy: 0.6883
    Epoch 1813/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3739 - accuracy: 0.8435 - val_loss: 0.6028 - val_accuracy: 0.6818
    Epoch 1814/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3737 - accuracy: 0.8435 - val_loss: 0.6027 - val_accuracy: 0.6818
    Epoch 1815/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3737 - accuracy: 0.8435 - val_loss: 0.6013 - val_accuracy: 0.6883
    Epoch 1816/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3736 - accuracy: 0.8435 - val_loss: 0.6033 - val_accuracy: 0.6883
    Epoch 1817/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3734 - accuracy: 0.8457 - val_loss: 0.6021 - val_accuracy: 0.6883
    Epoch 1818/2000
    1/1 [==============================] - 0s 56ms/step - loss: 0.3733 - accuracy: 0.8435 - val_loss: 0.6028 - val_accuracy: 0.6883
    Epoch 1819/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3731 - accuracy: 0.8435 - val_loss: 0.6033 - val_accuracy: 0.6818
    Epoch 1820/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3730 - accuracy: 0.8435 - val_loss: 0.6028 - val_accuracy: 0.6948
    Epoch 1821/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3729 - accuracy: 0.8435 - val_loss: 0.6034 - val_accuracy: 0.6883
    Epoch 1822/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3727 - accuracy: 0.8435 - val_loss: 0.6024 - val_accuracy: 0.6883
    Epoch 1823/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3726 - accuracy: 0.8435 - val_loss: 0.6025 - val_accuracy: 0.6883
    Epoch 1824/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3724 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6883
    Epoch 1825/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3723 - accuracy: 0.8457 - val_loss: 0.6033 - val_accuracy: 0.6883
    Epoch 1826/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3722 - accuracy: 0.8457 - val_loss: 0.6031 - val_accuracy: 0.6883
    Epoch 1827/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3721 - accuracy: 0.8435 - val_loss: 0.6038 - val_accuracy: 0.6818
    Epoch 1828/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3720 - accuracy: 0.8457 - val_loss: 0.6018 - val_accuracy: 0.6883
    Epoch 1829/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3720 - accuracy: 0.8435 - val_loss: 0.6023 - val_accuracy: 0.6818
    Epoch 1830/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3718 - accuracy: 0.8435 - val_loss: 0.6040 - val_accuracy: 0.6818
    Epoch 1831/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3717 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6883
    Epoch 1832/2000
    1/1 [==============================] - 0s 73ms/step - loss: 0.3717 - accuracy: 0.8435 - val_loss: 0.6043 - val_accuracy: 0.6883
    Epoch 1833/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3714 - accuracy: 0.8435 - val_loss: 0.6035 - val_accuracy: 0.6818
    Epoch 1834/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3712 - accuracy: 0.8435 - val_loss: 0.6016 - val_accuracy: 0.6883
    Epoch 1835/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3713 - accuracy: 0.8413 - val_loss: 0.6030 - val_accuracy: 0.6818
    Epoch 1836/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3711 - accuracy: 0.8435 - val_loss: 0.6028 - val_accuracy: 0.6818
    Epoch 1837/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3708 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6818
    Epoch 1838/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3708 - accuracy: 0.8435 - val_loss: 0.6049 - val_accuracy: 0.6818
    Epoch 1839/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3707 - accuracy: 0.8457 - val_loss: 0.6033 - val_accuracy: 0.6818
    Epoch 1840/2000
    1/1 [==============================] - 0s 61ms/step - loss: 0.3705 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6818
    Epoch 1841/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3703 - accuracy: 0.8435 - val_loss: 0.6032 - val_accuracy: 0.6818
    Epoch 1842/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3703 - accuracy: 0.8457 - val_loss: 0.6022 - val_accuracy: 0.6883
    Epoch 1843/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3702 - accuracy: 0.8435 - val_loss: 0.6033 - val_accuracy: 0.6883
    Epoch 1844/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.3700 - accuracy: 0.8457 - val_loss: 0.6029 - val_accuracy: 0.6818
    Epoch 1845/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.3698 - accuracy: 0.8435 - val_loss: 0.6039 - val_accuracy: 0.6818
    Epoch 1846/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3697 - accuracy: 0.8435 - val_loss: 0.6054 - val_accuracy: 0.6818
    Epoch 1847/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3696 - accuracy: 0.8435 - val_loss: 0.6038 - val_accuracy: 0.6883
    Epoch 1848/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3695 - accuracy: 0.8457 - val_loss: 0.6037 - val_accuracy: 0.6818
    Epoch 1849/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3694 - accuracy: 0.8457 - val_loss: 0.6016 - val_accuracy: 0.6818
    Epoch 1850/2000
    1/1 [==============================] - 0s 74ms/step - loss: 0.3692 - accuracy: 0.8435 - val_loss: 0.6016 - val_accuracy: 0.6883
    Epoch 1851/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3691 - accuracy: 0.8435 - val_loss: 0.6028 - val_accuracy: 0.6883
    Epoch 1852/2000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3690 - accuracy: 0.8457 - val_loss: 0.6028 - val_accuracy: 0.6818
    Epoch 1853/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3689 - accuracy: 0.8435 - val_loss: 0.6045 - val_accuracy: 0.6883
    Epoch 1854/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3687 - accuracy: 0.8457 - val_loss: 0.6042 - val_accuracy: 0.6883
    Epoch 1855/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3685 - accuracy: 0.8457 - val_loss: 0.6042 - val_accuracy: 0.6818
    Epoch 1856/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3684 - accuracy: 0.8457 - val_loss: 0.6040 - val_accuracy: 0.6818
    Epoch 1857/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3683 - accuracy: 0.8457 - val_loss: 0.6030 - val_accuracy: 0.6883
    Epoch 1858/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3682 - accuracy: 0.8457 - val_loss: 0.6034 - val_accuracy: 0.6818
    Epoch 1859/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3680 - accuracy: 0.8457 - val_loss: 0.6034 - val_accuracy: 0.6818
    Epoch 1860/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3678 - accuracy: 0.8457 - val_loss: 0.6022 - val_accuracy: 0.6948
    Epoch 1861/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3679 - accuracy: 0.8457 - val_loss: 0.6042 - val_accuracy: 0.6883
    Epoch 1862/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3678 - accuracy: 0.8413 - val_loss: 0.6031 - val_accuracy: 0.6883
    Epoch 1863/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3676 - accuracy: 0.8435 - val_loss: 0.6048 - val_accuracy: 0.6818
    Epoch 1864/2000
    1/1 [==============================] - 0s 54ms/step - loss: 0.3674 - accuracy: 0.8457 - val_loss: 0.6046 - val_accuracy: 0.6818
    Epoch 1865/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.3673 - accuracy: 0.8457 - val_loss: 0.6025 - val_accuracy: 0.6948
    Epoch 1866/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3672 - accuracy: 0.8435 - val_loss: 0.6026 - val_accuracy: 0.6883
    Epoch 1867/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3670 - accuracy: 0.8457 - val_loss: 0.6022 - val_accuracy: 0.6883
    Epoch 1868/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3668 - accuracy: 0.8457 - val_loss: 0.6027 - val_accuracy: 0.6818
    Epoch 1869/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3667 - accuracy: 0.8435 - val_loss: 0.6049 - val_accuracy: 0.6883
    Epoch 1870/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3665 - accuracy: 0.8478 - val_loss: 0.6044 - val_accuracy: 0.6948
    Epoch 1871/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3664 - accuracy: 0.8457 - val_loss: 0.6044 - val_accuracy: 0.6883
    Epoch 1872/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3662 - accuracy: 0.8478 - val_loss: 0.6031 - val_accuracy: 0.6818
    Epoch 1873/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3661 - accuracy: 0.8457 - val_loss: 0.6026 - val_accuracy: 0.6883
    Epoch 1874/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3659 - accuracy: 0.8457 - val_loss: 0.6030 - val_accuracy: 0.6883
    Epoch 1875/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3659 - accuracy: 0.8478 - val_loss: 0.6028 - val_accuracy: 0.6818
    Epoch 1876/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3656 - accuracy: 0.8457 - val_loss: 0.6037 - val_accuracy: 0.6818
    Epoch 1877/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3656 - accuracy: 0.8435 - val_loss: 0.6038 - val_accuracy: 0.6883
    Epoch 1878/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3654 - accuracy: 0.8457 - val_loss: 0.6037 - val_accuracy: 0.6883
    Epoch 1879/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3653 - accuracy: 0.8457 - val_loss: 0.6037 - val_accuracy: 0.6818
    Epoch 1880/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3650 - accuracy: 0.8457 - val_loss: 0.6032 - val_accuracy: 0.6818
    Epoch 1881/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3649 - accuracy: 0.8457 - val_loss: 0.6033 - val_accuracy: 0.6883
    Epoch 1882/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3647 - accuracy: 0.8457 - val_loss: 0.6038 - val_accuracy: 0.6883
    Epoch 1883/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3647 - accuracy: 0.8478 - val_loss: 0.6040 - val_accuracy: 0.6818
    Epoch 1884/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3644 - accuracy: 0.8435 - val_loss: 0.6037 - val_accuracy: 0.6818
    Epoch 1885/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3643 - accuracy: 0.8435 - val_loss: 0.6035 - val_accuracy: 0.6883
    Epoch 1886/2000
    1/1 [==============================] - 0s 65ms/step - loss: 0.3641 - accuracy: 0.8457 - val_loss: 0.6032 - val_accuracy: 0.6883
    Epoch 1887/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3640 - accuracy: 0.8457 - val_loss: 0.6028 - val_accuracy: 0.6818
    Epoch 1888/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3638 - accuracy: 0.8435 - val_loss: 0.6033 - val_accuracy: 0.6818
    Epoch 1889/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3637 - accuracy: 0.8457 - val_loss: 0.6028 - val_accuracy: 0.6883
    Epoch 1890/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3636 - accuracy: 0.8435 - val_loss: 0.6039 - val_accuracy: 0.6818
    Epoch 1891/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3634 - accuracy: 0.8435 - val_loss: 0.6030 - val_accuracy: 0.6818
    Epoch 1892/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3634 - accuracy: 0.8413 - val_loss: 0.6052 - val_accuracy: 0.6883
    Epoch 1893/2000
    1/1 [==============================] - 0s 70ms/step - loss: 0.3632 - accuracy: 0.8435 - val_loss: 0.6041 - val_accuracy: 0.6883
    Epoch 1894/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3629 - accuracy: 0.8413 - val_loss: 0.6046 - val_accuracy: 0.6883
    Epoch 1895/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3627 - accuracy: 0.8457 - val_loss: 0.6047 - val_accuracy: 0.6818
    Epoch 1896/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3626 - accuracy: 0.8435 - val_loss: 0.6042 - val_accuracy: 0.6883
    Epoch 1897/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3625 - accuracy: 0.8435 - val_loss: 0.6058 - val_accuracy: 0.6883
    Epoch 1898/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3624 - accuracy: 0.8435 - val_loss: 0.6040 - val_accuracy: 0.6883
    Epoch 1899/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3623 - accuracy: 0.8413 - val_loss: 0.6051 - val_accuracy: 0.6883
    Epoch 1900/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3621 - accuracy: 0.8413 - val_loss: 0.6047 - val_accuracy: 0.6883
    Epoch 1901/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3619 - accuracy: 0.8435 - val_loss: 0.6047 - val_accuracy: 0.6883
    Epoch 1902/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3618 - accuracy: 0.8457 - val_loss: 0.6055 - val_accuracy: 0.6883
    Epoch 1903/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3617 - accuracy: 0.8435 - val_loss: 0.6045 - val_accuracy: 0.6883
    Epoch 1904/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3615 - accuracy: 0.8435 - val_loss: 0.6047 - val_accuracy: 0.6818
    Epoch 1905/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3613 - accuracy: 0.8435 - val_loss: 0.6060 - val_accuracy: 0.6818
    Epoch 1906/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3612 - accuracy: 0.8435 - val_loss: 0.6049 - val_accuracy: 0.6883
    Epoch 1907/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3613 - accuracy: 0.8391 - val_loss: 0.6066 - val_accuracy: 0.6883
    Epoch 1908/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3611 - accuracy: 0.8457 - val_loss: 0.6042 - val_accuracy: 0.6883
    Epoch 1909/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3610 - accuracy: 0.8413 - val_loss: 0.6061 - val_accuracy: 0.6883
    Epoch 1910/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3608 - accuracy: 0.8435 - val_loss: 0.6043 - val_accuracy: 0.6883
    Epoch 1911/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3606 - accuracy: 0.8435 - val_loss: 0.6055 - val_accuracy: 0.6818
    Epoch 1912/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3604 - accuracy: 0.8413 - val_loss: 0.6054 - val_accuracy: 0.6818
    Epoch 1913/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3602 - accuracy: 0.8413 - val_loss: 0.6052 - val_accuracy: 0.6883
    Epoch 1914/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3602 - accuracy: 0.8500 - val_loss: 0.6059 - val_accuracy: 0.6818
    Epoch 1915/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3600 - accuracy: 0.8435 - val_loss: 0.6055 - val_accuracy: 0.6818
    Epoch 1916/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.3599 - accuracy: 0.8435 - val_loss: 0.6064 - val_accuracy: 0.6818
    Epoch 1917/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.3597 - accuracy: 0.8457 - val_loss: 0.6068 - val_accuracy: 0.6883
    Epoch 1918/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3596 - accuracy: 0.8435 - val_loss: 0.6045 - val_accuracy: 0.6948
    Epoch 1919/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3596 - accuracy: 0.8391 - val_loss: 0.6058 - val_accuracy: 0.6948
    Epoch 1920/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3597 - accuracy: 0.8413 - val_loss: 0.6036 - val_accuracy: 0.6948
    Epoch 1921/2000
    1/1 [==============================] - 0s 73ms/step - loss: 0.3596 - accuracy: 0.8435 - val_loss: 0.6062 - val_accuracy: 0.6948
    Epoch 1922/2000
    1/1 [==============================] - 0s 67ms/step - loss: 0.3592 - accuracy: 0.8457 - val_loss: 0.6062 - val_accuracy: 0.6883
    Epoch 1923/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3589 - accuracy: 0.8457 - val_loss: 0.6064 - val_accuracy: 0.6948
    Epoch 1924/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3589 - accuracy: 0.8457 - val_loss: 0.6078 - val_accuracy: 0.7013
    Epoch 1925/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3590 - accuracy: 0.8435 - val_loss: 0.6037 - val_accuracy: 0.6948
    Epoch 1926/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3590 - accuracy: 0.8435 - val_loss: 0.6047 - val_accuracy: 0.6948
    Epoch 1927/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3587 - accuracy: 0.8435 - val_loss: 0.6052 - val_accuracy: 0.6948
    Epoch 1928/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3583 - accuracy: 0.8435 - val_loss: 0.6069 - val_accuracy: 0.7013
    Epoch 1929/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3584 - accuracy: 0.8457 - val_loss: 0.6111 - val_accuracy: 0.6948
    Epoch 1930/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3588 - accuracy: 0.8478 - val_loss: 0.6077 - val_accuracy: 0.7013
    Epoch 1931/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3583 - accuracy: 0.8478 - val_loss: 0.6074 - val_accuracy: 0.6948
    Epoch 1932/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3578 - accuracy: 0.8478 - val_loss: 0.6065 - val_accuracy: 0.6948
    Epoch 1933/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3578 - accuracy: 0.8413 - val_loss: 0.6052 - val_accuracy: 0.6948
    Epoch 1934/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3579 - accuracy: 0.8391 - val_loss: 0.6088 - val_accuracy: 0.7013
    Epoch 1935/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3577 - accuracy: 0.8457 - val_loss: 0.6066 - val_accuracy: 0.7078
    Epoch 1936/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3575 - accuracy: 0.8478 - val_loss: 0.6072 - val_accuracy: 0.6948
    Epoch 1937/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3572 - accuracy: 0.8478 - val_loss: 0.6060 - val_accuracy: 0.7013
    Epoch 1938/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3570 - accuracy: 0.8478 - val_loss: 0.6068 - val_accuracy: 0.7013
    Epoch 1939/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3569 - accuracy: 0.8478 - val_loss: 0.6083 - val_accuracy: 0.6883
    Epoch 1940/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3567 - accuracy: 0.8478 - val_loss: 0.6084 - val_accuracy: 0.7013
    Epoch 1941/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3566 - accuracy: 0.8435 - val_loss: 0.6072 - val_accuracy: 0.7013
    Epoch 1942/2000
    1/1 [==============================] - 0s 55ms/step - loss: 0.3565 - accuracy: 0.8457 - val_loss: 0.6065 - val_accuracy: 0.7013
    Epoch 1943/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3564 - accuracy: 0.8457 - val_loss: 0.6068 - val_accuracy: 0.7013
    Epoch 1944/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3563 - accuracy: 0.8457 - val_loss: 0.6083 - val_accuracy: 0.7013
    Epoch 1945/2000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3561 - accuracy: 0.8500 - val_loss: 0.6074 - val_accuracy: 0.7143
    Epoch 1946/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3561 - accuracy: 0.8457 - val_loss: 0.6085 - val_accuracy: 0.7013
    Epoch 1947/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3560 - accuracy: 0.8500 - val_loss: 0.6066 - val_accuracy: 0.7143
    Epoch 1948/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3560 - accuracy: 0.8457 - val_loss: 0.6079 - val_accuracy: 0.7013
    Epoch 1949/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3557 - accuracy: 0.8522 - val_loss: 0.6082 - val_accuracy: 0.7078
    Epoch 1950/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3555 - accuracy: 0.8478 - val_loss: 0.6086 - val_accuracy: 0.7078
    Epoch 1951/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3554 - accuracy: 0.8435 - val_loss: 0.6092 - val_accuracy: 0.7078
    Epoch 1952/2000
    1/1 [==============================] - 0s 52ms/step - loss: 0.3553 - accuracy: 0.8522 - val_loss: 0.6070 - val_accuracy: 0.7208
    Epoch 1953/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3553 - accuracy: 0.8500 - val_loss: 0.6077 - val_accuracy: 0.6948
    Epoch 1954/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3551 - accuracy: 0.8522 - val_loss: 0.6076 - val_accuracy: 0.7078
    Epoch 1955/2000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3549 - accuracy: 0.8478 - val_loss: 0.6086 - val_accuracy: 0.7143
    Epoch 1956/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3548 - accuracy: 0.8457 - val_loss: 0.6097 - val_accuracy: 0.7013
    Epoch 1957/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3548 - accuracy: 0.8543 - val_loss: 0.6078 - val_accuracy: 0.7208
    Epoch 1958/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3546 - accuracy: 0.8500 - val_loss: 0.6082 - val_accuracy: 0.7013
    Epoch 1959/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3545 - accuracy: 0.8522 - val_loss: 0.6074 - val_accuracy: 0.7143
    Epoch 1960/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3543 - accuracy: 0.8500 - val_loss: 0.6092 - val_accuracy: 0.7078
    Epoch 1961/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3541 - accuracy: 0.8500 - val_loss: 0.6099 - val_accuracy: 0.7078
    Epoch 1962/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3540 - accuracy: 0.8500 - val_loss: 0.6094 - val_accuracy: 0.7078
    Epoch 1963/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3539 - accuracy: 0.8500 - val_loss: 0.6089 - val_accuracy: 0.7078
    Epoch 1964/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3537 - accuracy: 0.8522 - val_loss: 0.6079 - val_accuracy: 0.7208
    Epoch 1965/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3537 - accuracy: 0.8500 - val_loss: 0.6099 - val_accuracy: 0.7078
    Epoch 1966/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3535 - accuracy: 0.8522 - val_loss: 0.6104 - val_accuracy: 0.7208
    Epoch 1967/2000
    1/1 [==============================] - 0s 62ms/step - loss: 0.3534 - accuracy: 0.8500 - val_loss: 0.6119 - val_accuracy: 0.7143
    Epoch 1968/2000
    1/1 [==============================] - 0s 68ms/step - loss: 0.3533 - accuracy: 0.8522 - val_loss: 0.6109 - val_accuracy: 0.7143
    Epoch 1969/2000
    1/1 [==============================] - 0s 48ms/step - loss: 0.3531 - accuracy: 0.8522 - val_loss: 0.6093 - val_accuracy: 0.7208
    Epoch 1970/2000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3530 - accuracy: 0.8500 - val_loss: 0.6092 - val_accuracy: 0.7078
    Epoch 1971/2000
    1/1 [==============================] - 0s 71ms/step - loss: 0.3528 - accuracy: 0.8543 - val_loss: 0.6088 - val_accuracy: 0.7208
    Epoch 1972/2000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3527 - accuracy: 0.8500 - val_loss: 0.6108 - val_accuracy: 0.7143
    Epoch 1973/2000
    1/1 [==============================] - 0s 49ms/step - loss: 0.3526 - accuracy: 0.8522 - val_loss: 0.6105 - val_accuracy: 0.7208
    Epoch 1974/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3525 - accuracy: 0.8478 - val_loss: 0.6111 - val_accuracy: 0.7078
    Epoch 1975/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3523 - accuracy: 0.8522 - val_loss: 0.6101 - val_accuracy: 0.7143
    Epoch 1976/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3521 - accuracy: 0.8522 - val_loss: 0.6110 - val_accuracy: 0.7078
    Epoch 1977/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3520 - accuracy: 0.8522 - val_loss: 0.6102 - val_accuracy: 0.7208
    Epoch 1978/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3519 - accuracy: 0.8478 - val_loss: 0.6118 - val_accuracy: 0.7143
    Epoch 1979/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3519 - accuracy: 0.8478 - val_loss: 0.6093 - val_accuracy: 0.7143
    Epoch 1980/2000
    1/1 [==============================] - 0s 59ms/step - loss: 0.3519 - accuracy: 0.8500 - val_loss: 0.6111 - val_accuracy: 0.7078
    Epoch 1981/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3517 - accuracy: 0.8522 - val_loss: 0.6098 - val_accuracy: 0.7208
    Epoch 1982/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3514 - accuracy: 0.8478 - val_loss: 0.6104 - val_accuracy: 0.7208
    Epoch 1983/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3512 - accuracy: 0.8478 - val_loss: 0.6102 - val_accuracy: 0.7078
    Epoch 1984/2000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3510 - accuracy: 0.8522 - val_loss: 0.6087 - val_accuracy: 0.7143
    Epoch 1985/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3510 - accuracy: 0.8522 - val_loss: 0.6103 - val_accuracy: 0.7078
    Epoch 1986/2000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3509 - accuracy: 0.8500 - val_loss: 0.6095 - val_accuracy: 0.7143
    Epoch 1987/2000
    1/1 [==============================] - 0s 60ms/step - loss: 0.3507 - accuracy: 0.8500 - val_loss: 0.6121 - val_accuracy: 0.7078
    Epoch 1988/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3506 - accuracy: 0.8543 - val_loss: 0.6108 - val_accuracy: 0.7143
    Epoch 1989/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3505 - accuracy: 0.8500 - val_loss: 0.6124 - val_accuracy: 0.7143
    Epoch 1990/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3503 - accuracy: 0.8522 - val_loss: 0.6101 - val_accuracy: 0.7143
    Epoch 1991/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3502 - accuracy: 0.8500 - val_loss: 0.6117 - val_accuracy: 0.7208
    Epoch 1992/2000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3500 - accuracy: 0.8500 - val_loss: 0.6106 - val_accuracy: 0.7208
    Epoch 1993/2000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3497 - accuracy: 0.8522 - val_loss: 0.6108 - val_accuracy: 0.7273
    Epoch 1994/2000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3495 - accuracy: 0.8543 - val_loss: 0.6113 - val_accuracy: 0.7273
    Epoch 1995/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3495 - accuracy: 0.8565 - val_loss: 0.6095 - val_accuracy: 0.7273
    Epoch 1996/2000
    1/1 [==============================] - 0s 37ms/step - loss: 0.3495 - accuracy: 0.8500 - val_loss: 0.6119 - val_accuracy: 0.7143
    Epoch 1997/2000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3494 - accuracy: 0.8500 - val_loss: 0.6108 - val_accuracy: 0.7208
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
    Epoch 4/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.3482 - accuracy: 0.8500 - val_loss: 0.6130 - val_accuracy: 0.7208
    Epoch 5/2000
    1/1 [==============================] - 0s 31ms/step - loss: 0.3482 - accuracy: 0.8522 - val_loss: 0.6117 - val_accuracy: 0.7273
    Epoch 6/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.3481 - accuracy: 0.8500 - val_loss: 0.6143 - val_accuracy: 0.7273
    Epoch 7/2000
    1/1 [==============================] - 0s 34ms/step - loss: 0.3480 - accuracy: 0.8500 - val_loss: 0.6127 - val_accuracy: 0.7208
    Epoch 8/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3479 - accuracy: 0.8478 - val_loss: 0.6148 - val_accuracy: 0.7273
    Epoch 9/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.3477 - accuracy: 0.8522 - val_loss: 0.6130 - val_accuracy: 0.7273
    Epoch 10/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.3474 - accuracy: 0.8500 - val_loss: 0.6140 - val_accuracy: 0.7273
    Epoch 11/2000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3472 - accuracy: 0.8565 - val_loss: 0.6127 - val_accuracy: 0.7273
    Epoch 12/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.3470 - accuracy: 0.8543 - val_loss: 0.6138 - val_accuracy: 0.7208
    Epoch 13/2000
    1/1 [==============================] - 0s 30ms/step - loss: 0.3470 - accuracy: 0.8587 - val_loss: 0.6143 - val_accuracy: 0.7208
    Epoch 14/2000
    1/1 [==============================] - 0s 33ms/step - loss: 0.3467 - accuracy: 0.8565 - val_loss: 0.6140 - val_accuracy: 0.7273
    Epoch 15/2000
    1/1 [==============================] - 0s 38ms/step - loss: 0.3467 - accuracy: 0.8522 - val_loss: 0.6144 - val_accuracy: 0.7273
    Epoch 16/2000
    1/1 [==============================] - 0s 32ms/step - loss: 0.3466 - accuracy: 0.8565 - val_loss: 0.6130 - val_accuracy: 0.7273
    Epoch 17/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.3466 - accuracy: 0.8500 - val_loss: 0.6151 - val_accuracy: 0.7273
    Epoch 18/2000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3466 - accuracy: 0.8543 - val_loss: 0.6138 - val_accuracy: 0.7273
    Epoch 19/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.3467 - accuracy: 0.8500 - val_loss: 0.6162 - val_accuracy: 0.7208
    Epoch 20/2000
    1/1 [==============================] - 0s 35ms/step - loss: 0.3464 - accuracy: 0.8522 - val_loss: 0.6140 - val_accuracy: 0.7273
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

