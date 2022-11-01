```python
# tensorflow 버전 맞춰줌
pip install tensorflow==2.0.0



```python
#tensorflow 2.0에 최적화 된 code
```


```python
import tensorflow as tf
print(tf.__version__)
```

    2.0.0



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
!ls
```

    drive  sample_data



```python
# 마운트가 제대로 진행되었는지 확인합니다. 아래와 같이 폴더 안에 두 데이터 파일이 포함되어 있는 것으로 출력되어야 합니다.
# file_list: ['Faults27x7_var', 'Faults.NNA']
import os
os.chdir('/content/')
path = "/content/drive/MyDrive/DeepLearning/Park_Professor/DeepLearning(Image)"
file_list = os.listdir(path)

print ("file_list: {}".format(file_list))
```

    file_list: ['Data_Augmentation.ipynb', 'test', 'train']



```python
# Working directory를 설정합니다
import os
os.chdir('/content/drive/MyDrive/DeepLearning/Park_Professor/DeepLearning(Image)')
!ls
```

    Data_Augmentation.ipynb  test  train



```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics

np.random.seed(3)
tf.random.set_seed(3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,     #수평 대칭 이미지를 50% 확률로 만들어 추가합니다.
                                  width_shift_range=0.1,  #전체 크기의 10% 범위에서 좌우로 이동합니다.
                                  height_shift_range=0.1, #마찬가지로 위, 아래로 이동합니다.
                                  #rotation_range=5,
                                  #shear_range=0.7,
                                  #zoom_range=[0.9, 2.2],
                                  #vertical_flip=True,
                                  fill_mode='nearest') 

train_generator = train_datagen.flow_from_directory(
       'train',   #학습셋이 있는 폴더의 위치입니다.
       target_size=(150, 150),
       batch_size=5,
       class_mode='binary')

#테스트 셋은 이미지 부풀리기 과정을 진행하지 않습니다.
test_datagen = ImageDataGenerator(rescale=1./255)  

test_generator = test_datagen.flow_from_directory(
       'test',   #테스트셋이 있는 폴더의 위치입니다.
       target_size=(150, 150),
       batch_size=5,
       class_mode='binary')


# 앞서 배운 CNN 모델을 만들어 적용해 보겠습니다.
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

#모델을 컴파일 합니다. 
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002), metrics=['accuracy'])

#모델을 실행합니다
history = model.fit_generator(
       train_generator,
       steps_per_epoch=100,
       epochs=20,
       validation_data=test_generator,
       validation_steps=10)

#결과를 그래프로 표현하는 부분입니다.
acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))  
plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right') 
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()

```

    Found 160 images belonging to 2 classes.
    Found 120 images belonging to 2 classes.
    Epoch 1/20
    100/100 [==============================] - 63s 628ms/step - loss: 0.6948 - accuracy: 0.5400 - val_loss: 0.6637 - val_accuracy: 0.5200
    Epoch 2/20
    100/100 [==============================] - 21s 205ms/step - loss: 0.6300 - accuracy: 0.6920 - val_loss: 0.5260 - val_accuracy: 0.8000
    Epoch 3/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.5481 - accuracy: 0.7400 - val_loss: 0.3881 - val_accuracy: 0.8600
    Epoch 4/20
    100/100 [==============================] - 21s 210ms/step - loss: 0.3554 - accuracy: 0.8660 - val_loss: 0.2243 - val_accuracy: 0.9000
    Epoch 5/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.2676 - accuracy: 0.9020 - val_loss: 0.1519 - val_accuracy: 0.9800
    Epoch 6/20
    100/100 [==============================] - 23s 229ms/step - loss: 0.2354 - accuracy: 0.9100 - val_loss: 0.1067 - val_accuracy: 0.9800
    Epoch 7/20
    100/100 [==============================] - 21s 205ms/step - loss: 0.1406 - accuracy: 0.9560 - val_loss: 0.1035 - val_accuracy: 0.9400
    Epoch 8/20
    100/100 [==============================] - 21s 208ms/step - loss: 0.1460 - accuracy: 0.9580 - val_loss: 0.0487 - val_accuracy: 1.0000
    Epoch 9/20
    100/100 [==============================] - 21s 210ms/step - loss: 0.0894 - accuracy: 0.9760 - val_loss: 0.0366 - val_accuracy: 1.0000
    Epoch 10/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.0894 - accuracy: 0.9740 - val_loss: 0.0357 - val_accuracy: 1.0000
    Epoch 11/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.0772 - accuracy: 0.9740 - val_loss: 0.0610 - val_accuracy: 0.9600
    Epoch 12/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.1213 - accuracy: 0.9580 - val_loss: 0.0254 - val_accuracy: 1.0000
    Epoch 13/20
    100/100 [==============================] - 21s 208ms/step - loss: 0.0846 - accuracy: 0.9760 - val_loss: 0.0660 - val_accuracy: 0.9600
    Epoch 14/20
    100/100 [==============================] - 28s 281ms/step - loss: 0.0749 - accuracy: 0.9680 - val_loss: 0.0404 - val_accuracy: 0.9800
    Epoch 15/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.0758 - accuracy: 0.9720 - val_loss: 0.0283 - val_accuracy: 1.0000
    Epoch 16/20
    100/100 [==============================] - 21s 207ms/step - loss: 0.0728 - accuracy: 0.9780 - val_loss: 0.0180 - val_accuracy: 1.0000
    Epoch 17/20
    100/100 [==============================] - 21s 214ms/step - loss: 0.0695 - accuracy: 0.9760 - val_loss: 0.0667 - val_accuracy: 0.9600
    Epoch 18/20
    100/100 [==============================] - 21s 206ms/step - loss: 0.0705 - accuracy: 0.9820 - val_loss: 0.0130 - val_accuracy: 1.0000
    Epoch 19/20
    100/100 [==============================] - 21s 208ms/step - loss: 0.0512 - accuracy: 0.9840 - val_loss: 0.0172 - val_accuracy: 1.0000
    Epoch 20/20
    100/100 [==============================] - 21s 208ms/step - loss: 0.0691 - accuracy: 0.9800 - val_loss: 0.0258 - val_accuracy: 0.9800



![output_7_1](https://user-images.githubusercontent.com/87309905/199134974-97cf373a-116a-416d-a8fc-1b35628cad61.png)
    



```python
#결과를 그래프로 표현하는 부분입니다.
acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))  
plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right') 
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
```


![output_8_0](https://user-images.githubusercontent.com/87309905/199134962-3bb05c14-9ea8-4246-92d2-66cb6cee9ae2.png)




```python

```
