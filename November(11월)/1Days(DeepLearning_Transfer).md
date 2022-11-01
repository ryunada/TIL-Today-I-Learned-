```python
pip install tensorflow==2.0.0
```



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
import os
n_cpu=os.cpu_count()
print("The number of cpus: ",n_cpu)
n_thread=n_cpu*2
print("Expected number of threads:",n_thread)
```

    The number of cpus:  2
    Expected number of threads: 4



```python
# 마운트가 제대로 진행되었는지 확인합니다. 아래와 같이 폴더 안에 두 데이터 파일이 포함되어 있는 것으로 출력되어야 합니다.
# file_list: ['Faults27x7_var', 'Faults.NNA']
import os
os.chdir('/content/')
path = "./drive/MyDrive/DeepLearning/Park_Professor/DeepLearning(Image)"
file_list = os.listdir(path)

print ("file_list: {}".format(file_list))
```

    file_list: ['test', 'train', 'Data_Augmentation.ipynb', 'Transfer_Learning.ipynb']



```python
# Working directory를 설정합니다
import os
os.chdir('/content/drive/MyDrive/DeepLearning/Park_Professor/DeepLearning(Image)')
```


```python
!pwd
!ls
```

    /content/drive/MyDrive/DeepLearning/Park_Professor/DeepLearning(Image)
    Data_Augmentation.ipynb  test  train  Transfer_Learning.ipynb



```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, models, layers, optimizers, metrics
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

np.random.seed(3)
tf.compat.v1.set_random_seed(3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
       'train',
       target_size=(150, 150),
       batch_size=5,
       class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
       'test',
       target_size=(150, 150),
       batch_size=5,
       class_mode='binary')

# 이미지 불러오기
transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
transfer_model.trainable = False
transfer_model.summary()

# 모델 피팅
finetune_model = models.Sequential()
finetune_model.add(transfer_model)
finetune_model.add(Flatten())
finetune_model.add(Dense(64, activation='relu'))
finetune_model.add(Dense(2, activation='softmax'))
finetune_model.summary()

finetune_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002), metrics=['accuracy'])

history = finetune_model.fit_generator(
       train_generator,
       steps_per_epoch=100,
       epochs=20,
       validation_data=test_generator,
       validation_steps=4)

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
plt.plot(x_len, y_vloss, marker='.', c="cornflowerblue", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()

```

    Found 160 images belonging to 2 classes.
    Found 120 images belonging to 2 classes.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 3s 0us/step
    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 0
    Non-trainable params: 14,714,688
    _________________________________________________________________
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Model)                (None, 4, 4, 512)         14714688  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                524352    
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 130       
    =================================================================
    Total params: 15,239,170
    Trainable params: 524,482
    Non-trainable params: 14,714,688
    _________________________________________________________________
    Epoch 1/20
    100/100 [==============================] - 385s 4s/step - loss: 0.4558 - accuracy: 0.7640 - val_loss: 0.2284 - val_accuracy: 1.0000
    Epoch 2/20
    100/100 [==============================] - 367s 4s/step - loss: 0.2117 - accuracy: 0.9460 - val_loss: 0.2273 - val_accuracy: 0.9000
    Epoch 3/20
    100/100 [==============================] - 362s 4s/step - loss: 0.1446 - accuracy: 0.9580 - val_loss: 0.1129 - val_accuracy: 1.0000
    ...
    Epoch 17/20
    100/100 [==============================] - 362s 4s/step - loss: 0.0247 - accuracy: 0.9960 - val_loss: 0.1331 - val_accuracy: 0.9000
    Epoch 18/20
    100/100 [==============================] - 356s 4s/step - loss: 0.0273 - accuracy: 0.9900 - val_loss: 0.0775 - val_accuracy: 0.9500
    Epoch 19/20
    100/100 [==============================] - 362s 4s/step - loss: 0.0198 - accuracy: 0.9960 - val_loss: 0.1099 - val_accuracy: 0.9000
    Epoch 20/20
    100/100 [==============================] - 359s 4s/step - loss: 0.0228 - accuracy: 0.9960 - val_loss: 0.0601 - val_accuracy: 1.0000



![output_8_1](https://user-images.githubusercontent.com/87309905/199171075-fa93045e-dc39-4a21-aa68-e958c135e511.png)
    



```python

```
