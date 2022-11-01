```python
pip install tensorflow==2.0.0
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting tensorflow==2.0.0
      Downloading tensorflow-2.0.0-cp37-cp37m-manylinux2010_x86_64.whl (86.3 MB)
    [K     |████████████████████████████████| 86.3 MB 57 kB/s 
    [?25hRequirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (0.8.1)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (2.0.1)
    Collecting tensorflow-estimator<2.1.0,>=2.0.0
      Downloading tensorflow_estimator-2.0.1-py2.py3-none-any.whl (449 kB)
    [K     |████████████████████████████████| 449 kB 70.9 MB/s 
    [?25hRequirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (3.3.0)
    Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.14.1)
    Requirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (3.17.3)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (0.37.1)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.1.2)
    Collecting gast==0.2.2
      Downloading gast-0.2.2.tar.gz (10 kB)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.15.0)
    Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.3.0)
    Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.50.0)
    Collecting keras-applications>=1.0.8
      Downloading Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
    [K     |████████████████████████████████| 50 kB 7.7 MB/s 
    [?25hRequirement already satisfied: google-pasta>=0.1.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (0.2.0)
    Requirement already satisfied: numpy<2.0,>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.0.0) (1.21.6)
    Collecting tensorboard<2.1.0,>=2.0.0
      Downloading tensorboard-2.0.2-py3-none-any.whl (3.8 MB)
    [K     |████████████████████████████████| 3.8 MB 44.6 MB/s 
    [?25hRequirement already satisfied: h5py in /usr/local/lib/python3.7/dist-packages (from keras-applications>=1.0.8->tensorflow==2.0.0) (3.1.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (3.4.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (2.23.0)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (57.4.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (1.0.1)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (1.35.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (0.4.6)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (4.2.4)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (4.13.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (4.1.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (3.10.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (0.4.8)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (2022.9.24)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (3.0.4)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.1.0,>=2.0.0->tensorflow==2.0.0) (3.2.2)
    Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py->keras-applications>=1.0.8->tensorflow==2.0.0) (1.5.2)
    Building wheels for collected packages: gast
      Building wheel for gast (setup.py) ... [?25l[?25hdone
      Created wheel for gast: filename=gast-0.2.2-py3-none-any.whl size=7554 sha256=f0feaaee27c15fe025899270f5d8324ba5c3c5893a46688bf0452082a75f8bf2
      Stored in directory: /root/.cache/pip/wheels/21/7f/02/420f32a803f7d0967b48dd823da3f558c5166991bfd204eef3
    Successfully built gast
    Installing collected packages: tensorflow-estimator, tensorboard, keras-applications, gast, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.9.0
        Uninstalling tensorflow-estimator-2.9.0:
          Successfully uninstalled tensorflow-estimator-2.9.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.9.1
        Uninstalling tensorboard-2.9.1:
          Successfully uninstalled tensorboard-2.9.1
      Attempting uninstall: gast
        Found existing installation: gast 0.4.0
        Uninstalling gast-0.4.0:
          Successfully uninstalled gast-0.4.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.9.2
        Uninstalling tensorflow-2.9.2:
          Successfully uninstalled tensorflow-2.9.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow-probability 0.16.0 requires gast>=0.3.2, but you have gast 0.2.2 which is incompatible.[0m
    Successfully installed gast-0.2.2 keras-applications-1.0.8 tensorboard-2.0.2 tensorflow-2.0.0 tensorflow-estimator-2.0.1



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
