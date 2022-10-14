# MiniProject [ Lotto number 예측 ]

## 로컬에 있는 파일 불러오기


```python
from google.colab import files
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import io
import pandas as pd
import numpy as np
```


```python
uploaded = files.upload()
```



     <input type="file" id="files-9068e3c3-02e4-438b-bda7-d9ba5fd36f31" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-9068e3c3-02e4-438b-bda7-d9ba5fd36f31">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving Lotto.csv to Lotto (3).csv



```python
Lotto = pd.read_csv(io.StringIO(uploaded['Lotto.csv'].decode('utf-8')))
```


```python
Lotto
```





  <div id="df-8d31e5d6-ce6c-43cc-8950-587d420ae87f">
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
      <th>ȸ· ß÷᰺</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Unnamed: 11</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
      <th>Unnamed: 17</th>
      <th>Unnamed: 18</th>
      <th>Unnamed: 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>⵵</td>
      <td>ȸ·</td>
      <td>ß÷</td>
      <td>1td&gt;</td>
      <td>NaN</td>
      <td>2td&gt;</td>
      <td>NaN</td>
      <td>3td&gt;</td>
      <td>NaN</td>
      <td>4td&gt;</td>
      <td>NaN</td>
      <td>5td&gt;</td>
      <td>NaN</td>
      <td>烷ȣ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>烷__d&gt;</td>
      <td>烷ݾ׼/td&gt;</td>
      <td>烷__d&gt;</td>
      <td>烷ݾ׼/td&gt;</td>
      <td>烷__d&gt;</td>
      <td>烷ݾ׼/td&gt;</td>
      <td>烷__d&gt;</td>
      <td>烷ݾ׼/td&gt;</td>
      <td>烷__d&gt;</td>
      <td>烷ݾ׼/td&gt;</td>
      <td>1</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>ʽ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>1036</td>
      <td>2022.10.8</td>
      <td>9</td>
      <td>2,837,323,167</td>
      <td>64</td>
      <td>66,499,762</td>
      <td>2,593</td>
      <td>1,641,337</td>
      <td>133,443</td>
      <td>50,000</td>
      <td>2,231,388</td>
      <td>5,000</td>
      <td>2</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>1035</td>
      <td>2022.10.1</td>
      <td>8</td>
      <td>3,231,193,735</td>
      <td>71</td>
      <td>60,679,695</td>
      <td>2,848</td>
      <td>1,512,732</td>
      <td>141,624</td>
      <td>50,000</td>
      <td>2,366,499</td>
      <td>5,000</td>
      <td>9</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>1034</td>
      <td>2022.9.24</td>
      <td>9</td>
      <td>2,868,856,209</td>
      <td>66</td>
      <td>65,201,278</td>
      <td>2,898</td>
      <td>1,484,916</td>
      <td>142,939</td>
      <td>50,000</td>
      <td>2,400,364</td>
      <td>5,000</td>
      <td>26</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
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
    </tr>
    <tr>
      <th>1033</th>
      <td>NaN</td>
      <td>5</td>
      <td>2003.1.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>6,033,800</td>
      <td>3,043</td>
      <td>166,500</td>
      <td>60,434</td>
      <td>10,000</td>
      <td>16</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2002</td>
      <td>4</td>
      <td>2002.12.28</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>211,191,200</td>
      <td>29</td>
      <td>7,282,400</td>
      <td>2,777</td>
      <td>152,100</td>
      <td>52,382</td>
      <td>10,000</td>
      <td>14</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>NaN</td>
      <td>3</td>
      <td>2002.12.21</td>
      <td>1</td>
      <td>2,000,000,000</td>
      <td>0</td>
      <td>0</td>
      <td>139</td>
      <td>1,174,100</td>
      <td>5,940</td>
      <td>54,900</td>
      <td>73,256</td>
      <td>10,000</td>
      <td>11</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>NaN</td>
      <td>2</td>
      <td>2002.12.14</td>
      <td>1</td>
      <td>2,002,006,800</td>
      <td>2</td>
      <td>94,866,800</td>
      <td>103</td>
      <td>1,842,000</td>
      <td>3,763</td>
      <td>100,800</td>
      <td>55,480</td>
      <td>10,000</td>
      <td>9</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>NaN</td>
      <td>1</td>
      <td>2002.12.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>143,934,100</td>
      <td>28</td>
      <td>5,140,500</td>
      <td>2,537</td>
      <td>113,400</td>
      <td>40,155</td>
      <td>10,000</td>
      <td>10</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1038 rows × 20 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8d31e5d6-ce6c-43cc-8950-587d420ae87f')"
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
          document.querySelector('#df-8d31e5d6-ce6c-43cc-8950-587d420ae87f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8d31e5d6-ce6c-43cc-8950-587d420ae87f');
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




## 전처리

### 필요없는 행 삭제


```python
Lotto = Lotto.drop([0,1])
```


```python
Lotto = Lotto.reset_index()
```


```python
Lotto
```





  <div id="df-3efaf136-0232-4010-8266-d374dc899fec">
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
      <th>index</th>
      <th>ȸ· ß÷᰺</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>...</th>
      <th>Unnamed: 10</th>
      <th>Unnamed: 11</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
      <th>Unnamed: 17</th>
      <th>Unnamed: 18</th>
      <th>Unnamed: 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2022</td>
      <td>1036</td>
      <td>2022.10.8</td>
      <td>9</td>
      <td>2,837,323,167</td>
      <td>64</td>
      <td>66,499,762</td>
      <td>2,593</td>
      <td>1,641,337</td>
      <td>...</td>
      <td>50,000</td>
      <td>2,231,388</td>
      <td>5,000</td>
      <td>2</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>NaN</td>
      <td>1035</td>
      <td>2022.10.1</td>
      <td>8</td>
      <td>3,231,193,735</td>
      <td>71</td>
      <td>60,679,695</td>
      <td>2,848</td>
      <td>1,512,732</td>
      <td>...</td>
      <td>50,000</td>
      <td>2,366,499</td>
      <td>5,000</td>
      <td>9</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>NaN</td>
      <td>1034</td>
      <td>2022.9.24</td>
      <td>9</td>
      <td>2,868,856,209</td>
      <td>66</td>
      <td>65,201,278</td>
      <td>2,898</td>
      <td>1,484,916</td>
      <td>...</td>
      <td>50,000</td>
      <td>2,400,364</td>
      <td>5,000</td>
      <td>26</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>NaN</td>
      <td>1033</td>
      <td>2022.9.17</td>
      <td>13</td>
      <td>1,913,414,943</td>
      <td>79</td>
      <td>52,477,625</td>
      <td>3,083</td>
      <td>1,344,708</td>
      <td>...</td>
      <td>50,000</td>
      <td>2,376,004</td>
      <td>5,000</td>
      <td>3</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>NaN</td>
      <td>1032</td>
      <td>2022.9.10</td>
      <td>10</td>
      <td>2,675,257,538</td>
      <td>90</td>
      <td>49,541,807</td>
      <td>3,078</td>
      <td>1,448,591</td>
      <td>...</td>
      <td>50,000</td>
      <td>2,458,611</td>
      <td>5,000</td>
      <td>1</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>42.0</td>
      <td>28</td>
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
      <th>1031</th>
      <td>1033</td>
      <td>NaN</td>
      <td>5</td>
      <td>2003.1.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>6,033,800</td>
      <td>...</td>
      <td>166,500</td>
      <td>60,434</td>
      <td>10,000</td>
      <td>16</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>1034</td>
      <td>2002</td>
      <td>4</td>
      <td>2002.12.28</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>211,191,200</td>
      <td>29</td>
      <td>7,282,400</td>
      <td>...</td>
      <td>152,100</td>
      <td>52,382</td>
      <td>10,000</td>
      <td>14</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>1035</td>
      <td>NaN</td>
      <td>3</td>
      <td>2002.12.21</td>
      <td>1</td>
      <td>2,000,000,000</td>
      <td>0</td>
      <td>0</td>
      <td>139</td>
      <td>1,174,100</td>
      <td>...</td>
      <td>54,900</td>
      <td>73,256</td>
      <td>10,000</td>
      <td>11</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>1036</td>
      <td>NaN</td>
      <td>2</td>
      <td>2002.12.14</td>
      <td>1</td>
      <td>2,002,006,800</td>
      <td>2</td>
      <td>94,866,800</td>
      <td>103</td>
      <td>1,842,000</td>
      <td>...</td>
      <td>100,800</td>
      <td>55,480</td>
      <td>10,000</td>
      <td>9</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1037</td>
      <td>NaN</td>
      <td>1</td>
      <td>2002.12.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>143,934,100</td>
      <td>28</td>
      <td>5,140,500</td>
      <td>...</td>
      <td>113,400</td>
      <td>40,155</td>
      <td>10,000</td>
      <td>10</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1036 rows × 21 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3efaf136-0232-4010-8266-d374dc899fec')"
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
          document.querySelector('#df-3efaf136-0232-4010-8266-d374dc899fec button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3efaf136-0232-4010-8266-d374dc899fec');
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




### 필요없는 열 삭제


```python
Lotto = Lotto.drop(Lotto.iloc[:,3:14], axis = 1)
```


```python
Lotto = Lotto.drop(Lotto.iloc[:,:2], axis = 1)
Lotto
```





  <div id="df-a4d2f003-4ac3-4c6c-b53d-b326a2b494ab">
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
      <th>Unnamed: 1</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
      <th>Unnamed: 17</th>
      <th>Unnamed: 18</th>
      <th>Unnamed: 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1036</td>
      <td>2</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1035</td>
      <td>9</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1034</td>
      <td>26</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1033</td>
      <td>3</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1032</td>
      <td>1</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>42.0</td>
      <td>28</td>
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
    </tr>
    <tr>
      <th>1031</th>
      <td>5</td>
      <td>16</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>4</td>
      <td>14</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>3</td>
      <td>11</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2</td>
      <td>9</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1</td>
      <td>10</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1036 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a4d2f003-4ac3-4c6c-b53d-b326a2b494ab')"
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
          document.querySelector('#df-a4d2f003-4ac3-4c6c-b53d-b326a2b494ab button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a4d2f003-4ac3-4c6c-b53d-b326a2b494ab');
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
Lotto
```





  <div id="df-de061d7e-07aa-4a33-92e0-ca7c10c5ff3e">
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
      <th>Unnamed: 1</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
      <th>Unnamed: 17</th>
      <th>Unnamed: 18</th>
      <th>Unnamed: 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1036</td>
      <td>2</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1035</td>
      <td>9</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1034</td>
      <td>26</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1033</td>
      <td>3</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1032</td>
      <td>1</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>42.0</td>
      <td>28</td>
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
    </tr>
    <tr>
      <th>1031</th>
      <td>5</td>
      <td>16</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>4</td>
      <td>14</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>3</td>
      <td>11</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2</td>
      <td>9</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1</td>
      <td>10</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1036 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-de061d7e-07aa-4a33-92e0-ca7c10c5ff3e')"
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
          document.querySelector('#df-de061d7e-07aa-4a33-92e0-ca7c10c5ff3e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-de061d7e-07aa-4a33-92e0-ca7c10c5ff3e');
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




### 인덱스 명 변경


```python
Lotto.rename(columns = {'Unnamed: 1' : 'Round',	
                       'Unnamed: 13' : 'First',	
                       'Unnamed: 14' : 'Second',	
                       'Unnamed: 15' : 'Thrid',
                       'Unnamed: 16' : 'Fourth',
                       'Unnamed: 17' : 'Fifth',
                       'Unnamed: 18' : 'Sixth',
                       'Unnamed: 19' : 'Bonus'}, inplace = True)
```


```python
Lotto
```





  <div id="df-ddb210ae-4b14-4f0f-a785-924557d0ca47">
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
      <th>Round</th>
      <th>First</th>
      <th>Second</th>
      <th>Thrid</th>
      <th>Fourth</th>
      <th>Fifth</th>
      <th>Sixth</th>
      <th>Bonus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1036</td>
      <td>2</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1035</td>
      <td>9</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1034</td>
      <td>26</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1033</td>
      <td>3</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1032</td>
      <td>1</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>42.0</td>
      <td>28</td>
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
    </tr>
    <tr>
      <th>1031</th>
      <td>5</td>
      <td>16</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>4</td>
      <td>14</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>3</td>
      <td>11</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2</td>
      <td>9</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1</td>
      <td>10</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1036 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ddb210ae-4b14-4f0f-a785-924557d0ca47')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
  </div>





```python
Lotto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1036 entries, 0 to 1035
    Data columns (total 8 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Round   1036 non-null   object 
     1   First   1036 non-null   object 
     2   Second  1036 non-null   float64
     3   Thrid   1036 non-null   float64
     4   Fourth  1036 non-null   float64
     5   Fifth   1036 non-null   float64
     6   Sixth   1036 non-null   float64
     7   Bonus   1036 non-null   object 
    dtypes: float64(5), object(3)
    memory usage: 64.9+ KB



```python
Lotto['First'] = Lotto['First'].astype(float)
```


```python
Lotto
```





  <div id="df-5789e0f6-b97a-4d3b-be9e-cdf2ed105c0e">
    <div class="colab-df-container">
      <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Round</th>
      <th>First</th>
      <th>Second</th>
      <th>Thrid</th>
      <th>Fourth</th>
      <th>Fifth</th>
      <th>Sixth</th>
      <th>Bonus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1036</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1035</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1034</td>
      <td>26.0</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1033</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1032</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>42.0</td>
      <td>28</td>
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
    </tr>
    <tr>
      <th>1031</th>
      <td>5</td>
      <td>16.0</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>4</td>
      <td>14.0</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>3</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>31.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1</td>
      <td>10.0</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1036 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5789e0f6-b97a-4d3b-be9e-cdf2ed105c0e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
  </div>




### 원핫 인코딩


```python
row_count = len(Lotto)

# 당첨번호를 원핫 인코딩벡터로 변환
def numbers2ohbin(numbers):

  ohbin = np.zeros(45)

  for i in range(6):
    ohbin[int(numbers[i]) - 1] = 1

  return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

  numbers = []
  
  for i in range(len(ohbin)):
    if ohbin[i] == 1.0:
      numbers.append(i + 1)

  return numbers

numbers = Lotto.iloc[:, 1:7].values
ohbins = list(map(numbers2ohbin, numbers))

X = ohbins[0: row_count-1]
y = ohbins[1:row_count]
```


```python
train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(X))
```


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# 모델을 정의합니다.
model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 매 에포크마다 훈련과 검증의 손실 및 정확도를 기록하기 위한 변수
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# 최대 100번 에포크까지 수행
for epoch in range(100):

    model.reset_states() # 중요! 매 에포크마다 1회부터 다시 훈련하므로 상태 초기화 필요

    batch_train_loss = []
    batch_train_acc = []
    
    for i in range(train_idx[0], train_idx[1]):
        
        xs = X[i].reshape(1, 1, 45)
        ys = y[i].reshape(1, 45)
        
        loss, acc = model.train_on_batch(xs, ys) #배치만큼 모델에 학습시킴

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):

        xs = X[i].reshape(1, 1, 45)
        ys = y[i].reshape(1, 45)
        
        loss, acc = model.test_on_batch(xs, ys) #배치만큼 모델에 입력하여 나온 답을 정답과 비교함
        
        batch_val_loss.append(loss)
        batch_val_acc.append(acc)

    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))

    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))
```

    epoch    0 train acc 0.020 loss 0.408 val acc 0.000 loss 0.396
    epoch    1 train acc 0.019 loss 0.396 val acc 0.000 loss 0.395
    epoch    2 train acc 0.025 loss 0.394 val acc 0.010 loss 0.395
    epoch    3 train acc 0.031 loss 0.392 val acc 0.020 loss 0.395
    ...
    epoch   97 train acc 0.150 loss 0.011 val acc 0.020 loss 1.225
    epoch   98 train acc 0.166 loss 0.010 val acc 0.030 loss 1.233
    epoch   99 train acc 0.172 loss 0.014 val acc 0.040 loss 1.222



```python
# 번호 뽑기
def gen_numbers_from_probability(nums_prob):

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls
    
print('receive numbers')

xs = X[-1].reshape(1, 1, 45)

ys_pred = model.predict_on_batch(xs)

list_numbers = []

for n in range(5):
    numbers = gen_numbers_from_probability(ys_pred[0])
    numbers.sort()
    print('{0} : {1}'.format(n, numbers))
    list_numbers.append(numbers)
```

    receive numbers
    0 : [3, 12, 16, 28, 41, 42]
    1 : [2, 3, 20, 41, 42, 44]
    2 : [2, 3, 16, 26, 28, 42]
    3 : [2, 3, 16, 25, 28, 41]
    4 : [2, 3, 16, 28, 41, 42]
