# RNN을 활용한 자전거 대여량(시계열 데이터) 예측

### 밑의 사이트에서 만든 데이터로 RNN 학습

https://github.com/mikehzz/Python/tree/main/Machine%20learning/bike_sharing_demand


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import calendar

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
path = "data/"
print(os.listdir("data/"))
```

    ['sampleSubmission.csv', 'test.csv', 'train.csv']
    

# load data


```python
df_train = pd.read_csv(path + "train.csv", parse_dates = ['datetime'],
                         index_col='datetime', infer_datetime_format=True)
df_test = pd.read_csv(path + 'test.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)
df_submission = pd.read_csv(path+'sampleSubmission.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)
```

# Columns 

- datetime : 시간별 날짜
- season :  
  1.(1분기)  
  2.(2분기)   
  3.(3분기)  
  4.(4분기)
- holiday : 하루가 휴일로 간주되는지 여부
- workingday : 주말과 휴일이 아닌 일하는 날
- weather :   
  1.(맑음, 구름, 조금, 흐림)  
  2.(안개+흐림, 안개+구름, 안개+구름이 거의 없음 + 흐림)  
  3.(가벼운 눈, 가벼운 비 + 천둥 + 구름, 가벼운 비 + 구름)  
  4.(폭우 + 우박 + 천둥 + 안개, 눈 + 안개)
- temp : 섭씨 온도
- atemp : 섭씨 온도의 느낌
- humidity : 상대 습도
- windspeed : 풍속
- casual : 미등록 사용자 대여수
- registered : 등록된 사용자 대여수
- count : 대여수

 


```python
df_train.head(3)
```




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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>count</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.head(3)
```




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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-20 00:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>2011-01-20 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2011-01-20 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_submission.head(3)
```




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
      <th>count</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-20 00:00:00</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 01:00:00</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 02:00:00</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## datetime 에서 년,월,일,시간,분,초를 추출해 column 추가


```python
df_train['year'] = df_train.index.year
df_train['month'] = df_train.index.month
df_train['day'] = df_train.index.day
df_train['hour'] = df_train.index.hour
df_train['minute'] = df_train.index.minute
df_train['second'] = df_train.index.second
```


```python
df_test['year'] = df_test.index.year
df_test['month'] = df_test.index.month
df_test['day'] = df_test.index.day
df_test['hour'] = df_test.index.hour
df_test['minute'] = df_test.index.minute
df_test['second'] = df_test.index.second
```


```python
df_test.head()
```




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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>second</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-20 00:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 03:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-20 04:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 연도별, 월별,시간별에 따른 대여량 평균치 분석


```python
def bar_plot(df, x, ax):
    fig = plt.figure(figsize=(5,3))
    sns.barplot(data=df, x=x, y="count", palette="Blues_d", ax=ax)
```


```python
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 10)

bar_plot(df_train, "year", ax=ax1)
bar_plot(df_train, "month", ax=ax2)
bar_plot(df_train, "day", ax=ax3)
bar_plot(df_train, "hour", ax=ax4)
bar_plot(df_train, "minute", ax=ax5)
bar_plot(df_train, "second", ax=ax6)
```


    
![png](output_18_0.png)
    



    <Figure size 360x216 with 0 Axes>



    <Figure size 360x216 with 0 Axes>



    <Figure size 360x216 with 0 Axes>



    <Figure size 360x216 with 0 Axes>



    <Figure size 360x216 with 0 Axes>



    <Figure size 360x216 with 0 Axes>


## 연도별,월별,시간별에 따른 분석 결과

- 연도별 : 2011년보다 2012년 대여량이 많아짐  
- 월별 : 월별 대여량은 6월에 가장 많고, 따뜻한 계절(5~10월달)에 대여량이 많음
- 일별 : 일별 대여량은 크게 차이점이 없고, 특징점 역시 없음.
- 시간별 : 오전에는 8시에 가장 많고, 오후에는 17시~18시에 가장 많음

### count에 영향을 주는 time feature는 year, month, hour이고, 나머지는 크게 영향을 안받음. 

## 시간대별 자전거 대여량 (근무일 유무, 요일, 시즌, 날씨)


```python
def point_plot(df, hue, ax):
    sns.pointplot(data=df, x="hour", y="count", ax=ax, hue=hue)
```


```python
fig,(ax1, ax2, ax3)= plt.subplots(nrows=3)
fig.set_size_inches(18,25)

#sns.pointplot(df_train, ax=ax1)
point_plot(df_train, 'workingday', ax=ax1)
point_plot(df_train, 'season', ax=ax2)
point_plot(df_train, 'weather', ax=ax3)
```


    
![png](output_23_0.png)
    


앞선 시간대별 자전거 대여량 그래프를 보면 오전8시, 오후 5~6시에 가장 대여량이 많았다.

근무일, 요일, 분기, 날씨에 따른 대여량에 대한 분석  
- 근무일에는 출근시간(8시), 퇴근시간(17 ~ 18시)에 가장 대여량이 높았고, 휴무일에는 12 ~ 16시에 가장 대여를 많이했다.
- 평일은 근무일 대여량, 주말은 휴무일 대여량에 따르는 것을 확인할 수 있다.
- 3분기(7 ~ 9월)에 가장 대여를 많이하고, 1분기(1 ~ 3월)에 가장 적게 대여를 하는 것을 볼 수 있다.
- 날씨가 좋을 수록 대여를 많이하고, 좋지 않을 때는 대여를 많이 안한다.

# scaler

StandardScaler는 평균값과 표준편차로 계산하기 때문에 이상치가 있는지 확인해줘야 함.


```python
from sklearn.preprocessing import StandardScaler
```


```python
def dist_plot(x, ax):
    sns.distplot(x=x, ax=ax)
```


```python
fig,((ax1, ax2),(ax3, ax4))= plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(10,10)

dist_plot(df_train["temp"], ax=ax1)
dist_plot(df_train["atemp"], ax=ax2)
dist_plot(df_train["humidity"], ax=ax3)
dist_plot(df_train["windspeed"], ax=ax4)
```


    
![png](output_29_0.png)
    


### temp, atemp,humidity,windspeed scaler


```python
columns_to_scale = ['temp','atemp','humidity','windspeed']
train_temp_hum_wind_transformer = StandardScaler().fit(df_train[columns_to_scale].to_numpy())
test_temp_hum_wind_transformer = StandardScaler().fit(df_test[columns_to_scale].to_numpy())

columns_to_scale_2 = ["count"]
count_transformer = StandardScaler().fit(df_train[columns_to_scale_2])

df_train.loc[:,columns_to_scale] = train_temp_hum_wind_transformer.transform(df_train[columns_to_scale].to_numpy())
df_test.loc[:,columns_to_scale] = test_temp_hum_wind_transformer.transform(df_test[columns_to_scale].to_numpy())

df_train["count"] = count_transformer.transform(df_train[columns_to_scale_2])
```

### time feature scaler

### year이 2011 -> 0, 2012 -> 1


```python
def scaler_year(year):
    if year in [2011]:
        return 0
    elif year in [2012]:
        return 1
    
df_train['year'] = df_train["year"].apply(scaler_year)
```


```python
def scaler_year(year):
    if year in [2011]:
        return 0
    elif year in [2012]:
        return 1

df_test['year'] = df_test["year"].apply(scaler_year)
```

### month scaler


```python
fig = plt.figure(figsize=(5,3))
sns.barplot(data=df_train, x="month", y="count")
```




    <AxesSubplot:xlabel='month', ylabel='count'>




    
![png](output_37_1.png)
    



```python
def scaler_month(month):
    if month in [1,2,3]:
        return 0
    elif month in [4,11,12]:
        return 1
    elif month in [5]:
        return 2
    elif month in [6,7,8,9,10]:
        return 3
df_train['month_group'] = df_train["month"].apply(scaler_month)
```


```python
def scaler_month(month):
    if month in [1,2,3]:
        return 0
    elif month in [4,11,12]:
        return 1
    elif month in [5]:
        return 2
    elif month in [6,7,8,9,10]:
        return 3
df_test['month_group'] = df_test["month"].apply(scaler_month)
```

### hour scaler


```python
fig = plt.figure(figsize=(5,3))
sns.barplot(data=df_train, x="hour", y="count")
```




    <AxesSubplot:xlabel='hour', ylabel='count'>




    
![png](output_41_1.png)
    



```python
def scaler_hour(hour):
    if hour in [0,1,2,3,4,5,6,22,23]:
        return 0
    elif hour in [7,9,10,11,20,21]:
        return 1
    elif hour in [8,17,18]:
        return 4
    elif hour in [6,7,8,9,10]:
        return 3
    else:
        return 2
df_train['hour_group'] = df_train["hour"].apply(scaler_hour)
```


```python
def scaler_hour(hour):
    if hour in [0,1,2,3,4,5,6,22,23]:
        return 0
    elif hour in [7,9,10,11,20,21]:
        return 1
    elif hour in [8,17,18]:
        return 4
    elif hour in [6,7,8,9,10]:
        return 3
    else:
        return 2
df_test['hour_group'] = df_test["hour"].apply(scaler_hour)
```

### drop features : month, day, hour, minute, second 


```python
drop_features = ["month","day","hour","minute","second"]

df_train = df_train.drop(drop_features, axis=1)
df_test = df_test.drop(drop_features, axis=1)
```


```python
df_train.head(3)
```




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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>count</th>
      <th>year</th>
      <th>month_group</th>
      <th>hour_group</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.322793</td>
      <td>-1.081736</td>
      <td>0.989740</td>
      <td>-1.569205</td>
      <td>-0.964093</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.427728</td>
      <td>-1.171110</td>
      <td>0.937835</td>
      <td>-1.569205</td>
      <td>-0.831593</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.427728</td>
      <td>-1.171110</td>
      <td>0.937835</td>
      <td>-1.569205</td>
      <td>-0.875760</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
training_set_size = int(len(df_train)*0.9)
test_set_size = len(df_train)-training_set_size
train_data,test_data = df_train.iloc[0:training_set_size],df_train.iloc[training_set_size:len(df_train)]
print("Length of training set is", len(train_data))    
print("Length of test set is",len(test_data))
```

    Length of training set is 9849
    Length of test set is 1095
    


```python
def create_data_sequence(X, y, time_steps=1):
    """ Create data sequence
    
    Arguments:
        * X: time-series data
        * y: Count "cnt" value
        * time_steps: Used to create input sequence of timesteps
    
    Returns:
        * input_sequence: Numpy array of sequences of time-series data
        * output: Numpy array of output i.e. next value for respective sequence
    
    """
    input_sequence, target = [], []
    for i in range(len(X) - time_steps):
        sequence = X.iloc[i:(i + time_steps),[0,1,2,3,4,5,6,7,9,10,11]].values
        input_sequence.append(sequence)        
        target.append(y.iloc[i + time_steps])
    return np.array(input_sequence), np.array(target)
```


```python
time_steps = 24

# Here training_set_sequence, test_set_sequence are input features for training and test set, as numpy arrays. 
# training_set_output and test_set_output are "cnt" values for training and test set sequences, as numpy arrays.
X_train, y_train = create_data_sequence(train_data, train_data["count"], time_steps)
X_test, y_test = create_data_sequence(test_data, test_data["count"], time_steps)

# We get training and test set sequences as [samples, time_steps, n_features]

print("Training data shape", X_train.shape, "Training data output shape", y_train.shape)
print("Test data shape", X_test.shape, "Test data output shape", y_test.shape)
```

    Training data shape (9825, 24, 11) Training data output shape (9825,)
    Test data shape (1071, 24, 11) Test data output shape (1071,)
    

# RNN


```python
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, Sequential
```

## Simple RNN 


```python
n_steps = X_train.shape[1]
n_features = X_train.shape[2]
model = Sequential()
model.add(keras.layers.SimpleRNN(32, return_sequences=True, input_shape=(n_steps, n_features), dropout=0.0, recurrent_dropout=0.2,))
model.add(keras.layers.SimpleRNN(32, return_sequences=True, input_shape=(n_steps, n_features)))
#model.add(SimpleRNN(164, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(keras.layers.SimpleRNN(16))
model.add(keras.layers.Dense(1))

# define optimizer and compile model
model.compile(optimizer="adam", loss='mse', metrics = ['mse'])

# fit model
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=50, verbose=1)
```

    Epoch 1/50
    277/277 [==============================] - 25s 86ms/step - loss: 0.5227 - mse: 0.5227 - val_loss: 0.8513 - val_mse: 0.8513
    Epoch 2/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.3252 - mse: 0.3252 - val_loss: 0.5360 - val_mse: 0.5360
    Epoch 3/50
    277/277 [==============================] - 23s 83ms/step - loss: 0.2128 - mse: 0.2128 - val_loss: 0.3690 - val_mse: 0.3690
    Epoch 4/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.1556 - mse: 0.1556 - val_loss: 0.3164 - val_mse: 0.3164
    Epoch 5/50
    277/277 [==============================] - 24s 85ms/step - loss: 0.1205 - mse: 0.1205 - val_loss: 0.2222 - val_mse: 0.2222
    Epoch 6/50
    277/277 [==============================] - 24s 85ms/step - loss: 0.1034 - mse: 0.1034 - val_loss: 0.2527 - val_mse: 0.2527
    Epoch 7/50
    277/277 [==============================] - 23s 82ms/step - loss: 0.0980 - mse: 0.0980 - val_loss: 0.2079 - val_mse: 0.2079
    Epoch 8/50
    277/277 [==============================] - 22s 79ms/step - loss: 0.0865 - mse: 0.0865 - val_loss: 0.2171 - val_mse: 0.2171
    Epoch 9/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0829 - mse: 0.0829 - val_loss: 0.2387 - val_mse: 0.2387
    Epoch 10/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.0798 - mse: 0.0798 - val_loss: 0.1749 - val_mse: 0.1749
    Epoch 11/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0748 - mse: 0.0748 - val_loss: 0.1930 - val_mse: 0.1930
    Epoch 12/50
    277/277 [==============================] - 22s 80ms/step - loss: 0.0752 - mse: 0.0752 - val_loss: 0.2355 - val_mse: 0.2355
    Epoch 13/50
    277/277 [==============================] - 22s 81ms/step - loss: 0.0731 - mse: 0.0731 - val_loss: 0.1567 - val_mse: 0.1567
    Epoch 14/50
    277/277 [==============================] - 22s 80ms/step - loss: 0.0695 - mse: 0.0695 - val_loss: 0.1774 - val_mse: 0.1774
    Epoch 15/50
    277/277 [==============================] - 22s 80ms/step - loss: 0.0688 - mse: 0.0688 - val_loss: 0.1603 - val_mse: 0.1603
    Epoch 16/50
    277/277 [==============================] - 22s 81ms/step - loss: 0.0636 - mse: 0.0636 - val_loss: 0.1616 - val_mse: 0.1616
    Epoch 17/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0615 - mse: 0.0615 - val_loss: 0.1607 - val_mse: 0.1607
    Epoch 18/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.0629 - mse: 0.0629 - val_loss: 0.1694 - val_mse: 0.1694
    Epoch 19/50
    277/277 [==============================] - 25s 90ms/step - loss: 0.0597 - mse: 0.0597 - val_loss: 0.1885 - val_mse: 0.1885
    Epoch 20/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0594 - mse: 0.0594 - val_loss: 0.1516 - val_mse: 0.1516
    Epoch 21/50
    277/277 [==============================] - 24s 85ms/step - loss: 0.0611 - mse: 0.0611 - val_loss: 0.1679 - val_mse: 0.1679
    Epoch 22/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0571 - mse: 0.0571 - val_loss: 0.1637 - val_mse: 0.1637
    Epoch 23/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0583 - mse: 0.0583 - val_loss: 0.1883 - val_mse: 0.1883
    Epoch 24/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0551 - mse: 0.0551 - val_loss: 0.1423 - val_mse: 0.1423
    Epoch 25/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0561 - mse: 0.0561 - val_loss: 0.1700 - val_mse: 0.1700
    Epoch 26/50
    277/277 [==============================] - 24s 88ms/step - loss: 0.0533 - mse: 0.0533 - val_loss: 0.1472 - val_mse: 0.1472
    Epoch 27/50
    277/277 [==============================] - 24s 88ms/step - loss: 0.0536 - mse: 0.0536 - val_loss: 0.1432 - val_mse: 0.1432
    Epoch 28/50
    277/277 [==============================] - 24s 88ms/step - loss: 0.0535 - mse: 0.0535 - val_loss: 0.1648 - val_mse: 0.1648
    Epoch 29/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0538 - mse: 0.0538 - val_loss: 0.1395 - val_mse: 0.1395
    Epoch 30/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0501 - mse: 0.0501 - val_loss: 0.1454 - val_mse: 0.1454
    Epoch 31/50
    277/277 [==============================] - 24s 87ms/step - loss: 0.0514 - mse: 0.0514 - val_loss: 0.1668 - val_mse: 0.1668
    Epoch 32/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0495 - mse: 0.0495 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 33/50
    277/277 [==============================] - 24s 85ms/step - loss: 0.0498 - mse: 0.0498 - val_loss: 0.1583 - val_mse: 0.1583
    Epoch 34/50
    277/277 [==============================] - 23s 82ms/step - loss: 0.0481 - mse: 0.0481 - val_loss: 0.1564 - val_mse: 0.1564
    Epoch 35/50
    277/277 [==============================] - 23s 82ms/step - loss: 0.0474 - mse: 0.0474 - val_loss: 0.1565 - val_mse: 0.1565
    Epoch 36/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0464 - mse: 0.0464 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 37/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.0492 - mse: 0.0492 - val_loss: 0.1551 - val_mse: 0.1551
    Epoch 38/50
    277/277 [==============================] - 24s 88ms/step - loss: 0.0476 - mse: 0.0476 - val_loss: 0.1408 - val_mse: 0.1408
    Epoch 39/50
    277/277 [==============================] - 23s 83ms/step - loss: 0.0458 - mse: 0.0458 - val_loss: 0.1445 - val_mse: 0.1445
    Epoch 40/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0455 - mse: 0.0455 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 41/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0453 - mse: 0.0453 - val_loss: 0.1439 - val_mse: 0.1439
    Epoch 42/50
    277/277 [==============================] - 24s 88ms/step - loss: 0.0450 - mse: 0.0450 - val_loss: 0.1668 - val_mse: 0.1668
    Epoch 43/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0443 - mse: 0.0443 - val_loss: 0.1630 - val_mse: 0.1630
    Epoch 44/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0435 - mse: 0.0435 - val_loss: 0.1637 - val_mse: 0.1637
    Epoch 45/50
    277/277 [==============================] - 24s 85ms/step - loss: 0.0443 - mse: 0.0443 - val_loss: 0.1738 - val_mse: 0.1738
    Epoch 46/50
    277/277 [==============================] - 23s 82ms/step - loss: 0.0434 - mse: 0.0434 - val_loss: 0.1709 - val_mse: 0.1709
    Epoch 47/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.0421 - mse: 0.0421 - val_loss: 0.1737 - val_mse: 0.1737
    Epoch 48/50
    277/277 [==============================] - 23s 84ms/step - loss: 0.0424 - mse: 0.0424 - val_loss: 0.1430 - val_mse: 0.1430
    Epoch 49/50
    277/277 [==============================] - 23s 85ms/step - loss: 0.0417 - mse: 0.0417 - val_loss: 0.1609 - val_mse: 0.1609
    Epoch 50/50
    277/277 [==============================] - 24s 86ms/step - loss: 0.0421 - mse: 0.0421 - val_loss: 0.1756 - val_mse: 0.1756
    


```python
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     simple_rnn_6 (SimpleRNN)    (None, 24, 32)            1408      
                                                                     
     simple_rnn_7 (SimpleRNN)    (None, 24, 32)            2080      
                                                                     
     simple_rnn_8 (SimpleRNN)    (None, 16)                784       
                                                                     
     dense_2 (Dense)             (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 4,289
    Trainable params: 4,289
    Non-trainable params: 0
    _________________________________________________________________
    


```python
fig,ax = plt.subplots()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='test loss')
ax.set_xlabel('EPOCHS')
ax.set_ylabel('Loss value')
plt.legend()
```


    
![png](output_55_0.png)
    



```python
test_set_predictions = model.predict(X_test)
```

## 실제 값과 예측값 비교


```python
fig,ax = plt.subplots()
plt.plot(test_set_predictions[:100,], label='Predicted count')
plt.plot(y_test[:100,], label='Actual count')
ax.set_xlabel('Hours')
ax.set_ylabel('Count (cnt)')
plt.legend();
plt.show()
```


    
![png](output_58_0.png)
    


## LSTM


```python
n_steps = X_train.shape[1]
n_features = X_train.shape[2]
model = Sequential()
model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(n_steps, n_features), dropout=0.0, recurrent_dropout=0.2,))
#model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(n_steps, n_features)))
#model.add(keras.layers.LSTM(164, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(keras.layers.LSTM(16))
model.add(keras.layers.Dense(1))

# define optimizer and compile model
model.compile(optimizer="adam", loss='mse', metrics = ['mse'])

# fit model
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=10, verbose=1)
```

    WARNING:tensorflow:Layer lstm will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
    Epoch 1/10
    277/277 [==============================] - 36s 120ms/step - loss: 0.4201 - mse: 0.4201 - val_loss: 0.6254 - val_mse: 0.6254
    Epoch 2/10
    277/277 [==============================] - 33s 118ms/step - loss: 0.2051 - mse: 0.2051 - val_loss: 0.3496 - val_mse: 0.3496
    Epoch 3/10
    277/277 [==============================] - 33s 119ms/step - loss: 0.1274 - mse: 0.1274 - val_loss: 0.3430 - val_mse: 0.3430
    Epoch 4/10
    277/277 [==============================] - 33s 118ms/step - loss: 0.1043 - mse: 0.1043 - val_loss: 0.2542 - val_mse: 0.2542
    Epoch 5/10
    277/277 [==============================] - 33s 120ms/step - loss: 0.0858 - mse: 0.0858 - val_loss: 0.2059 - val_mse: 0.2059
    Epoch 6/10
    277/277 [==============================] - 33s 121ms/step - loss: 0.0767 - mse: 0.0767 - val_loss: 0.1955 - val_mse: 0.1955
    Epoch 7/10
    277/277 [==============================] - 33s 120ms/step - loss: 0.0721 - mse: 0.0721 - val_loss: 0.2230 - val_mse: 0.2230
    Epoch 8/10
    277/277 [==============================] - 33s 118ms/step - loss: 0.0670 - mse: 0.0670 - val_loss: 0.1615 - val_mse: 0.1615
    Epoch 9/10
    277/277 [==============================] - 32s 116ms/step - loss: 0.0637 - mse: 0.0637 - val_loss: 0.1898 - val_mse: 0.1898
    Epoch 10/10
    277/277 [==============================] - 32s 115ms/step - loss: 0.0640 - mse: 0.0640 - val_loss: 0.1900 - val_mse: 0.1900
    


```python
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 24, 32)            5632      
                                                                     
     lstm_1 (LSTM)               (None, 16)                3136      
                                                                     
     dense_3 (Dense)             (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 8,785
    Trainable params: 8,785
    Non-trainable params: 0
    _________________________________________________________________
    


```python
fig,ax = plt.subplots()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='test loss')
ax.set_xlabel('EPOCHS')
ax.set_ylabel('Loss value')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2011e3f1c40>




    
![png](output_62_1.png)
    



```python
test_set_predictions = model.predict(X_test)
```


```python
fig,ax = plt.subplots()
plt.plot(test_set_predictions[:100,], label='Predicted count')
plt.plot(y_test[:100,], label='Actual count')
ax.set_xlabel('Hours')
ax.set_ylabel('Count (cnt)')
plt.legend();
plt.show()
```


    
![png](output_64_0.png)
    

