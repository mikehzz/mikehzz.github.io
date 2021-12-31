# RandomForest를 활용한 자전거 대여량 예측하기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
```


```python
from tensorflow.keras import datasets, layers, models
```


```python
path = "data/"
print(os.listdir("data/"))
```

    ['sampleSubmission.csv', 'test.csv', 'train.csv', 'zNex~$haretrain.csv']
    

# load data


```python
df_train_eda = pd.read_csv(path + "train.csv", parse_dates = ['datetime'],
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
df_train_eda.head(3)
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
      <th>casual</th>
      <th>registered</th>
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
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
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
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
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
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
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



# EDA

## 데이터 타입 확인


```python
print(df_train_eda.shape, df_test.shape)
print("훈련 데이터")
print(df_train_eda.dtypes)
print("테스트 데이터")
print(df_test.dtypes)
```

    (10886, 11) (6493, 8)
    훈련 데이터
    season          int64
    holiday         int64
    workingday      int64
    weather         int64
    temp          float64
    atemp         float64
    humidity        int64
    windspeed     float64
    casual          int64
    registered      int64
    count           int64
    dtype: object
    테스트 데이터
    season          int64
    holiday         int64
    workingday      int64
    weather         int64
    temp          float64
    atemp         float64
    humidity        int64
    windspeed     float64
    dtype: object
    

## 결측치 확인


```python
print(df_train_eda.isnull().sum())
```

    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    casual        0
    registered    0
    count         0
    dtype: int64
    


```python
print(df_test.isnull().sum())
```

    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    dtype: int64
    


```python
print(df_submission.isnull().sum())
```

    count    0
    dtype: int64
    

## datetime 에서 년,월,일,시간,분,초를 추출해 column 추가


```python
df_train_eda['year'] = df_train_eda.index.year
```


```python
df_train_eda['year'] = df_train_eda.index.year
df_train_eda['month'] = df_train_eda.index.month
df_train_eda['day'] = df_train_eda.index.day
df_train_eda['hour'] = df_train_eda.index.hour
df_train_eda['minute'] = df_train_eda.index.minute
df_train_eda['second'] = df_train_eda.index.second
```


```python
df_train_eda.head()
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
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
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
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
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
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 04:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
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

bar_plot(df_train_eda, "year", ax=ax1)
bar_plot(df_train_eda, "month", ax=ax2)
bar_plot(df_train_eda, "day", ax=ax3)
bar_plot(df_train_eda, "hour", ax=ax4)
bar_plot(df_train_eda, "minute", ax=ax5)
bar_plot(df_train_eda, "second", ax=ax6)
```


    
![png](output_24_0.png)
    



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

## datetime을 기반으로 요일 추출


```python
df_train_eda['dayofweek'] = df_train_eda.index.dayofweek
df_train_eda.head(3)
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
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>second</th>
      <th>dayofweek</th>
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
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## 시간대별 자전거 대여량 (근무일 유무, 요일, 시즌, 날씨)


```python
def point_plot(df, hue, ax):
    sns.pointplot(data=df, x="hour", y="count", ax=ax, hue=hue)
```


```python
fig,(ax1, ax2, ax3, ax4)= plt.subplots(nrows=4)
fig.set_size_inches(18,25)

#sns.pointplot(df_train, ax=ax1)
point_plot(df_train_eda, 'workingday', ax=ax1)
point_plot(df_train_eda, 'dayofweek', ax=ax2)
point_plot(df_train_eda, 'season', ax=ax3)
point_plot(df_train_eda, 'weather', ax=ax4)
```


    
![png](output_31_0.png)
    


앞선 시간대별 자전거 대여량 그래프를 보면 오전8시, 오후 5~6시에 가장 대여량이 많았다.

근무일, 요일, 분기, 날씨에 따른 대여량에 대한 분석  
- 근무일에는 출근시간(8시), 퇴근시간(17 ~ 18시)에 가장 대여량이 높았고, 휴무일에는 12 ~ 16시에 가장 대여를 많이했다.
- 평일은 근무일 대여량, 주말은 휴무일 대여량에 따르는 것을 확인할 수 있다.
- 3분기(7 ~ 9월)에 가장 대여를 많이하고, 1분기(1 ~ 3월)에 가장 적게 대여를 하는 것을 볼 수 있다.
- 날씨가 좋을 수록 대여를 많이하고, 좋지 않을 때는 대여를 많이 안한다.

# EDA 분석 끝 다시 원본 데이터 불러오기


```python
df_train = pd.read_csv(path + "train.csv", parse_dates = ['datetime'],
                       index_col='datetime', infer_datetime_format=True)
```


```python
df_train.shape
```




    (10886, 11)




```python
df_train.tail()
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
      <th>casual</th>
      <th>registered</th>
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
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-12-19 19:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>15.58</td>
      <td>19.695</td>
      <td>50</td>
      <td>26.0027</td>
      <td>7</td>
      <td>329</td>
      <td>336</td>
    </tr>
    <tr>
      <th>2012-12-19 20:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>14.76</td>
      <td>17.425</td>
      <td>57</td>
      <td>15.0013</td>
      <td>10</td>
      <td>231</td>
      <td>241</td>
    </tr>
    <tr>
      <th>2012-12-19 21:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.94</td>
      <td>15.910</td>
      <td>61</td>
      <td>15.0013</td>
      <td>4</td>
      <td>164</td>
      <td>168</td>
    </tr>
    <tr>
      <th>2012-12-19 22:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.94</td>
      <td>17.425</td>
      <td>61</td>
      <td>6.0032</td>
      <td>12</td>
      <td>117</td>
      <td>129</td>
    </tr>
    <tr>
      <th>2012-12-19 23:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.12</td>
      <td>16.665</td>
      <td>66</td>
      <td>8.9981</td>
      <td>4</td>
      <td>84</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>



### df_train데이터는 한 시간 단위로 2011-01-01 00:00:00  ~ 2012-12-19 23 23:00:00 ( 1일부터19일)
### 즉, 2(years) * 12(month) * 19(days) * 24(hours) =  10944(timestep)이어야한다.
### 하지만 df_train데이터는 10886(timestep) 이므로 중간에 누락된 정보가 있다.
### 따라서 (10944-10886) = 58개의 timestep을 채워야 한다.


```python
num_months_per_year = 12
year_list = [2011, 2012]
```


```python
df_train_temp = pd.DataFrame(columns=df_train.columns)

for year in year_list: # 2011 2012
    for month in range(num_months_per_year):# 0~11
        start_date = datetime.datetime(year, month+1, 1, 0, 0, 0)
        end_date = datetime.datetime(year, month+1, 19, 23, 0, 0)
        temp = df_train[start_date:end_date].resample('H').asfreq()
        df_train_temp = df_train_temp.append(temp)
        
train_data = df_train_temp
```


```python
train_data.shape
```




    (10944, 11)



### 채워준 timestep의 결측치를 채워야한다.


```python
train_data.isna().sum()
```




    season        58
    holiday       58
    workingday    58
    weather       58
    temp          58
    atemp         58
    humidity      58
    windspeed     58
    casual        58
    registered    58
    count         58
    dtype: int64




```python
null_feature = train_data[train_data['count'].isnull()].index
null_feature
```




    DatetimeIndex(['2011-01-02 05:00:00', '2011-01-03 02:00:00',
                   '2011-01-03 03:00:00', '2011-01-04 03:00:00',
                   '2011-01-05 03:00:00', '2011-01-06 03:00:00',
                   '2011-01-07 03:00:00', '2011-01-11 03:00:00',
                   '2011-01-11 04:00:00', '2011-01-12 03:00:00',
                   '2011-01-12 04:00:00', '2011-01-14 04:00:00',
                   '2011-01-18 00:00:00', '2011-01-18 01:00:00',
                   '2011-01-18 02:00:00', '2011-01-18 03:00:00',
                   '2011-01-18 04:00:00', '2011-01-18 05:00:00',
                   '2011-01-18 06:00:00', '2011-01-18 07:00:00',
                   '2011-01-18 08:00:00', '2011-01-18 09:00:00',
                   '2011-01-18 10:00:00', '2011-01-18 11:00:00',
                   '2011-01-19 03:00:00', '2011-02-01 04:00:00',
                   '2011-02-03 04:00:00', '2011-02-04 04:00:00',
                   '2011-02-09 04:00:00', '2011-02-10 03:00:00',
                   '2011-02-11 03:00:00', '2011-02-11 04:00:00',
                   '2011-02-13 05:00:00', '2011-02-15 03:00:00',
                   '2011-02-16 02:00:00', '2011-03-06 05:00:00',
                   '2011-03-07 02:00:00', '2011-03-10 03:00:00',
                   '2011-03-10 04:00:00', '2011-03-11 04:00:00',
                   '2011-03-13 02:00:00', '2011-03-14 04:00:00',
                   '2011-03-15 03:00:00', '2011-03-16 02:00:00',
                   '2011-03-18 04:00:00', '2011-04-11 03:00:00',
                   '2011-09-06 01:00:00', '2011-09-08 02:00:00',
                   '2011-09-12 03:00:00', '2011-10-19 03:00:00',
                   '2012-01-02 03:00:00', '2012-01-10 03:00:00',
                   '2012-01-17 03:00:00', '2012-02-06 03:00:00',
                   '2012-03-11 02:00:00', '2012-04-02 03:00:00',
                   '2012-04-11 04:00:00', '2012-11-08 03:00:00'],
                  dtype='datetime64[ns]', freq=None)



### season, holiday, workingday, weather들은 backfill로 채운다.(같은 날짜 이기 때문)


```python
backfill_features = ['season', 'holiday', 'workingday', 'weather']
train_data[backfill_features] = train_data[backfill_features].fillna(method='backfill')
```

### temp, atemp, humidity, windspeed 들은 linear로 채운다.


```python
fill_linear_features = ['temp', 'atemp', 'humidity', 'windspeed']
train_data[fill_linear_features] = train_data[fill_linear_features].interpolate(method='linear')
```

### 문제는 casual, registered, count들이다.
### target값과 연관된 column이기 때문에 머신러닝 기법을 활용해 채운다.
### casual, registered 들의 합은 count이기 때문에 drop한다.


```python
null_df = train_data.loc[null_feature]
null_df = null_df.drop("casual",axis=1)
null_df = null_df.drop("registered",axis=1)
null_df.head() # 
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
  </thead>
  <tbody>
    <tr>
      <th>2011-01-02 05:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>18.040000</td>
      <td>21.9675</td>
      <td>85.5</td>
      <td>16.498750</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-01-03 02:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.653333</td>
      <td>7.8300</td>
      <td>45.0</td>
      <td>27.333767</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-01-03 03:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.106667</td>
      <td>7.3250</td>
      <td>46.0</td>
      <td>26.668233</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-01-04 03:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.740000</td>
      <td>8.3325</td>
      <td>63.0</td>
      <td>7.500650</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-01-05 03:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8.200000</td>
      <td>10.6075</td>
      <td>61.0</td>
      <td>10.502250</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_df = train_data.drop(index = null_feature)
full_df = full_df.drop("casual",axis=1)
full_df = full_df.drop("registered",axis=1)
full_df.head() 
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
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>2011-01-01 04:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## EDA분석에서 시간대별 대여량에 영향이 있는  column을 갖고온다.


```python
null_df['year'] = null_df.index.year
null_df['month'] = null_df.index.month
null_df['day'] = null_df.index.day
null_df['hour'] = null_df.index.hour

full_df['year'] = full_df.index.year
full_df['month'] = full_df.index.month
full_df['day'] = full_df.index.day
full_df['hour'] = full_df.index.hour
```


```python
X = full_df.drop("count", axis=1)
X
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81.0</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80.0</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75.0</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2011-01-01 04:00:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75.0</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
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
    </tr>
    <tr>
      <th>2012-12-19 19:00:00</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.58</td>
      <td>19.695</td>
      <td>50.0</td>
      <td>26.0027</td>
      <td>2012</td>
      <td>12</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2012-12-19 20:00:00</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.76</td>
      <td>17.425</td>
      <td>57.0</td>
      <td>15.0013</td>
      <td>2012</td>
      <td>12</td>
      <td>19</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2012-12-19 21:00:00</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>13.94</td>
      <td>15.910</td>
      <td>61.0</td>
      <td>15.0013</td>
      <td>2012</td>
      <td>12</td>
      <td>19</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2012-12-19 22:00:00</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>13.94</td>
      <td>17.425</td>
      <td>61.0</td>
      <td>6.0032</td>
      <td>2012</td>
      <td>12</td>
      <td>19</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2012-12-19 23:00:00</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>13.12</td>
      <td>16.665</td>
      <td>66.0</td>
      <td>8.9981</td>
      <td>2012</td>
      <td>12</td>
      <td>19</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>10886 rows × 12 columns</p>
</div>




```python
y = full_df["count"]
y
```




    2011-01-01 00:00:00     16.0
    2011-01-01 01:00:00     40.0
    2011-01-01 02:00:00     32.0
    2011-01-01 03:00:00     13.0
    2011-01-01 04:00:00      1.0
                           ...  
    2012-12-19 19:00:00    336.0
    2012-12-19 20:00:00    241.0
    2012-12-19 21:00:00    168.0
    2012-12-19 22:00:00    129.0
    2012-12-19 23:00:00     88.0
    Name: count, Length: 10886, dtype: float64



## randomforest 학습


```python
from sklearn.model_selection import train_test_split
```

### 데이터 split


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

### 필요한 라이브러리


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV  # 매개변수를 자동으로 변수 설정해서 최적 찾기 위한 방법
from sklearn.model_selection import KFold  # 몇번 나눌지
from sklearn.metrics import r2_score, mean_squared_error  # SVR 모델의 예측과 실제의 결과를 확인하는 성능지표

def rmsle(y,pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle
```


```python
seed = 2021
```

### random_grid 파라미터 값 설정


```python
"""
# 랜덤 포레스트의 트리 수 
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)] 
# 모든 분할에서 고려해야 할 기능의 수 
max_features = ['auto', ' qrt '] 
# 트리의 최대 레벨 수 
max_depth = [int(x) for x in np.linspace(10, 11, num = 11)] 
max_depth.append(None) 
# 노드를 분할하는 데 필요한 최소 샘플 수 
min_samples_split = [2, 5, 10] 
# 각 리프 노드에 필요한 최소 샘플 수 
min_samples_leaf = [1, 2, 4] 
# 각 트리 
# 부트스트랩 훈련을 위한 샘플 선택 방법 = [True, False]
# 랜덤 그리드 생성 
random_grid = {'n_estimators': n_estimators, 
               'max_features': max_features, 
               'max_depth': max_depth, 
               'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf, 
               }"""
```


```python
random_grid = {'n_estimators': [100], 
               'max_features': ["auto"], 
               'max_depth': [10], 
               'min_samples_split': [2], 
               'min_samples_leaf': [1], 
               }
```


```python
rf = RandomForestRegressor(random_state=seed, bootstrap=True)
cv = KFold(n_splits = 5, shuffle = True, random_state=seed)

grid=GridSearchCV(estimator=rf,
                 param_grid=random_grid,
                 cv=cv)

grid.fit(X_train, y_train)
```

## best param 값 


```python
best_parameters =grid.best_params_
print(best_parameters)
```

    {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    

## final model에 다시 적용


```python
final_model=RandomForestRegressor(**best_parameters, random_state=seed)
final_model.fit(X_train, y_train)

predict = final_model.predict(X_test)
```


```python
Score=r2_score(y_test,predict)
Rmsle = rmsle(y_test,predict)
print("r2 score:{:0.3f}, rmsle:{:0.3f}".format(Score, Rmsle))
```

    r2 score:0.933, rmsle:0.361
    
