---
layout: page  
title:  "반도체(화학식) bandgap 예측"

categories:
  - Project
tags:
  - RandomForestRegressor
  - ensemble
  - chemicalFormula
  - bandgap predict
---

# chemicalFormula of Semiconductor Predict
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.core.composition import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
import matminer
from matminer.featurizers.composition import ElementProperty

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import random
```


```python
df=pd.read_csv("bandgap_data_v2.csv")  
print(df.shape) 
df.head()
```

    (1447, 6)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>chemicalFormula Clean</th>
      <th>Band gap values Clean</th>
      <th>Band gap units</th>
      <th>Band gap method</th>
      <th>Reliability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Li1F1</td>
      <td>13.60</td>
      <td>eV</td>
      <td>Reflection</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Li1F1</td>
      <td>12.61</td>
      <td>eV</td>
      <td>Reflection</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Li1F1</td>
      <td>12.60</td>
      <td>eV</td>
      <td>Estimated</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Li1F1</td>
      <td>12.10</td>
      <td>eV</td>
      <td>Absorption</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Li1F1</td>
      <td>12.00</td>
      <td>eV</td>
      <td>Absorption</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 원본 데이터 전처리

 Reliability = 1 추출


```python
df_Reliability_1 = df[df["Reliability"]== 1]
print(df_Reliability_1.shape) 
df_Reliability_1.head() 
```

    (535, 6)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>chemicalFormula Clean</th>
      <th>Band gap values Clean</th>
      <th>Band gap units</th>
      <th>Band gap method</th>
      <th>Reliability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Li1F1</td>
      <td>13.60</td>
      <td>eV</td>
      <td>Reflection</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Li1F1</td>
      <td>12.61</td>
      <td>eV</td>
      <td>Reflection</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Li1Cl1</td>
      <td>9.33</td>
      <td>eV</td>
      <td>Reflection</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Li1Br1</td>
      <td>7.95</td>
      <td>eV</td>
      <td>Absorption</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Li3Sb1</td>
      <td>1.00</td>
      <td>eV</td>
      <td>Thermal activation</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



chemicalFormula 그룹화 (value를 평균값으로) 

문자로 구성된 열은 자동필터링


```python
df_clean = df_Reliability_1.groupby("chemicalFormula Clean", as_index=False).mean()  # 정해진컬럼 같은거 그룹화 해서 평균취하기
print(df_clean.shape) 
df_clean.head() 
```

    (467, 4)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ag1Te1</td>
      <td>820.0</td>
      <td>0.850</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ag2O1</td>
      <td>785.0</td>
      <td>1.200</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



target(밴드갭) --> 분포도 확인


```python
df_clean["Band gap values Clean"].describe().round(3)
```




    count    467.000
    mean       2.231
    std        2.287
    min        0.009
    25%        0.695
    50%        1.435
    75%        3.000
    max       13.105
    Name: Band gap values Clean, dtype: float64




```python
def histogram_plot(data):
    fig1, ax1 = plt.subplots()
    ax1.hist(data, bins=range(13), density=1)
    ax1.set_xticks(range(14))
    ax1.set_xlabel('Measured Bandgap [eV]')
    ax1.set_ylabel('Counts [fraction]')
    plt.show()
```


```python
histogram_plot(df_clean["Band gap values Clean"].astype("float"))
```


    
![11_0](/assets/images/bandgap_predict/output_11_0.png)
    



```python
sns.distplot(df_clean["Band gap values Clean"], color="green")
```




    <AxesSubplot:xlabel='Band gap values Clean'>




    
![12_1](/assets/images/bandgap_predict/output_12_1.png)
    


## 각 chemicalFormula의 특성들을 get_composition함수를 사용해 추가함


```python
def get_composition(c):
    try:
        return Composition(c)
    except:
        return None
```


```python
df_clean["composition"] = df_clean["chemicalFormula Clean"].apply(get_composition)
```


```python
df_clean.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
      <th>composition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
      <td>(Ag, Br)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
      <td>(Ag, Cl)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
      <td>(Ag, N)</td>
    </tr>
  </tbody>
</table>
</div>



chemicalFormula들의 magpie 값


```python
ep_feat = ElementProperty.from_preset(preset_name = 'magpie')
df_magpie = ep_feat.featurize_dataframe(df_clean, col_id ="composition")
print(df_magpie.shape)
df_magpie.head()
```
    (467, 137)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
      <th>composition</th>
      <th>MagpieData minimum Number</th>
      <th>MagpieData maximum Number</th>
      <th>MagpieData range Number</th>
      <th>MagpieData mean Number</th>
      <th>MagpieData avg_dev Number</th>
      <th>...</th>
      <th>MagpieData range GSmagmom</th>
      <th>MagpieData mean GSmagmom</th>
      <th>MagpieData avg_dev GSmagmom</th>
      <th>MagpieData mode GSmagmom</th>
      <th>MagpieData minimum SpaceGroupNumber</th>
      <th>MagpieData maximum SpaceGroupNumber</th>
      <th>MagpieData range SpaceGroupNumber</th>
      <th>MagpieData mean SpaceGroupNumber</th>
      <th>MagpieData avg_dev SpaceGroupNumber</th>
      <th>MagpieData mode SpaceGroupNumber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
      <td>(Ag, Br)</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>12.0</td>
      <td>41.0</td>
      <td>6.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>225.0</td>
      <td>161.0</td>
      <td>144.50</td>
      <td>80.500000</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
      <td>(Ag, Cl)</td>
      <td>17.0</td>
      <td>47.0</td>
      <td>30.0</td>
      <td>32.0</td>
      <td>15.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>225.0</td>
      <td>161.0</td>
      <td>144.50</td>
      <td>80.500000</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
      <td>(Ag, N)</td>
      <td>7.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>17.0</td>
      <td>15.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>194.0</td>
      <td>225.0</td>
      <td>31.0</td>
      <td>201.75</td>
      <td>11.625000</td>
      <td>194.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ag1Te1</td>
      <td>820.0</td>
      <td>0.850</td>
      <td>1.0</td>
      <td>(Ag, Te)</td>
      <td>47.0</td>
      <td>52.0</td>
      <td>5.0</td>
      <td>49.5</td>
      <td>2.500000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>152.0</td>
      <td>225.0</td>
      <td>73.0</td>
      <td>188.50</td>
      <td>36.500000</td>
      <td>152.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ag2O1</td>
      <td>785.0</td>
      <td>1.200</td>
      <td>1.0</td>
      <td>(Ag, O)</td>
      <td>8.0</td>
      <td>47.0</td>
      <td>39.0</td>
      <td>34.0</td>
      <td>17.333333</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>225.0</td>
      <td>213.0</td>
      <td>154.00</td>
      <td>94.666667</td>
      <td>225.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 137 columns</p>
</div>




```python
df_magpie.columns[5:]
```




    Index(['MagpieData minimum Number', 'MagpieData maximum Number',
           'MagpieData range Number', 'MagpieData mean Number',
           'MagpieData avg_dev Number', 'MagpieData mode Number',
           'MagpieData minimum MendeleevNumber',
           'MagpieData maximum MendeleevNumber',
           'MagpieData range MendeleevNumber', 'MagpieData mean MendeleevNumber',
           ...
           'MagpieData range GSmagmom', 'MagpieData mean GSmagmom',
           'MagpieData avg_dev GSmagmom', 'MagpieData mode GSmagmom',
           'MagpieData minimum SpaceGroupNumber',
           'MagpieData maximum SpaceGroupNumber',
           'MagpieData range SpaceGroupNumber', 'MagpieData mean SpaceGroupNumber',
           'MagpieData avg_dev SpaceGroupNumber',
           'MagpieData mode SpaceGroupNumber'],
          dtype='object', length=132)



chemicalFormula들의 norm 값


```python
ep_feat = MultipleFeaturizer([cf.Stoichiometry()])
df_norm = ep_feat.featurize_dataframe(df_clean, col_id ="composition")
print(df_norm.shape)
df_norm.head()
```
    (467, 11)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
      <th>composition</th>
      <th>0-norm</th>
      <th>2-norm</th>
      <th>3-norm</th>
      <th>5-norm</th>
      <th>7-norm</th>
      <th>10-norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
      <td>(Ag, Br)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
      <td>(Ag, Cl)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
      <td>(Ag, N)</td>
      <td>2</td>
      <td>0.790569</td>
      <td>0.759147</td>
      <td>0.750616</td>
      <td>0.750049</td>
      <td>0.750001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ag1Te1</td>
      <td>820.0</td>
      <td>0.850</td>
      <td>1.0</td>
      <td>(Ag, Te)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ag2O1</td>
      <td>785.0</td>
      <td>1.200</td>
      <td>1.0</td>
      <td>(Ag, O)</td>
      <td>2</td>
      <td>0.745356</td>
      <td>0.693361</td>
      <td>0.670782</td>
      <td>0.667408</td>
      <td>0.666732</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_norm.columns[5:]
```




    Index(['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm'], dtype='object')



chemicalFormula들의 avg 값


```python
ep_feat = MultipleFeaturizer([cf.ValenceOrbital(props=["avg"])])
df_avg = ep_feat.featurize_dataframe(df_clean, col_id ="composition")
print(df_avg.shape)
df_avg.head()
```
    (467, 9)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
      <th>composition</th>
      <th>avg s valence electrons</th>
      <th>avg p valence electrons</th>
      <th>avg d valence electrons</th>
      <th>avg f valence electrons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
      <td>(Ag, Br)</td>
      <td>1.500000</td>
      <td>2.500000</td>
      <td>10.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
      <td>(Ag, Cl)</td>
      <td>1.500000</td>
      <td>2.500000</td>
      <td>5.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
      <td>(Ag, N)</td>
      <td>1.750000</td>
      <td>2.250000</td>
      <td>2.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ag1Te1</td>
      <td>820.0</td>
      <td>0.850</td>
      <td>1.0</td>
      <td>(Ag, Te)</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ag2O1</td>
      <td>785.0</td>
      <td>1.200</td>
      <td>1.0</td>
      <td>(Ag, O)</td>
      <td>1.333333</td>
      <td>1.333333</td>
      <td>6.666667</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_avg.columns[5:]
```




    Index(['avg s valence electrons', 'avg p valence electrons',
           'avg d valence electrons', 'avg f valence electrons'],
          dtype='object')



chemicalFormula들의 norm 값


```python
ep_feat = MultipleFeaturizer([cf.ElementFraction()])
df_ele = ep_feat.featurize_dataframe(df_clean, col_id ="composition")
print(df_ele.shape)
df_ele.head()
```
    (467, 108)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chemicalFormula Clean</th>
      <th>index</th>
      <th>Band gap values Clean</th>
      <th>Reliability</th>
      <th>composition</th>
      <th>H</th>
      <th>He</th>
      <th>Li</th>
      <th>Be</th>
      <th>B</th>
      <th>...</th>
      <th>Pu</th>
      <th>Am</th>
      <th>Cm</th>
      <th>Bk</th>
      <th>Cf</th>
      <th>Es</th>
      <th>Fm</th>
      <th>Md</th>
      <th>No</th>
      <th>Lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ag1Br1</td>
      <td>808.5</td>
      <td>3.485</td>
      <td>1.0</td>
      <td>(Ag, Br)</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ag1Cl1</td>
      <td>793.5</td>
      <td>4.190</td>
      <td>1.0</td>
      <td>(Ag, Cl)</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ag1N3</td>
      <td>783.0</td>
      <td>3.900</td>
      <td>1.0</td>
      <td>(Ag, N)</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ag1Te1</td>
      <td>820.0</td>
      <td>0.850</td>
      <td>1.0</td>
      <td>(Ag, Te)</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ag2O1</td>
      <td>785.0</td>
      <td>1.200</td>
      <td>1.0</td>
      <td>(Ag, O)</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 108 columns</p>
</div>




```python
df_ele.columns[5:]
```




    Index(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
           ...
           'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
          dtype='object', length=103)



## get_composition을 활용한 chemicalFormula의 특성들을 합침


```python
f = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                       cf.ValenceOrbital(props=["avg"]), cf.ElementFraction()])
```


```python
df_feature=pd.DataFrame()
df_feature['composition'] =df_clean['composition'] 

df_feature= f.featurize_dataframe(df_feature,col_id='composition')
df_feature.head()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>composition</th>
      <th>0-norm</th>
      <th>2-norm</th>
      <th>3-norm</th>
      <th>5-norm</th>
      <th>7-norm</th>
      <th>10-norm</th>
      <th>MagpieData minimum Number</th>
      <th>MagpieData maximum Number</th>
      <th>MagpieData range Number</th>
      <th>...</th>
      <th>Pu</th>
      <th>Am</th>
      <th>Cm</th>
      <th>Bk</th>
      <th>Cf</th>
      <th>Es</th>
      <th>Fm</th>
      <th>Md</th>
      <th>No</th>
      <th>Lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Ag, Br)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Ag, Cl)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>17.0</td>
      <td>47.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Ag, N)</td>
      <td>2</td>
      <td>0.790569</td>
      <td>0.759147</td>
      <td>0.750616</td>
      <td>0.750049</td>
      <td>0.750001</td>
      <td>7.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Ag, Te)</td>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>47.0</td>
      <td>52.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Ag, O)</td>
      <td>2</td>
      <td>0.745356</td>
      <td>0.693361</td>
      <td>0.670782</td>
      <td>0.667408</td>
      <td>0.666732</td>
      <td>8.0</td>
      <td>47.0</td>
      <td>39.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 246 columns</p>
</div>



composition column은 학습할 때 필요 없기 때문에 drop


```python
df_feature_2=df_feature.drop(['composition'], axis=1)
print("shape : ",df_feature_2.shape)
df_feature_2.head()
```

    shape :  (467, 245)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0-norm</th>
      <th>2-norm</th>
      <th>3-norm</th>
      <th>5-norm</th>
      <th>7-norm</th>
      <th>10-norm</th>
      <th>MagpieData minimum Number</th>
      <th>MagpieData maximum Number</th>
      <th>MagpieData range Number</th>
      <th>MagpieData mean Number</th>
      <th>...</th>
      <th>Pu</th>
      <th>Am</th>
      <th>Cm</th>
      <th>Bk</th>
      <th>Cf</th>
      <th>Es</th>
      <th>Fm</th>
      <th>Md</th>
      <th>No</th>
      <th>Lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>12.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>17.0</td>
      <td>47.0</td>
      <td>30.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.790569</td>
      <td>0.759147</td>
      <td>0.750616</td>
      <td>0.750049</td>
      <td>0.750001</td>
      <td>7.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>17.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0.707107</td>
      <td>0.629961</td>
      <td>0.574349</td>
      <td>0.552045</td>
      <td>0.535887</td>
      <td>47.0</td>
      <td>52.0</td>
      <td>5.0</td>
      <td>49.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0.745356</td>
      <td>0.693361</td>
      <td>0.670782</td>
      <td>0.667408</td>
      <td>0.666732</td>
      <td>8.0</td>
      <td>47.0</td>
      <td>39.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 245 columns</p>
</div>



## df_feature_2 전처리

std가0인 columns는 학습데이터에서 제외


```python
finding_columns=df_feature_2.loc[:, df_feature.std() == 0]
findling_columns_list=finding_columns.columns
findling_columns_list
df_feature_2 = df_feature_2.drop(findling_columns_list,axis=1)
df_feature_2.shape
```




    (467, 215)



pearson correlation method

-0.95 이하 또는 +0.95 이상의 상관관계를 가지는 column는 제외


```python
# Remove Highly correlated Features
# using notes here for methodology: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

features_corr_df = df_feature_2.corr(method="pearson").abs()
features_corr_df.iloc[:5, :5] # Preview the first 5 rows/columns of the correlation matrix
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0-norm</th>
      <th>2-norm</th>
      <th>3-norm</th>
      <th>5-norm</th>
      <th>7-norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-norm</th>
      <td>1.000000</td>
      <td>0.807060</td>
      <td>0.747765</td>
      <td>0.684427</td>
      <td>0.653834</td>
    </tr>
    <tr>
      <th>2-norm</th>
      <td>0.807060</td>
      <td>1.000000</td>
      <td>0.994481</td>
      <td>0.976386</td>
      <td>0.962693</td>
    </tr>
    <tr>
      <th>3-norm</th>
      <td>0.747765</td>
      <td>0.994481</td>
      <td>1.000000</td>
      <td>0.993388</td>
      <td>0.984962</td>
    </tr>
    <tr>
      <th>5-norm</th>
      <td>0.684427</td>
      <td>0.976386</td>
      <td>0.993388</td>
      <td>1.000000</td>
      <td>0.998202</td>
    </tr>
    <tr>
      <th>7-norm</th>
      <td>0.653834</td>
      <td>0.962693</td>
      <td>0.984962</td>
      <td>0.998202</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### before removing correlated features


```python
def plot_corr(data):
    fig1, ax1 = plt.subplots(figsize=(10,5))
    c = ax1.pcolor(data,cmap="Blues")
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel('Feature Numbers')
    ax1.set_ylabel('Feature Numbers')
    ax1.set_aspect('equal')
    plt.colorbar(c,ax=ax1)
    plt.show()
```


```python
plot_corr(features_corr_df)
```


    
![41_0](/assets/images/bandgap_predict/output_41_0.png)
    


상관계수가 0.95이상인 열 제거


```python
upper = features_corr_df.where(np.triu(np.ones(features_corr_df.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
features_df_lowcorr = df_feature_2.drop(columns=to_drop)
features_df_lowcorr.shape
```




    (467, 163)



recalculate the correlation matrix so we can compare


```python
features_corr_df_update = features_df_lowcorr.corr(method="pearson").abs()
```

### after removing correlated features 


```python
plot_corr(features_corr_df_update)
```


    
![47_0](/assets/images/bandgap_predict/output_47_0.png)
    


## plot correlation after removing highly correlated features


```python
colormap = plt.cm.Greens

fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
c1 = ax1.pcolor(features_corr_df,cmap=colormap)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.xaxis.set_ticks_position('top')
ax1.xaxis.set_label_position('top')
ax1.set_xlabel('Feature Numbers')
ax1.set_ylabel('Feature Numbers')
ax1.set_aspect('equal')

plt.colorbar(c1,ax=ax1)



c2 = ax2.pcolor(features_corr_df_update,cmap=colormap)
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax2.set_xlabel('Feature Numbers')
ax2.set_ylabel('Feature Numbers')
ax2.set_aspect('equal')
plt.colorbar(c2,ax=ax2)
plt.show()
```


    
![49_0](/assets/images/bandgap_predict/output_49_0.png)
    


## 표준편차가0, 높은 상관계수 열 제거한 상태에서 MinMaxScaler


```python
minmax_feature = MinMaxScaler().fit_transform(features_df_lowcorr)
df_minmax_feature=pd.DataFrame(minmax_feature, columns=features_df_lowcorr.columns)
print(df_minmax_feature.shape)
df_minmax_feature.head()
```

    (467, 163)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0-norm</th>
      <th>2-norm</th>
      <th>MagpieData minimum Number</th>
      <th>MagpieData maximum Number</th>
      <th>MagpieData range Number</th>
      <th>MagpieData mean Number</th>
      <th>MagpieData mode Number</th>
      <th>MagpieData minimum MendeleevNumber</th>
      <th>MagpieData maximum MendeleevNumber</th>
      <th>MagpieData mean MendeleevNumber</th>
      <th>...</th>
      <th>Re</th>
      <th>Os</th>
      <th>Ir</th>
      <th>Pt</th>
      <th>Hg</th>
      <th>Tl</th>
      <th>Pb</th>
      <th>Bi</th>
      <th>Th</th>
      <th>U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.4000</td>
      <td>0.482759</td>
      <td>0.142857</td>
      <td>0.461538</td>
      <td>0.400</td>
      <td>0.673684</td>
      <td>0.975610</td>
      <td>0.783784</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.1750</td>
      <td>0.482759</td>
      <td>0.357143</td>
      <td>0.346154</td>
      <td>0.175</td>
      <td>0.673684</td>
      <td>0.951220</td>
      <td>0.777027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.284959</td>
      <td>0.0500</td>
      <td>0.482759</td>
      <td>0.476190</td>
      <td>0.153846</td>
      <td>0.050</td>
      <td>0.673684</td>
      <td>0.658537</td>
      <td>0.753378</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.5500</td>
      <td>0.540230</td>
      <td>0.059524</td>
      <td>0.570513</td>
      <td>0.550</td>
      <td>0.673684</td>
      <td>0.853659</td>
      <td>0.750000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.130591</td>
      <td>0.0625</td>
      <td>0.482759</td>
      <td>0.464286</td>
      <td>0.371795</td>
      <td>0.550</td>
      <td>0.673684</td>
      <td>0.780488</td>
      <td>0.680180</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 163 columns</p>
</div>



## X, y 데이터


```python
X=df_minmax_feature
y=y_value
```

# training data  --> 모델  --> test 검증 : train(0.9), test (0.1)


```python
seed =np.random.seed(22)  # random seed를 결정
```


```python
test_fraction =0.1
```


```python
X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=test_fraction, shuffle=True, random_state=seed)
```

## X,y의 train, test가 적절히 잘 분배되었는지 확인


```python
fig, (ax1, ax2) = plt.subplots(2, figsize=(10,5), sharex = True, gridspec_kw={'hspace': 0})
fig.set_tight_layout(False)
myarray = df_clean["Band gap values Clean"]

bins = np.true_divide(range(28),2)

l1 = sns.distplot(y_train.astype("float"), hist = True, norm_hist = True, kde = False, 
                  bins = bins, hist_kws={"edgecolor": "white"}, label = 'training set', ax = ax1)
l2 = sns.distplot(y_test.astype("float"), hist = True, norm_hist = True, kde = False, 
                  bins = bins, hist_kws={"edgecolor": "white", "color": "orange"}, label = 'test set', ax = ax2)
l3 = sns.distplot(myarray, hist = True, norm_hist = True, kde = False, 
                  bins = bins, hist_kws={"histtype": "step","linewidth": 3, "alpha": 1, "color": "grey"}, ax = ax1)
l4 = sns.distplot(myarray, hist = True, norm_hist = True, kde = False, 
                  bins = bins, hist_kws={"histtype": "step","linewidth": 3, "alpha": 1, "color": "grey"}, 
                  label = 'full dataset', ax = ax2)


ax1.set_xticks(range(14))
ax2.set_xticks(range(14))
ax2.xaxis.label.set_visible(False)
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
fig.suptitle('Comparing histograms of the train/test split')
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.legend(handles, labels, loc = 'center left', bbox_to_anchor=(1, 0.5),prop={'size': 16})
plt.xlabel('Measured Bandgap (eV)')
_ = plt.ylabel('Density')
```


    
![59_0](/assets/images/bandgap_predict/output_59_0.png)
    



```python
import utils
```


```python
from helper_functions import *
```


```python
from sklearn.model_selection import GridSearchCV  # 매개변수를 자동으로 변수 설정해서 최적 찾기 위한 방법
from sklearn.model_selection import cross_validate  # 교차 검증
from sklearn.model_selection import KFold  # 몇번 나눌지
from sklearn.model_selection import cross_val_score  # 교차검증시의 성능지표
from sklearn.metrics import r2_score, mean_squared_error  # SVR 모델의 예측과 실제의 결과를 확인하는 성능지표
```


```python
from sklearn.ensemble import RandomForestRegressor
```


```python
from sklearn.model_selection import cross_validate, GridSearchCV, ParameterGrid
from sklearn.model_selection import KFold, RepeatedKFold
```


```python
model = RandomForestRegressor(random_state=seed, bootstrap=True ).fit(X_train,y_train)
```


```python
cv = KFold(n_splits = 5, shuffle = True, random_state=seed)
```


```python
parameter_candidates = {
    'n_estimators': [500],   
    'max_features': [20],
    'max_depth': [10],
    "min_samples_leaf": [1],
    "min_samples_split": [2]  
}
```


```python
grid=GridSearchCV(estimator=model,
                 param_grid=parameter_candidates,
                 cv=cv)
```


```python
grid.fit(X_train, y_train)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=True),
                 estimator=RandomForestRegressor(),
                 param_grid={'max_depth': [10], 'max_features': [20],
                             'min_samples_leaf': [1], 'min_samples_split': [2],
                             'n_estimators': [500]})




```python
best_parameters =grid.best_params_
print(best_parameters)
```

    {'max_depth': 10, 'max_features': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    


```python
final_model=RandomForestRegressor(**best_parameters, random_state=seed)
final_model.fit(X_train, y_train)
```




    RandomForestRegressor(max_depth=10, max_features=20, n_estimators=500)




```python
Train_pred=final_model.predict(X_train)
Test_pred=final_model.predict(X_test)
```


```python
utils.plot_act_vs_pred(y_test,Test_pred)
score=r2_score(y_test,Test_pred)
rmse =np.sqrt(mean_squared_error(y_test,Test_pred ))
print("r2 score:{:0.3f}, rmse:{:0.3f}".format(score, rmse))
```

    r2 score:0.840, rmse:0.932
    


    
![73_1](/assets/images/bandgap_predict/output_73_1.png)
    



```python
parity_plots_side_by_side(y_train,Train_pred ,y_test,
                          Test_pred ,title_left="Training Data Parity Plot (RF)",title_right="Test Data Parity Plot (SVR)") # build both plots
parity_stats_side_by_side(y_train,Train_pred,y_test,
                          Test_pred ,"Training Set (train)","test set")
```


    
![74_0](/assets/images/bandgap_predict/output_74_0.png)
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Error Metric</th>
      <th>Training Set (train)</th>
      <th>test set</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RMSE</td>
      <td>0.4568 (eV)</td>
      <td>0.9321 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RMSE/std</td>
      <td>0.2004</td>
      <td>0.4005</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.3359 (eV)</td>
      <td>0.601 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R2</td>
      <td>0.9598</td>
      <td>0.8396</td>
      <td>(1.0 for perfect prediction)</td>
    </tr>
  </tbody>
</table>
</div>



## column이 많기 때문에 특성중요도가 높은 것만 학습

## 1번째 방법 Feature importance based on mean decrease in impurity


```python
m = final_model
```


```python
import time

start_time = time.time()
importances = m.feature_importances_
std = np.std([m.feature_importances_ for tree in m.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
```

    Elapsed time to compute the importances: 28.372 seconds
    


```python
forest_importances = pd.Series(importances, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
```


    
![79_0](/assets/images/bandgap_predict/output_79_0.png)
    



```python
forest_importances_df = pd.DataFrame(forest_importances, index = X_train.columns, columns = ["Importance"])
forest_importances_df_sorted = forest_importances_df.sort_values(by='Importance', ascending = False)
forest_importances_df_sorted.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MagpieData range Electronegativity</th>
      <td>0.063350</td>
    </tr>
    <tr>
      <th>MagpieData range NpValence</th>
      <td>0.060658</td>
    </tr>
    <tr>
      <th>MagpieData maximum NpUnfilled</th>
      <td>0.055583</td>
    </tr>
    <tr>
      <th>MagpieData maximum MendeleevNumber</th>
      <td>0.039980</td>
    </tr>
    <tr>
      <th>MagpieData maximum Electronegativity</th>
      <td>0.036747</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
forest_importances_df_sorted[:20].plot.bar(yerr=std[0:20], ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
```


    
![81_0](/assets/images/bandgap_predict/output_81_0.png)
    



```python
# 특성중요도 20개
top20_importance_feature = forest_importances_df_sorted[:20].index
```


```python
top20_importance_feature
```




    Index(['MagpieData range Electronegativity', 'MagpieData range NpValence',
           'MagpieData maximum NpUnfilled', 'MagpieData maximum MendeleevNumber',
           'MagpieData maximum Electronegativity', 'MagpieData mean NUnfilled',
           'MagpieData maximum NUnfilled', 'MagpieData mean CovalentRadius',
           'MagpieData mode MeltingT', 'MagpieData mean Number',
           'MagpieData mode CovalentRadius', 'MagpieData minimum CovalentRadius',
           'MagpieData mean NpUnfilled', 'MagpieData minimum NValence',
           'MagpieData maximum Number', 'MagpieData mean MeltingT',
           'MagpieData mean NValence', 'MagpieData minimum MeltingT',
           'MagpieData minimum MendeleevNumber',
           'MagpieData mode Electronegativity'],
          dtype='object')




```python
# 특성 중요도가 제일 높았던 20개의 feature값을 X로
X = df_minmax_feature[top20_importance_feature]
```


```python
# train. test 분류
X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=test_fraction, shuffle=True, random_state=seed) 
```


```python
model_1 = RandomForestRegressor(random_state=seed, bootstrap=True ).fit(X_train,y_train)
```


```python
cv = KFold(n_splits = 5, shuffle = True, random_state=seed)
```


```python
parameter_candidates = {
    'n_estimators': [500],   
    'max_features': [20],
    'max_depth': [10],
    "min_samples_leaf": [1],
    "min_samples_split": [2]  
}
```


```python
grid=GridSearchCV(estimator=model,
                 param_grid=parameter_candidates,
                 cv=cv)
```


```python
grid.fit(X_train, y_train)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=True),
                 estimator=RandomForestRegressor(),
                 param_grid={'max_depth': [10], 'max_features': [20],
                             'min_samples_leaf': [1], 'min_samples_split': [2],
                             'n_estimators': [500]})




```python
best_parameters =grid.best_params_
print(best_parameters)
```

    {'max_depth': 10, 'max_features': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    


```python
final_model_1=RandomForestRegressor(**best_parameters, random_state=seed)
```


```python
final_model_1.fit(X_train, y_train)
```




    RandomForestRegressor(max_depth=10, max_features=20, n_estimators=500)




```python
Train_pred=final_model_1.predict(X_train)
Test_pred=final_model_1.predict(X_test)
```


```python
utils.plot_act_vs_pred(y_test,Test_pred)
score=r2_score(y_test,Test_pred)
rmse =np.sqrt(mean_squared_error(y_test,Test_pred ))
print("r2 score:{:0.3f}, rmse:{:0.3f}".format(score, rmse))
```

    r2 score:0.701, rmse:1.134
    


    
![95_1](/assets/images/bandgap_predict/output_95_1.png)
    


r2 score:0.665, rmse:1.200
RMSE0.3641 (eV) 1.2003 (eV)
RMSE/std0.1579 0.5788
MAE0.2458 (eV) 0.8102 (eV)
R20.9751       0.665


```python
parity_plots_side_by_side(y_train,Train_pred ,y_test,
                          Test_pred ,title_left="Training Data Parity Plot (RF)",title_right="Test Data Parity Plot (SVR)") # build both plots
parity_stats_side_by_side(y_train,Train_pred,y_test,
                          Test_pred ,"Training Set (train)","test set")
```


    
![97_0](/assets/images/bandgap_predict/output_97_0.png)
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Error Metric</th>
      <th>Training Set (train)</th>
      <th>test set</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RMSE</td>
      <td>0.427 (eV)</td>
      <td>1.134 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RMSE/std</td>
      <td>0.1851</td>
      <td>0.5468</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.3126 (eV)</td>
      <td>0.7489 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R2</td>
      <td>0.9657</td>
      <td>0.701</td>
      <td>(1.0 for perfect prediction)</td>
    </tr>
  </tbody>
</table>
</div>



# 2번째 방법 Feature importance based on feature permutation


```python
# 2번째 방법을 하기 위해 X,y 데이터를 전처리 이전의 데이터로 복구

X = df_minmax_feature.iloc[:,:163]
y = y_value
X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=test_fraction, shuffle=True, random_state=seed) 
```


```python
from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    m, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=X_train.columns)
```

    Elapsed time to compute the importances: 67.243 seconds
    


```python
forest_importances_df_2 = pd.DataFrame(forest_importances, index = X_train.columns, columns = ["Importance"])
forest_importances_df_2_sorted = forest_importances_df_2.sort_values(by='Importance', ascending = False)
forest_importances_df_2_sorted.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MagpieData range Electronegativity</th>
      <td>0.035507</td>
    </tr>
    <tr>
      <th>MagpieData mean CovalentRadius</th>
      <td>0.024377</td>
    </tr>
    <tr>
      <th>MagpieData maximum Electronegativity</th>
      <td>0.024080</td>
    </tr>
    <tr>
      <th>MagpieData mean NUnfilled</th>
      <td>0.019647</td>
    </tr>
    <tr>
      <th>MagpieData range NpValence</th>
      <td>0.019100</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
forest_importances_df_2_sorted[:20].plot.bar(yerr=result.importances_std[0:20], ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
```


    
![102_0](/assets/images/bandgap_predict/output_102_0.png)
    



```python
# 특성중요도 20개
top20_importance_feature_2 = forest_importances_df_2_sorted[:20].index
```


```python
# 특성 중요도가 제일 높았던 20개의 feature값을 X로
X = df_minmax_feature[top20_importance_feature_2]
```


```python
# train. test 분류
X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=test_fraction, shuffle=True, random_state=seed) 
```


```python
model_2 = RandomForestRegressor(random_state=seed, bootstrap=True ).fit(X_train,y_train)
```


```python
cv = KFold(n_splits = 5, shuffle = True, random_state=seed)
```


```python
parameter_candidates = {
    'n_estimators': [500],   
    'max_features': [20],
    'max_depth': [10],
    "min_samples_leaf": [1],
    "min_samples_split": [2]  
}
```


```python
grid=GridSearchCV(estimator=model_2,
                 param_grid=parameter_candidates,
                 cv=cv)
```


```python
grid.fit(X_train, y_train)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=True),
                 estimator=RandomForestRegressor(),
                 param_grid={'max_depth': [10], 'max_features': [20],
                             'min_samples_leaf': [1], 'min_samples_split': [2],
                             'n_estimators': [500]})




```python
best_parameters =grid.best_params_
print(best_parameters)
```

    {'max_depth': 10, 'max_features': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    


```python
final_model_2=RandomForestRegressor(**best_parameters, random_state=seed)
```


```python
final_model_2.fit(X_train, y_train)
```




    RandomForestRegressor(max_depth=10, max_features=20, n_estimators=500)




```python
Train_pred=final_model_2.predict(X_train)
Test_pred=final_model_2.predict(X_test)
```


```python
utils.plot_act_vs_pred(y_test,Test_pred)
score=r2_score(y_test,Test_pred)
rmse =np.sqrt(mean_squared_error(y_test,Test_pred ))
print("r2 score:{:0.3f}, rmse:{:0.3f}".format(score, rmse))
```

    r2 score:0.883, rmse:0.749
    


    
![115_1](/assets/images/bandgap_predict/output_115_1.png)
    



```python
parity_plots_side_by_side(y_train,Train_pred ,y_test,
                          Test_pred ,title_left="Training Data Parity Plot (RF)",title_right="Test Data Parity Plot (SVR)") # build both plots
parity_stats_side_by_side(y_train,Train_pred,y_test,
                          Test_pred ,"Training Set (train)","test set")
```


    
![116_0](/assets/images/bandgap_predict/output_116_0.png)
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Error Metric</th>
      <th>Training Set (train)</th>
      <th>test set</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RMSE</td>
      <td>0.4434 (eV)</td>
      <td>0.7488 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RMSE/std</td>
      <td>0.1933</td>
      <td>0.3417</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.3231 (eV)</td>
      <td>0.613 (eV)</td>
      <td>(0.0 for perfect prediction)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R2</td>
      <td>0.9626</td>
      <td>0.8832</td>
      <td>(1.0 for perfect prediction)</td>
    </tr>
  </tbody>
</table>
</div>



# 두 번째 방법의 score값이 성능이 좋다.

# def 사용하여 SiN 반도체 재료 밴드갭 예측


```python
def prediction(element, target_value):
    df_formula = pd.DataFrame() # 새로운 df
    df_formula['formula'] = [element] # 예측할 element 변수 입력
    df_formula['target'] = [target_value] # 우리가 알고있는 값,구글링을 통해서 알고 있는값 (실제 값)
    df_formula['Composition'] = df_formula['formula'].apply(get_composition) # get_composition 함수 적용
    
    df_feature_test=pd.DataFrame() # 학습할 df 생성
    df_feature_test['Composition']=df_formula['Composition'] # df 복사
    df_feature_test= f.featurize_dataframe(df_feature_test,col_id='Composition') # 학습할 df의 feature 생성
    
    df_feature_test=df_feature_test[X_train.columns] # 두 번째 방법의 특성중요도가높은 20개의 column
    features_df_lowcorr_1 = features_df_lowcorr[X_train.columns] # 기존의 467 rows 데이터
    features_df_lowcorr_1=features_df_lowcorr_1.append(df_feature_test) # 학습할 데이터 행 추가
    
    minmax_feature_1 = MinMaxScaler().fit_transform(features_df_lowcorr_1) # 합친 df의 minmaxscaler 적용
    df_minmax_feature_1=pd.DataFrame(minmax_feature_1, columns=features_df_lowcorr_1.columns) 
    
    df_feature_corr=df_minmax_feature_1[467:] # 기존의 데이터 제거해서 학습할 데이터 행(마지막 행) 추출 
    y_predicted = final_model_2.predict(df_feature_corr) # 기존의 모델에 학습데이터 적용
    df_formula["predict"] = y_predicted
    return df_formula
```


```python
prediction("SiN", 4.57) 
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>formula</th>
      <th>target</th>
      <th>Composition</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SiN</td>
      <td>4.57</td>
      <td>(Si, N)</td>
      <td>4.546774</td>
    </tr>
  </tbody>
</table>
</div>

정리
---

- 데이터 가져오기
- 데이터 전처리 및 가공  
1. get_composition 함수 사용 column 추가
2. 필요없는 column 제거
3. 상관계수가 높거나 표준편차가 0인 column 제거
4. MinMaxScaler로 데이터 scale  
- 데이터 학습
1. X : 반도체 화학식에 대한 정보, y : bandgap
2. X,y의 train,test 비율은 0.1
3. 적절히 잘 분배되었는지 그래프화하여 비교
4. GridSearchCV함수를 사용해 최적의 parameter값 도출 및 학습 - RF
5. feature permutation 을 활용해 특성중요도가 제일 높은 20개 외의 열 제거 및 학습  
- test셋 검증    
test 검증 결과 r2 score : 0.883, rmse : 0.749 도출
- def 함수 사용해 새로운 데이터(SiN 반도체) 예측 및 비교

한계점
---

- 도메인 지식이 부족한 데이터 분석하는 데에 어려움이 있었음.
- 데이터 전처리를 하는 것에 상당한 시간을 쏟음.
- 데이터 특성상 비효율적인 column이 많아 최대한 줄이고자 노력

배운점
---

- matminer, composition 라이브러리 사용
- 특성중요도를 추출하는 다양한 method가 있음을 알게됨
- 데이터 전처리의 중요성을 알게됨
