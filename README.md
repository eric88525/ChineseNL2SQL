#  NL2SQL



+ install

  ```
  pip install -r requirements.txt
  ```

+ download model 放在 saved_models/

  ```python
  # m1
  https://drive.google.com/file/d/1Lln6HxJ1aNuswQnZrgR681fDdqT4U4dv/view?usp=sharing
  # m2
  https://drive.google.com/file/d/1e60AThMqOr3Tad10a6aD3XciJSeFCbEr/view?usp=sharing
  ```

  

+ 引用

```python
from N2S.model import *
path1 = './saved_models/M1.pt'
path2 = './saved_models/M2.pt'
model_type = 'hfl/chinese-bert-wwm'
model = NL2SQL(model_type,torch.device('cpu'),path1,path2,analyze=True)
```

+ 輸入格式

```python
data = {
    'question': '呃共有多少家公司17年定增，而且是为了配套融资的呀', 
    'headers': [['证券代码', '证券简称', '最新收盘价', '定增价除权后至今价格', '增发价格', '倒挂率', '定增年度', '增发目的'], 
    ['text', 'text', 'real', 'real', 'real', 'real', 'real', 'text']], 
    'table': [['002678.SZ', '珠江钢琴', 5.82, 9.37, 12.3, 62.09, 2017.0, '项目融资'], 
            ['002696.SZ', '百洋股份', 8.87, 10.53, 18.19, 84.26, 2017.0, '配套融资'], 
            ['002696.SZ', '百洋股份', 8.87, 12.0, 20.73, 73.93, 2017.0, '融资收购其他资产'], 
            ['300044.SZ', '赛为智能', 6.62, 7.24, 13.06, 91.49, 2017.0, '融资收购其他资产'], 
            ['300050.SZ', '世纪鼎利', 5.6, 12.64, 12.68, 44.3, 2017.0, '配套融资'], 
            ['300050.SZ', '世纪鼎利', 5.6, 12.22, 12.26, 45.82, 2017.0, '融资收购其他资产']]
}
```

  

+ 執行

```python=
model.go(data)
```



+ 結果

```python
  ['2017', '17'] number from question
  定增年度=2017 0.9999343156814575
  定增年度=17 0.00010633746569510549
  增发目的=项目融资 8.339100168086588e-05
  增发目的=配套融资 0.9999605417251587
  增发目的=融资收购其他资产 8.809728024061769e-05
  SELECT COUNT 证券简称 WHERE 定增年度=2017 AND 增发目的=配套融资 
```
+ 準確率

![](https://i.imgur.com/TJWh3DQ.png)

+ 資料跟天池主辦方要 我怕被告