import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data ={
    'Math':[8.5,7.0,9.0,6.5,7.5,8.0,5.5,6.0,9.5,8.0],
    'Physics':[7.5,6.0,8.5,6.0,7.0,8.0,5.0,6.0,9.0,7.5],
    'Chemistry':[8.0,6.5,9.0,5.5,7.0,7.5,5.0,5.5,9.0,8.0],
    'FinalScore':[24.0,19.5,26.5,18.0,21.5,23.5,15.5,17.5,27.5,23.5]
}
df=pd.DataFrame(data)

x=df[['Math','Physics','Chemistry']]
y=df['FinalScore']
model=LinearRegression()
model.fit (x,y)

y_pred=model.predict(x)

mse=mean_squared_error(y, y_pred)
r2=r2_score(y, y_pred)
print ("He so hoi quy:",model.coef_)
print ("Sai so trung binh binh phuong(mse):",mse)
print ("do chinh xac (r2):",r2)

new_student=[[7.5,8.0,7.0]]
predicted_score=model.predict(new_student)
print ("Diem du doan cho hoc sinh moi:", predicted_score[0])