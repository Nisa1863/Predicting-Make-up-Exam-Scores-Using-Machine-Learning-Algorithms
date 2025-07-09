import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("StudentsPerformance.csv")

print(df.shape)
print(df.columns)
print(df.head())
print(df.info())
print(df.isnull().sum())

col_names=["gender","race/ethnicity","parental level of education","lunch","Discontinuity","Vize Note","Final Note","Butunleme"]
df.columns=col_names
print(df.head())

print(df.value_counts())
print(df["Discontinuity"].value_counts())

X=df[["Vize Note","Final Note","gender","race/ethnicity","parental level of education","lunch","Discontinuity"]]
y=df["Butunleme"]

X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def calculate_metrics(true,predict):
    mae=mean_absolute_error(true,predict)
    mse=mean_squared_error(true,predict)
    rmse=np.sqrt(mean_squared_error(true,predict))
    r2_square=r2_score(true,predict)

    return mae,mse,r2_square,rmse

models={
    "LinearRegression":LinearRegression(),
    "Lasso":Lasso(),
    "Ridge":Ridge(),
    "K-Neighbors Regression":KNeighborsRegressor(),
    "DecisionTreeREgressor":DecisionTreeRegressor(),
    "Random Forest Regression":RandomForestRegressor()
}
for i in range (len(list(models))):

    model=list(models.values())[i]
    model.fit(X_train,y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

model_train_mae, model_train_mse, model_train_r2, model_train_rmse = calculate_metrics(y_train, y_train_pred)
model_test_mae, model_test_mse, model_test_r2, model_test_rmse = calculate_metrics(y_test, y_test_pred)

print(list(models.values())[i])

print("---------TRAIN----------")
print("RMSE",model_train_rmse)
print("MAE",model_train_mae)
print("MSE",model_train_mse)
print("R2",model_train_r2)

print("--------TEST------------")
print("RMSE",model_test_rmse)
print("MAE",model_test_mae)
print("MSE",model_test_mse)
print("R2",model_test_r2)

knn_params={"n_neighbors":[2,3,10,20,40,50]}
rf_params={
    "max_depth":[5,8,10,15,None],
    "max_features":["sqrt","log2",5,7,10],
    "min_samples_split":[2,8,12,20],
    "n_estimators":[100,200,300,1000]
}
from sklearn.model_selection import RandomizedSearchCV
randomcv_models=[ 
    ("KNN",KNeighborsRegressor(),knn_params),
    ("RF",RandomForestRegressor(),rf_params),
]
for name,model,params in randomcv_models:
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=6,cv=3,n_jobs=-1)

    randomcv.fit(X_train,y_train)
    print("best params for",name, randomcv.best_params_)


print("-----------------------")
print("knn best params",knn_params)
print("RF best param",rf_params)


best_knn=KNeighborsRegressor(n_neighbors=3)
best_knn.fit(X_train,y_train)
y_pred_knn = best_knn.predict(X_test)

mae, mse, r2, rmse = calculate_metrics(y_test, y_pred_knn)
print("---------KNN - Best Params ile-----------")
print("MAE:", mae)
print("MSE",mse)
print("RMSE:", rmse )
print("R2:", r2)

best_RF=RandomForestRegressor(n_estimators= 200, min_samples_split=20, max_features=7, max_depth=15)
best_RF.fit(X_train,y_train)
y_pred_RF=best_RF.predict(X_test)

mae,mse,r2,rmse=calculate_metrics(y_test,y_pred_RF)
print("---------RF-Best Parms Ile---------")
print("MAE:", mae)
print("RMSE:", rmse )
print("R2:", r2)
print("MSE",mse)
print("------------------------------------------------------------------------------------")
