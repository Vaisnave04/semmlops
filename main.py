from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import numpy as np
iris = load_iris()
X= iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_class = np.round(y_pred).astype(int)
acc = accuracy_score(y_pred_class,y_test)

print(acc)