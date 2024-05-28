import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_excel(r'C:\Users\saras\OneDrive\Desktop\oose data sets\oosetrain.xlsx')
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,y, test_size=0.5)


gbr = GradientBoostingClassifier()
gbr.fit(X_train_scaled,y_train)
joblib.dump(gbr, 'model.joblib') 

if __name__ == "__main__":
    print("Model training and evaluation completed successfully.")
