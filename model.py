from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle 

df = pd.read_csv('car_price_dataset.csv')

X = df.drop(columns=['Price', 'Brand'])  # все признаки, кроме Price и Brand
y = df['Price']  # целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numeric_cols = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']
categorical_cols = ['Model', 'Fuel_Type', 'Transmission']

#Обработка данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols), #кодирование числовых признаков
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) #кодирование категориальных признаков
    ])

# Создание pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Обучение модели
pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:  
        pickle.dump(pipeline, f) #сохранение модели
