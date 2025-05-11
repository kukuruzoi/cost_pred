from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Загрузка модели
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    plot_url = None
    
    if request.method == 'POST':
        # Получение данных из формы
        form_data = {
            'Year': float(request.form['Year']),
            'Engine_Size': float(request.form['Engine_Size']),
            'Mileage': float(request.form['Mileage']),
            'Model': request.form['Model'],
            'Fuel_Type': request.form['Fuel_Type'],
            'Transmission': request.form['Transmission'],
            'Doors': int(request.form['Doors']),
            'Owner_Count': int(request.form['Owner_Count'])
        }
        
        # Создание DataFrame
        sample_data = pd.DataFrame([form_data])
        
        # Предсказание
        prediction = pipeline.predict(sample_data)[0]

        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
    # Получаем все важности и названия признаков
            importances = pipeline.named_steps['model'].feature_importances_
            all_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
            
            # Преобразуем входные данные через препроцессор
            transformed_data = pipeline.named_steps['preprocessor'].transform(sample_data)
                
                # Конвертируем разреженную матрицу в плотную для анализа
            if hasattr(transformed_data, 'toarray'):
                transformed_data = transformed_data.toarray()
                
                # Получаем маску используемых признаков
            used_features_mask = (transformed_data != 0).any(axis=0)
                
                # Фильтруем только используемые признаки
            used_features = all_features[used_features_mask]
            used_importances = importances[used_features_mask]
                
                # Сортируем по важности
            sorted_idx = used_importances.argsort()[::-1]
            used_features = used_features[sorted_idx]
            used_importances = used_importances[sorted_idx]
                
                # Визуализация
            plt.figure(figsize=(10, 5))
            plot = sns.barplot(x=used_importances, y=used_features)
            plot.set_title('Важность используемых признаков', pad=20)
            plot.set_xlabel('Важность признака')
            plot.set_ylabel('Признаки')
                
                # Улучшаем читаемость длинных названий
            plt.tight_layout()
                
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
                
    
    return render_template(
        'index.html',
        prediction=round(prediction, 2) if prediction else None,
        plot_url=plot_url
    )

if __name__ == '__main__':
    app.run(debug=True)
