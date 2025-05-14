# Car Price Prediction Project
Flask-приложение для прогнозирования рыночной стоимости автомобилей на основе их характеристик.

## Основные функции: 
- Прогноз цены – точный расчет стоимости по характеристикам авто  
- Графики – наглядное отображение статистики и корреляций  
- Анализ модели – какие параметры сильнее всего влияют на цену  
- Простой интерфейс – ввод данных в 3 клика  
 
## Технологии:
- Python 3.8+, Flask (бэкенд)
- Scikit-learn, Pandas (обучение модели)
- RndomForestReggression (модель)
- Matplotlib (визуализация)
- HTML (фронтенд)

## Локальный запуск
1. Клонируйте репозиторий:
```
git clone https://github.com/kukuruzoi/cost_pred.git
cd cost_pred
```
2. Поместите данные:
Добавьте car_price_dataset.csv в папку проекта, index.html в папку проекта/templates
3. Запустите файл model.py
4. Запустите приложение:
```
python app.py
```
5. Откройте в браузере:
```http://localhost:5000```

## Графики распределения переменных:
- Распределение Price 
![price_dist](https://github.com/user-attachments/assets/aa944a17-9eb3-4145-9373-3562f8c1b912)
- Распределение числовых признаков
![numeric_dist](https://github.com/user-attachments/assets/0ed8f16f-6332-4c5a-8cd9-ddbd8490e019)
- Распределение категориальных признаков
![categorical_dist](https://github.com/user-attachments/assets/0e96488d-5d83-46ac-8546-0b9768636354)

## Пример работы 
Вводимые данные:
![image](https://github.com/user-attachments/assets/94e9e053-d7bb-4120-87bc-b48ee2ea0bd5)
Вывод:
![image](https://github.com/user-attachments/assets/281fe2b0-6c90-4be1-aab4-8639a70eb22e)
В результате работы программы вывелось предсказанныая цена для заданных характеристик автомобиля и важность признаков для расчета цены.
