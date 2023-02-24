# Pet-Project-Credit-Scoring
Задача кредитного скоринга с данными с одного из хакатонов
## Используемые методы 
* Разведывательный анализ данных
* Предобработка данных
* Создание признаков
* Обучение моделей
* Отбор признаков
* Оптимизация гиперпараметров
* Интерпретация предсказаний модели
## Используемые библиотеки
*  Python, Jupyter Notebook
*  pandas, NumPy
*  Matplotlib, seaborn
*  scikit-learn
*  LightGBM, CatBoost, XGBoost
*  Keras
*  LightAutoML
*  Optuna
*  SHAP
## Описание ноутбуков
| Ноутбук |  Описание |
| :----------------------------|:-----------|
| [Preprocessing](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/preprocessing/preprocessing.py)| Функции для предобработки данных: предобработка категориальных признаков и заполнение пропущенных значений|
| [Feature Engeneering](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/feature_engeneering/feature_engineering.py)| Функции для создания признаков|
| [Preprocessing + Feature Engeneering](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/feature_engeneering/feature_engeneering.ipynb)| Предобработка данных и создание признаков|
| [Sklearn Models](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/sklearn_models.ipynb)| Обучение моделей из библиотеки  scikit-learn: логистическая регрессия, случайный лес, градиентный бустинг и сравнение результатов|
| [Keras + KerasTuner](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/keras_nn.ipynb)| Нейронная сеть на Keras и оптимизация гиперпараметров с помощью библиотеки KerasTuner     | 
| [CatBoost](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/catboost_training.ipynb)| Обучение CatBoost и отбор признаков встроенными методами библиотеки              | 
