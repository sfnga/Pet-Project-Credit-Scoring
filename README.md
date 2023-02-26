# Pet-Project-Credit-Scoring
Задача кредитного скоринга с данными с одного из хакатонов
## Постановка задачи
Цифровое подразделение банка сталкивается с проблемами, связанными с конверсией потенциальных клиентов. Основная цель этого подразделения — увеличить привлечение клиентов через цифровые каналы. Подразделение было создано несколько лет назад, и основной задачей подразделения на протяжении этих лет было увеличение числа потенциальных клиентов, попадающих в воронку конверсии.
## Данные
* **ID** - уникальный индентификатор заявителя
* **Gender** - пол заявителя
* **DOB** - дата рождения заявителя
* **Lead_Creation_Date** - дата создания лида
* **City_Code** - анонимный код города
* **City_Category** -
* **Employer_Code** - анонимизированный код работодателя
* **Employer_Category1** -
* **Employer_Category2** - 
* **Monthly_Income** - ежемесячный доход в долларах
* **Customer_Existing_Primary_Bank_Code** - анонимизированный код банка
* **Primary_Bank_Type**
* **Contacted** - подтвержен контакт или нет
* **Source** - источник лида
* **Source_Category** - категория источника лида
* **Existing_EMI** - ежемесячный платеж существующих кредитов
* **Loan_Amount** - запрошенная сумма кредита
* **Loan_Period** - срок кредита в годах
* **Interest_Rate** - процентная ставка по запрошенной сумме кредита
* **EMI** - ежемесячный платеж по запрошенной сумме кредита
* **Var1** - анонимизированная категориальная переменная
* **Approved** - целевая переменная. Одобрен кредит или нет
## Метрика
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
| [EDA utils](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/eda/eda_utils.py)| Вспомогательные функции для выполнения разведывательного анализа данных|
| [EDA features](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/eda/eda_features.ipynb)| Анализ признаков|
| [EDA target](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/eda/eda_target.ipynb)| Изучение влияния признаков на целевую переменную|
| [Preprocessing](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/preprocessing/preprocessing.py)| Функции для предобработки данных: предобработка категориальных признаков и заполнение пропущенных значений|
| [Feature Engeneering](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/feature_engeneering/feature_engineering.py)| Функции для создания признаков|
| [Preprocessing + Feature Engeneering](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/feature_engeneering/feature_engeneering.ipynb)| Предобработка данных и создание признаков|
| [Sklearn Models](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/sklearn_models.ipynb)| Обучение моделей из библиотеки  scikit-learn: логистическая регрессия, случайный лес, градиентный бустинг и сравнение результатов|
| [XGBoost + Optuna](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/xgboost_tuning.ipynb)| Обучение XGBoost и оптимизация гиперпараметров с помощью Optuna|
| [CatBoost](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/catboost_training.ipynb)| Обучение CatBoost и отбор признаков встроенными методами библиотеки              | 
| [Keras + KerasTuner](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/keras_nn.ipynb)| Нейронная сеть на Keras и оптимизация гиперпараметров с помощью библиотеки KerasTuner     | 
| [LightAutoML](https://github.com/sfnga/Pet-Project-Credit-Scoring/blob/main/models/lightautoml.ipynb) | Решение задачи с помощью LightAutoML |
