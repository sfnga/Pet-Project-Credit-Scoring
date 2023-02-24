import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class RareColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Преобразование редко встречающихся значений категориального признака.
    """
    def __init__(self, categories):
        """
        - categories - категории, которые необходимо преобразовать
        key - название категории, value - порог
        """
        self.categories = categories
        self.top_cats = {}

    def fit(self, df):
        # получаем категории, которые встречаются большее число раз, чем порог
        for column in self.categories:
            self.top_cats[column] = df[column].value_counts(
            )[lambda x: x > self.categories[column]].index

    def transform(self, X):
        df = X.copy()
        # не трогаем категории, которые встречаются большее число раз, чем порог
        # остальные заполняем значением another
        for column in self.categories:
            df[column] = np.where(df[column].isin(self.top_cats[column]),
                                  df[column], 'another')
        return df

    def fit_transform(self, df):
        self.fit(df)
        df = self.transform(df)
        return df
    
    
class Imputer(BaseEstimator, TransformerMixin):
    """
    Заполнение пропущенных значений.
    """
    def __init__(self,
                 features_to_imput: dict,
                 model_features: list = None,
                 regression_params: dict = {},
                 classification_params: dict = {}):
        """
        - features_to_imput: признаки, в которых необходимо заполнить пропущенные значения
        key - название признака, value - тип заполнения
        - model_features - признаки, по которым будет обучаться модель для заполнения пропущенных значений
        - regression_params - гиперпараметры модели регрессии
        - classification_params - гиперпараметры модели классификации
        """
        self.features_to_imput = features_to_imput
        self.model_features = model_features
        self.regression_params = regression_params
        self.classification_params = classification_params

        # словарь, в котором будем хранить значение для заполнения пропущенных
        self.imput_values: dict = {}
        # словарь, в котором будем хранить модели для предсказания пропущенных значений
        self.model_values: dict = {}

    def fit(self, X):
        for feature in self.features_to_imput:

            # заполнение средним
            if self.features_to_imput[feature] == 'mean':
                if X[feature].dtype == 'object':
                    print(
                        f'Impossible to fill categorical feature {feature} with mean'
                    )
                    return
                else:
                    self.imput_values[feature] = X[feature].mean()

            # заполнение медианой
            elif self.features_to_imput[feature] == 'median':
                if X[feature].dtype == 'object':
                    print(
                        f'Impossible to fill categorical feature {feature} with median'
                    )
                    return
                else:
                    self.imput_values[feature] = X[feature].mean()

            # заполнение модой
            elif self.features_to_imput[feature] == 'mode':
                self.imput_values[feature] = X[feature].mode().item()

            # заполнение константой и добавление нового признака, показывающего в каких строках значение признака было пропущено
            elif self.features_to_imput[feature] == 'indicator':
                self.imput_values[feature] = 'indicator'

            # удаление признака
            elif self.features_to_imput[feature] == 'drop':
                self.imput_values[feature] = 'drop'

            # заполнение на основе предсказаний модели
            elif self.features_to_imput[feature] == 'model':
                train_index = X[X[feature].notnull()].index
                X_train = X.loc[train_index, self.model_features]
                y_train = X.loc[train_index, feature]
                # решаем задачу регрессии, если у целевой переменной > 10 уникальных значений
                if y_train.dtype in [float, int] and y_train.nunique() > 10:
                    model = Ridge(**self.regression_params)
                else:
                    model = LogisticRegression(**self.classification_params)

                # будем считать признак числовым, если у него > 25 уникальных значений
                numeric = X_train.select_dtypes(
                    include=[int, float]).nunique()[lambda x: x > 25].index
                categorical = list(set(self.model_features) - set(numeric))

                column_transformer = ColumnTransformer([
                    ('ohe', OneHotEncoder(sparse=False,
                                          handle_unknown='ignore'),
                     categorical),
                    ('scaling', StandardScaler(), numeric),
                ])
                model = make_pipeline(column_transformer, model)
                model.fit(X_train, y_train)
                self.model_values[feature] = model

            # заполнение константой
            else:
                if X[feature].dtype in [int, float] and type(
                        self.features_to_imput[feature]) == str:
                    self.imput_values[feature] = 0
                else:
                    self.imput_values[feature] = self.features_to_imput[
                        feature]

    def transform(self, X):
        res = X.copy()
        for feature in self.imput_values:
            # в случае заполнения индикатором, заполняем константой и создаем признак,
            # показывающий, в каких строках были пропущенные значения
            if self.imput_values[feature] == 'indicator':
                res[f"{feature}_is_null"] = 0
                res.loc[res[feature].isnull(), f"{feature}_is_null"] = 1
                if res[feature].dtype in [int, float]:
                    fill_value = 0
                else:
                    fill_value = 'no_value'
                res[feature] = res[feature].fillna(fill_value)
            # удаление признака
            elif self.imput_values[feature] == 'drop':
                res = res.drop(columns=feature)
            else:
                res[feature] = res[feature].fillna(self.imput_values[feature])
        # заполнение на основе модели
        for feature in self.model_values:
            model = self.model_values[feature]
            test_index = res[res[feature].isnull()].index
            X_test = res.loc[test_index, self.model_features]
            res.loc[test_index, feature] = model.predict(X_test)
        return res

    def fit_transform(self, X):
        self.fit(X)
        res = self.transform(X)
        return res