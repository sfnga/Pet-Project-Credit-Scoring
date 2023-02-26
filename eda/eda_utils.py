import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def get_nulls(df):
    """
    Функция для подсчета количества и процента
    пропущенных значений признаков.
    
    Параметры
    ----------
    df - датафрейм для подсчета

    Возвращает
    -------
    nulls - количество пропущенных значений в признаках
    nulls_pct - процент пропущенных значений в признаках
    """
    nulls = df.isnull().sum()[lambda x: x > 0].sort_values(ascending=False)
    nulls = nulls.rename_axis('Признак')
    nulls_pct = nulls.reset_index(name='Процент')
    nulls_pct['Процент'] = (nulls_pct['Процент'] / len(df)).round(3)
    nulls = nulls.reset_index(name='Количество')
    return nulls, nulls_pct


def plot_nulls(df):
    """
    Строит график количества и процента
    пропущенных значений признаков.
    """
    nulls, nulls_pct = get_nulls(df)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig1 = sns.barplot(x='Признак', y='Количество', data=nulls, ax=axes[0])
    fig1.bar_label(fig1.containers[0])
    fig2 = sns.barplot(x='Признак', y='Процент', data=nulls_pct, ax=axes[1])
    fig2.bar_label(fig2.containers[0])
    fig.suptitle('Пропущенные значения в признаках')
    for item in axes[0].get_xticklabels():
        item.set_rotation('vertical')
    axes[0].set(xlabel=None)
    for item in axes[1].get_xticklabels():
        item.set_rotation('vertical')
    axes[1].set(xlabel=None)
    axes[0].set_title('Количество')
    axes[1].set_title('В процентах')
    sns.despine(fig)
    
    
def get_target_distribution(df, target):
    """
    Функция для нахождения
    распределения целевой переменной.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    target - название целевой переменной

    Возвращает
    -------
    target - количественное распределение целевой переменной
    nulls_pct - распределение целевой переменной в процентах
    """
    target = df[target].value_counts()
    target = target.rename_axis('Значение')
    target_pct = target.reset_index(name='Процент')
    target_pct['Процент'] = (target_pct['Процент'] / len(df)).round(3)
    target = target.reset_index(name='Количество')
    return target, target_pct


def plot_target_distribution(df, target):
    """
    Строит график распределения целевой переменной
    
    Параметры
    ----------
    df - датафрейм для подсчета
    target - название целевой переменной
    """
    target_values, target_pct = get_target_distribution(df, target)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig1 = sns.barplot(x='Значение',
                       y='Количество',
                       data=target_values,
                       ax=axes[0])
    fig1.bar_label(fig1.containers[0])
    fig2 = sns.barplot(x='Значение', y='Процент', data=target_pct, ax=axes[1])
    fig2.bar_label(fig2.containers[0])
    fig.suptitle('Распределение целевой переменной')
    axes[0].set_title('Количественное')
    axes[1].set_title('В процентах')
    sns.despine(fig)
    
    
def plot_categorical_feature(df, feature, target, ax=None, figsize=None, label_fontsize=None,title_fontsize=None):
    """
    Строит зависимость целевой переменной
    от категориального признака.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    feature - признак
    target - целевая переменная
    ax - область для построения графика
    figsize - размер графика
    """
    if figsize:
        plt.figure(figsize=figsize)
    elif not (ax):
        plt.figure(figsize=(12, 8))
    fig = sns.countplot(x=feature, hue=target, data=df, ax=ax)
    for container in fig.containers:
        fig.bar_label(container)
    if ax:
        ax.set_xlabel('Значение', fontsize=label_fontsize)
        ax.set_ylabel('Количество', fontsize=label_fontsize)
        ax.set_title(feature, fontsize=title_fontsize)
    else:
        fig.set(xlabel='Значение', ylabel='Количество', title=feature)
    sns.despine()
    

def find_mode_with_frequency(df, feature):
    """
    Находит моду признака и количество ее вхождения.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    feature - признак
    """
    counts = df[feature].value_counts()
    mode = counts.index[0]
    freq = counts.values[1]
    return mode, freq


def describe_with_mode(df, features=None):
    """
    Функция для нахождения описательных статистик:
    - количество ненулевых значений
    - среднее
    - стандартное отклонение
    - минимум 
    - 25 квартиль
    - 50 квартиль
    - 75 квартиль
    - максимум
    - мода
    - количество вхождений моды
    
    Параметры
    ----------
    df - датафрейм для подсчета
    features - признаки
    """
    if features:
        info = df[features].describe()
    else:
        info = df.describe()
        features = info.columns
    info.loc['nunique'] = df[features].nunique()
    for feature in features:
        mode, freq = find_mode_with_frequency(df, feature)
        info.loc['mode', feature] = mode
        info.loc['freq', feature] = freq
    return info


def plot_correlation(df, columns=None):
    """
    Строит график корреляции между признаками.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    columns - признаки для подсчета корреляции
    """
    if columns:
        correlation_matrix = df[columns].corr()
    else:
        correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix,
                annot=True,
                fmt='.2g',
                mask=np.triu(correlation_matrix),
                cmap='coolwarm')
    plt.title('Корреляционная матрица', fontsize=15)
    
    
def plot_numerical_feature(df, feature, max_quantile=None, figsize=None):
    """
    Строит графики распределения числовых признаков
    - ящик с усами
    - гистограмма
    
    Параметры
    ----------
    df - датафрейм для подсчета
    feature - признак
    max_quantile - значение признака будет рассматриваться от минимума до max_quantile квартиля
    figsize - размер графика
    """
    feature_values = df[feature]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(x=feature_values, ax=axes[0])
    if max_quantile:
        mask = df[feature].quantile(max_quantile)
        feature_values = feature_values[feature_values < mask]
    sns.histplot(x=feature_values, kde=True, ax=axes[1])
    axes[1].set_ylabel('Количество')
    axes[0].set_xlabel('Значение')
    axes[1].set_xlabel('Значение')
    plt.suptitle(feature, fontsize=15)
    sns.despine()
    
    
def plot_categorical_dependence(df, feature, figsize=(10, 5)):
    """
    Построение графика зависимости целевой переменной от категориальной.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    feature - признак
    figsize - размер графика
    """
    plt.figure(figsize=figsize)
    cat_df = (df.groupby('Approved')[feature].value_counts(normalize=True).
              rename('Percentage').mul(100).reset_index().sort_values(feature))
    cat_df['Percentage'] = np.round(cat_df['Percentage'])
    fig = sns.barplot(x=feature, y='Percentage', hue='Approved', data=cat_df)

    for container in fig.containers:
        fig.bar_label(container)

    plt.title(feature)
    sns.despine()
    plt.show()
    
    
def plot_numerical_dependence(df, feature):
    """
    Построение графика зависимости целевой переменной от числовой.
    
    Параметры
    ----------
    df - датафрейм для подсчета
    feature - признак
    """
    sns.displot(
        {
            'Approved : 1': df[df['Approved'] == 1][feature],
            'Not Approved : 0': df[df['Approved'] == 0][feature]
        },
        kind="kde",
        common_norm=False)
    plt.xlabel(feature)
    plt.title(feature)
    plt.show()