import pandas as pd


def get_creation_date(row):
    """Получение даты """
    row = row.split('/')
    return '-'.join(row[:-1]) + '-' + '2016'


def get_dob_date(row):
    row = row.split('/')
    return '-'.join(row[:-1]) + '-' + '19' + row[-1]


def get_age(df):
    df = df.copy()
    df['Lead_Creation_Date'] = df['Lead_Creation_Date'].apply(
        lambda x: get_creation_date(x))
    df['Lead_Creation_Date'] = pd.to_datetime(df['Lead_Creation_Date'])

    df['DOB'] = df['DOB'].apply(lambda x: get_dob_date(x))
    df['DOB'] = pd.to_datetime(df['DOB'])

    df['Age'] = (df['Lead_Creation_Date'] -
                 df['DOB']).astype('timedelta64[Y]').astype(int)
    df = df.drop(columns=['Lead_Creation_Date', 'DOB'])
    return df


def create_credit_features(df):
    df = df.copy()
    # запрошеннная сумма кредита за период
    df['Loan_Amount_per_Period'] = df['Loan_Amount'] / df['Loan_Period']
    # всего процентов по кредиту
    df['Credit_pct'] = df['Loan_Amount'] * df['Interest_Rate'] / 100
    # проценты по кредиту за период
    df['Credit_pct_per_Period'] = df['Credit_pct'] / df['Loan_Period']
    # отношение суммы кредита к сумме процентов
    df['Amount_over_pct'] = df['Loan_Amount'] / df['Credit_pct']
    # cумма кредита + сумма процентов
    df['Amount_plus_pct'] = (df['Loan_Amount'] + df['Credit_pct'])
    # cумма кредита + сумма процентов за период
    df['Amount_plus_pct_per_period'] = (df['Amount_plus_pct'] /
                                         df['Loan_Period'])
    # сумма кредита с процентами / сумма кредита
    df['Amount_plus_pct_over_amount'] = (df['Amount_plus_pct'] /
                                          df['Loan_Amount'])
    # сумма кредита с процентами / сумма кредита за период
    df['Amount_pct_per_period'] = (df['Amount_plus_pct_over_amount'] /
                                   df['Loan_Period'])
    # запрошенная сумма кредита за период / месячный доход
    df['Credit_over_income'] = (df['Loan_Amount_per_Period'] /
                                df['Monthly_Income'])
    # проценты по кредиту / месячный доход
    df['Credit_pct_over_income'] = (df['Credit_pct_per_Period'] /
                                    df['Monthly_Income'])

    return df


def create_groupby_features(df):
    df = df.copy()
    # месячный доход в городе
    df['Monthly_Income_in_city'] = (
        df.groupby('City_Code')['Monthly_Income'].transform('mean'))
    # месячный доход у работодателя
    df['Monthly_Income_at_employer'] = (
        df.groupby('Employer_Code')['Monthly_Income'].transform('mean'))
    # месячный доход в категории источника кредита
    df['Monthly_Income_in_source_category'] = (
        df.groupby('Source_Category')['Monthly_Income'].transform('mean'))
    # месячный доход конкретного человека / месячный доход в городе, в котором он живет
    df['Monthly_Income_over_city_income'] = (df['Monthly_Income'] /
                                             df['Monthly_Income_in_city'])
    # месячный доход конкретного человека / месячный доход у его работодателя
    df['Monthly_Income_over_employeer_income'] = (
        df['Monthly_Income'] / df['Monthly_Income_at_employer'])
    # месячный доход конкретного человека / месячный доход в категории источника кредита
    df['Monthly_Income_over_category_income'] = (
        df['Monthly_Income'] / df['Monthly_Income_in_source_category'])

    return df