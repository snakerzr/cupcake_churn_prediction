# imports

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool, metrics, cv

from pickle import load

"""
На вход поступают данные со столбцами:
clnt_ID - str
timestamp - datetime
gest_Sum - float
gest_Discount - float

Это данные по сделкам, каждая строка - 1 сделка (чек)

Для предсказания нужно:
1. Удалить дубликаты (опционально уведомить о наличии дубликатов)
2. Удалить записи, где сумма и скидка == 0 (опционально уведомить)
3. Создать столбцы `'days_delta'` и `'clnt_buys_count'`.  
4. Сгруппировать по айдишникам и по столбцам посчитать агрегирующие функции.  
5. Всем клиентам с только 1 покупкой поставить таргет 0 или вероятность 0.  
6. Сделать предсказание по остальным клиентам

Предсказывать будем по сгруппированным по клиентам данным со следующими агрегирующими фукнциями:

* gest_Sum
    * min
    * max
    * median
    * mean
* gest_Discount
    * min
    * max
    * median
    * mean
* days_delta
    * mean
* clnt_buys_count
    * last
    
***
***
***


dups_deletion() - удаление дубликатов
zeros_deletion() - удаление чеков, где сумма и скидка == 0
prep_add_features() - добавление фичей days_delta, clnt_buys_count
prep_groupby_clnt_ID() -  группировка по clnt_ID
only_1_buy_split() - разделение датафрейма на группу, в которой больше 1 покупки и else
predict() - предсказание моделью тарегта (предназначена только для группы с покупками > 1)
predict_proba() -  предсказание моделью вероятности отнесения к таргету 1 (предназначена только для группы с покупками > 1)
fill_only_1_buys_target() - заполнение 0 таргета (предназначена только для группы с покупками = 1)
fill_only_1_buys_proba() - заполнение вероятности отнесения к таргету 1 нулями ((предназначена только для группы с покупками = 1)
create_inference_file() - создание файла с предсказаниями (clnt_ID, predict_proba)
full_inference() - функция, включающая полный цикл
"""

def dups_deletion(df):
    '''
    Удаление дубликатов.
    '''
    if df.duplicated().sum() > 0:
        print('df.duplicated().sum() > 0 == True')
        df = df.drop_duplicates()
    return df

def zeros_deletion(df):
    """
    Удаление сделок, где сумма и скидка == 0
    """
    if df.loc[(df['gest_Sum'] == 0) & (df['gest_Discount'] == 0),'clnt_ID'].count() > 0:
        print("df.loc[(df['gest_Sum'] == 0) & (df['gest_Discount'] == 0)].count() > 0 == True")
        df = df.drop(index=df.loc[(df['gest_Sum'] == 0) & (df['gest_Discount'] == 0)].index)
    return df

def prep_add_features(df):
    """
    Ф-ция создает новые фичи, необходимые для группировки и предикта
    """
    
    def time_delta(col):
        return col.diff()

    df['days_delta'] = df.groupby('clnt_ID')['timestamp'].transform(time_delta)
    df['days_delta'] = df['days_delta'].dt.days
    df['days_delta'] = df['days_delta'].fillna(-1) # Это делаем, чтобы значение "вообще не было покупок ранее" отличалось от "прошло 0 дней с прошлой покупки"
    
    
    def expand_count(col):
        return col.expanding().count()

    df['clnt_buys_count'] = df.groupby('clnt_ID')['timestamp'].transform(expand_count)
    
    return df

def prep_groupby_clnt_ID(df):
    """
    Группировка по айди клиента с агрегирующими функциями по столбцам
    """
    aggfunc_dict = {
                'gest_Sum': ['min','max','median','mean',], 
                'gest_Discount': ['min','max','median','mean'], 
                'days_delta': ['mean'],
                'clnt_buys_count': ['last']

               } 
    df = df.pivot_table(index='clnt_ID', 
                    aggfunc=aggfunc_dict, 
                    )
    df.columns = ['_'.join(column) for column in df.columns]
    return df

def only_1_buy_split(df):
    """
    Разделение датасета на клиентов с кол-вом покупок больше 1 и равным 1
    """
    print('1 buy only ==',df.loc[df['clnt_buys_count_last'] == 1,'clnt_buys_count_last'].count())
    df_1_buy_only = df.loc[df['clnt_buys_count_last'] == 1]
    df = df.drop(index=df[df['clnt_buys_count_last'] == 1].index)
    return df, df_1_buy_only

def predict(df):
    """
    Предсказание класса моделью
    """
    with open('./models/model.pcl', 'rb') as file:
        model = load(file)
        
    return model.predict(df)

def predict_proba(df):
    """
    Предсказание вероятности отнесения к классу 1
    """
    with open('./models/model.pcl', 'rb') as file:
        model = load(file)
        
    return model.predict_proba(df)[:,1]

def fill_only_1_buys_target(df):
    """
    Ф-ция выставления класса 0 для всех пользователей, у кого только 1 покупка
    """
    df['target'] = 0
    return df

def fill_only_1_buys_proba(df):
    """
    Ф-ция выставления вероятности класса 1 равной 0 для всех пользователей, у кого только 1 покупка
    """    
    df['proba'] = 0
    return df

def create_inference_file(df,df_1_buy):
    """
    Создание файла с предсказаниями
    """
    result = pd.concat([df,df_1_buy])
    result['proba'].to_csv('inference.csv')

def full_inference(df:pd.DataFrame):
    df = dups_deletion(df)
    df = zeros_deletion(df)
    df = prep_add_features(df)
    df = prep_groupby_clnt_ID(df)
    df, df_only_1 = only_1_buy_split(df)
    
    df['target'] = predict(df)
    df['proba'] = predict_proba(df)
    
    df_only_1 = fill_only_1_buys_target(df_only_1)
    df_only_1 = fill_only_1_buys_proba(df_only_1)
    
    create_inference_file(df,df_only_1)

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', parse_dates=[1])
    full_inference(df)