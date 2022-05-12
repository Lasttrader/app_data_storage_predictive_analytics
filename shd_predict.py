import os
import pandas as pd
import psycopg2
import numpy as np
import requests	
import re

from datetime import timedelta, datetime, timezone, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import tensorflow as tf
import tensorflow.keras as keras

FOLDER = 'data_container/' # локальный путь для хранения загруженных данных
#FOLDER = '/root/model/data_container/' # локальный путь для хранения загруженных данных
START_TIME = '30-03-21 14:00'
CURRENT_TIME_NUM = int(datetime.now(timezone.utc).timestamp()) # convert to UNIX time
TIMEPERIOD_NUM = 600*1 # hours between queries
RESAMPLE_STEP = '60min' 
STEPS_BEFORE = 24 # временные шаги для обучения
STEPS_AHEAD = 2 # временные шаги для предсказания

SNMP_DICT = {
#'pool_size': '.1.3.6.1.4.1.%'
#'pool_free_space': '.1.3.6.1.4.1.%',
'vol_size_total': '.1.3.6.1.4.1.%',
'vol_size_used': '.1.3.6.1.4.1.%'
}

######################################################################################## 
####### ----------------- Функции для загрузки данных ------------------- ############
#########################################################################################

def get_data(folder, current_time, timeperiod, snmp_dict=SNMP_DICT, check_income_data=True):
    """Функция для загрузки данных из БД и 
    их проверка на превышение критических значений"""
    con = psycopg2.connect(host = '___',
                           dbname = '___',
                           user = '___',
                           password ='____',
                           port = '___') # подключение к сторонней базе данных
    cursor=con.cursor()

    for name, snmp in snmp_dict.items(): # итерируемся по всем snmp
        with open(folder+'download_log.txt', 'a') as f: 
            f.write(str(datetime.now(timezone.utc)) + ' downloading {}...\n'.format(name)) # делаем запись в лог о загрузке
        filename = name+'.csv'
        cursor.execute(
            """
            SELECT history_uint.itemid, history_uint.clock, history_uint.value, items.name FROM history_uint
            JOIN items ON (history_uint.itemid = items.itemid)
            AND items.hostid = 10281
            AND items.snmp_oid LIKE '{}'
            AND history_uint.clock BETWEEN {} AND {}
            """.format(snmp, current_time-timeperiod, current_time)) # sql запрос для извлечения данных
        data = cursor.fetchall() # выполнение запроса
        data = pd.DataFrame(data, columns = ['itemid', 'clock', 'value', 'name']) # формирование датасета
        data['clock'] = pd.to_datetime(data['clock'], unit='s') # приведение колонки к временному формату
        data.index = data['clock']
        data = data[['value', 'name']] # оставляем только две колонки
        data.sort_index(ascending=False)

        if name == 'vol_size_total': # если загружаем полный объем, то сохраняем в переменную, для вычисления занятой доли
            vol_total = data
        elif name == 'vol_size_used': # если загружаем текущий объем
            data_new = pd.DataFrame()
            for name in set(data['name']): # для каждого диска
                v_max = vol_total.loc[vol_total['name'] == 'Volsize by volume: '+ name.split(': ')[1],'value'].astype(
                    float).resample(RESAMPLE_STEP).mean().fillna(method='ffill') # делаем ресампл и извлекаем полный объем диска
                v_cur = data.loc[data['name']==name,'value'].astype(
                    float).resample(RESAMPLE_STEP).mean().fillna(method='ffill') # делаем ресампл и извлекаем текущее заполнение диска
                data_new = data_new.append(pd.DataFrame({
                    'value': v_cur.values/v_max.values, 
                    'name': name}, index = v_cur.index)) # определяем заполненную долю каждого диска в текущий момент
            data = data_new
            data['value'] = np.round(data['value'].astype(float), 4) # процент от максимального

            if os.path.exists(folder+filename): # запись в файл, если не существует - создаем
                data.to_csv(folder+filename, mode='a', header=False)
            else:
                data.to_csv(folder+filename, mode='w')
        
            if check_income_data: # проверка данных на превыщение критических значений
                check_data(data)
    
    
def check_data(data):
    """Функция для проверки данных на превышение критических значений"""
    for name in set(data['name']): # для каждого диска
        subset = data[data['name']==name] # подвыборка для диска
        if any(subset['value']>=0.8): # если где-то есть превышение
            idx = np.where(subset['value']>=0.8) # находим индекс
            send_notification('Истиные значения близки к критическим по параматру {} со значением \n {} в \n{}!'.format(
                name, 
                subset.iloc[idx]['value'].values, 
                subset.index[idx].astype(str).values)) # высылаем уведомление в телеграм

def load_data(file_name, folder=FOLDER, check_values=True, resample_step=RESAMPLE_STEP, notify=True):
    """Функция для загрузки данных с локального диска"""
    data = pd.read_csv(folder+file_name+'.csv')
    data['clock'] = pd.to_datetime(data['clock'])     # change index type to datetime
    return data


######################################################################################## 
####### ----------------- Функции связанные с моделью ------------------- ############
#########################################################################################

def seq2seq_window_dataset(series, steps_before, steps_ahead, batch_size=32, shuffle_buffer=1000):
    """Функция для создания временных окон для обучения нейронной сети"""
    window_size = steps_before + steps_ahead # создание окна
    series = tf.expand_dims(series, axis=-1) # расширяем размерность
    ds = tf.data.Dataset.from_tensor_slices(series) # формируем TF датасет
    ds = ds.window(window_size, shift=steps_ahead, drop_remainder=True) # создаем временные окна
    ds = ds.flat_map(lambda w: w.batch(window_size)) # делаем батчи временных окон
    ds = ds.shuffle(shuffle_buffer).batch(batch_size) # перемешиваем окна
    ds = ds.map(lambda window: (window[:, :-steps_ahead], window[:, steps_ahead:])) # делим каждое окно на Х и У
    return ds.prefetch(1)


def make_train_val_sets(series, ratio, steps_before, steps_ahead, batch_size=64):
    """Функция для создания тренировочной и валидационной подвыборок"""
    split_idx = len(series) # разделим на train и validation
    x_train = series[:int(ratio*split_idx)] # для тренировки модели
    x_valid = series[int(ratio*split_idx):] # оставим для проверки модели
    train_set = seq2seq_window_dataset(x_train, steps_before, steps_ahead, batch_size=batch_size) # формируем тренировочные батчи
    valid_set = seq2seq_window_dataset(x_valid, steps_before, steps_ahead, batch_size=batch_size) # формируем валидационные батчи
    return train_set, valid_set


def create_model(model_type=None):
    """Функция для создания архитектуры нейронной сети"""
    input_ = keras.Input(shape=[None, 1])
    x = keras.layers.Conv1D(filters=32, kernel_size=5,
              strides=1, padding="causal",
              activation="relu")(input_)
    x = keras.layers.LSTM(128, recurrent_dropout=0.3, return_sequences=True)(x)
    x = keras.layers.LSTM(128, recurrent_dropout=0.2, return_sequences=True)(x)
    output_ = keras.layers.Dense(1)(x)
    model = keras.Model(input_, output_, name='conv_lstm')
    return model


def train_model(file_names, steps_before, steps_ahead, epochs=500, 
                    folder=FOLDER, resample_step=RESAMPLE_STEP):
    """Функиця для тренировки/перетренировки нейронной сети"""
    for file_name in file_names:
        data = load_data(file_name, check_values=False) # загрузка датасета
        data['clock'] = pd.to_datetime(data['clock'])

        for name in sorted(set(data['name'])): # для каждого диска
            subset = data[data['name']==name].resample(resample_step, on='clock').mean().fillna(method='ffill') # собираем подвыборку
            series = subset['value'].values
            model_name = file_name + '_' + re.sub('/', '_', name.split(':')[1]) + '.h5' # определяем имя модели
            
            train_set, valid_set = make_train_val_sets(series, 0.8, steps_before, steps_ahead) # формируем тренировочные и валидационную выборки
            model_checkpoint = keras.callbacks.ModelCheckpoint(folder+model_name, save_best_only=True) # сохраняем лучшую модель
            early_stopping = keras.callbacks.EarlyStopping(patience=10) # раннее прекращение

            if os.path.exists(folder+model_name): # если модель существует, загружаем ее и перетренировываем
                model = keras.models.load_model(folder+model_name) # подгружаем модель
            else: # если модель не существует, создаем ее
                model = create_model() # создание модели с определенной архитектурой
                model.compile(loss=keras.losses.MeanSquaredError(),
                              optimizer=keras.optimizers.Adam(),
                              metrics=["mae"])  # определение параметров нейросети
            history = model.fit(train_set, epochs=epochs,
                                validation_data=valid_set,
                                callbacks=[early_stopping, model_checkpoint], verbose=1) # сохранение истории тренировки
            best_n = np.argmin(history.history['mae']) # определение лучшей модели

            with open(folder+'model_accuracy.txt', 'a') as f:
                f.write('{} - {} results: \n {} \n'.format(
                    str(datetime.now(timezone.utc)), 
                    model_name, 
                    [{item[0]:np.round(item[1][best_n], 4)} for item in history.history.items()]
                    )
                ) # запись качества модели в лог


def model_forecast(model, series, steps_before, steps_ahead):
    """Функция для создания временных окон для предсказаний нейросети"""
    window_size = steps_before  #  размер окна
    series = tf.expand_dims(series, axis=-1) # расширяем размерность
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=steps_ahead, drop_remainder=True)  #  создаем временные окна
    ds = ds.flat_map(lambda w: w.batch(window_size)) # создаем батчи
    ds = ds.batch(1).prefetch(1)
    forecast = model.predict(ds) # генерируем предсказание
    return forecast

def make_prediction(series, model_name, param_name, folder=FOLDER, resample_step=RESAMPLE_STEP, 
                    steps_before=STEPS_BEFORE, steps_ahead=STEPS_AHEAD):
    """Функция для предсказаний нейросети """
    model = keras.models.load_model(folder+model_name)  # загрузка натренированной модели
    pred = np.array(model_forecast(model, series, steps_before, steps_ahead)) # генерация предсказаний
    prediction = np.array([i[-steps_ahead:].ravel() for i in pred]).ravel() # выравнивание предсказаний
    time_step = int(resample_step.split('min')[0])
    prediction_idx = np.array([series.index.min() + i*timedelta(minutes=time_step) 
                        for i in range(len(series)+steps_ahead)]) # создание временного шага предсказаний
    last_idx_out = (len(series)-steps_before)%steps_ahead # в зависимости от размера окна, несколько конечных эл-ов могут остаться вне окна, поэтому 
    if last_idx_out == 0:
        prediction_idx = prediction_idx[steps_before:]
    else:
        prediction_idx = prediction_idx[steps_before:-last_idx_out]

    if np.any(prediction[-steps_ahead:] >= 0.8):  # проверка предсказаний на превышение критических значений
        idx = np.where(prediction[-steps_ahead:]>=0.8)[0]
        send_notification('Прогнозируемые значения близки к критическим по параматру {} со значением {} в {}!'.format(
            param_name, 
            np.round(prediction[-steps_ahead:][idx], 3), 
            prediction_idx[-steps_ahead:][idx].astype(str)))
    return prediction_idx, prediction

######################################################################################## 
####### ----------------- Функции для визуализации ------------------- ############
#########################################################################################

def predict_and_plot(file_name, active_plot, folder=FOLDER, resample_step=RESAMPLE_STEP, steps_ahead=STEPS_AHEAD):
    """Функция для визуализации текущих значений и предсказаний"""
    # подгрузка данных и препроцессинг
    data = load_data(file_name, notify=False)
    data['clock'] = pd.to_datetime(data['clock'])
    # Настройки графика
    x_min = data['clock'].min() - timedelta(hours=12) # настройка осей
    x_max = data['clock'].max() + timedelta(hours=12) # настройка осей
    fig = make_subplots(rows=len(set(data['name'])), cols=1)
    pallete = px.colors.qualitative.Plotly + ['black']  # палитра графика
    for i, name in enumerate(sorted(set(data['name']))): # для каждого диска
        subset = data[data['name']==name].resample(resample_step, on='clock')['value'].mean().fillna(method='ffill') # создаем подвыборку
        model_name = file_name + '_' + re.sub('/', '_', name.split(':')[1]) + '.h5' # имя модели
        prediction_index, prediction = make_prediction(subset, model_name, param_name=name) # делаем предсказание
        if active_plot:  # для отрисовки графика
            # для отрисовки допустимого диапазона
            fig.add_trace(go.Scatter(
                x=[x_min, x_min, x_max, x_max], 
                y=[0.0, 0.8, 0.8, 0.0], 
                mode='lines', fill='toself', fillcolor='rgba(0,100,80,0.1)', line=dict(width=0), showlegend=False), 
                row=i+1, col=1)
            # для отрисовки недопустимого диапазона
            fig.add_trace(go.Scatter(
                x=[x_min, x_min, x_max, x_max], 
                y=[0.8, 1.0, 1.0, 0.8], 
                mode='lines', fill="toself", 
                fillcolor='rgba(165,0,38,0.1)', line=dict(width=0), showlegend=False), 
                row=i+1, col=1)
            # для отрисовки текущих значений
            fig.add_trace(go.Scatter(
                x = subset.index, 
                y = subset.values, 
                name = name.split(':')[1], line=dict(width=3, color=pallete[i]), mode='lines'), 
                row=i+1, col=1)
            #для отрисовки прогнозных значений
            fig.add_trace(go.Scatter(
                x = prediction_index, 
                y = prediction.ravel(), 
                name = 'prediction', line=dict(width=3, color=pallete[i], dash='dot'), mode='lines'), 
                row=i+1, col=1)
        else:
            pass

        with open(folder+'prediction.txt', 'a') as f:
            f.write('{}_{},{},{}\n'.format(
                str(datetime.now(timezone.utc)), 
                name,
                str(prediction_index[-1]),
                str(prediction[-1])),
            ) # запись предсказаний в лог
        with open(folder+'real_values.txt', 'a') as f:
            f.write('{}_{},{},{}\n'.format(
                str(datetime.now(timezone.utc)), 
                name,
                str(subset.index[-1]),
                str(subset.values[-1])),
            ) # запись текущих значений в лог
     # общие настройки графика
    fig.update_layout(
        height=(i+1)*250,
        width=1300,
        title=file_name,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend = dict(borderwidth=1))
    fig.update_xaxes(range=[x_min, x_max], title_text='Date', row=i+1)
    fig.update_yaxes(range=[-0.00, 1.01])
    return fig


def visualize(file_names, active_plot=True):
    """Функия, определяющая создавать ли визуализаци"""
    p = []
    for file_name in file_names: # для каждого параметра
        predict_and_plot(file_name, active_plot=active_plot).show(renderer='browser')


########################################################################################
########### ------------------ Функции для уведомлений ------------------- ############
########################################################################################

def save_to_file(text_log, file_name, folder=FOLDER):
    """Функция для сохранения в файл"""
    with open(folder+file_name, 'a+', encoding='utf8') as f:
        f.write('{} {}\n'.format(str(datetime.now()), str(text_log)))


def send_notification(bot_message):
    """Функция для отправки уведомлений в Телеграм"""
    bot_token = '1577094629:AAHLZCWvvFxS1TV0h_1-cPP195psJBmyWPY'
    bot_chatID = '-555640103'
	#bot_chatID = '256804461'
    send_text = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
        bot_token, bot_chatID, bot_message)
    response = requests.get(send_text)
    save_to_file(bot_message, 'warning_log.txt')  # сохранение сообщения в лог


########################################################################################
########### ------------------ Организация пайплайна ------------------- ############
########################################################################################

# Дефолтные параметры для ДАГов
default_args = {'owner': 'airflow',
				'start_date': datetime.strptime(START_TIME, '%d-%m-%y %H:%M'),
				}

# Объединение функций в один пайплайн в ДАГе

with DAG('shd_predict', 
	default_args=default_args, schedule_interval=timedelta(minutes=int(TIMEPERIOD_NUM/60)), catchup=False) as dag:
# Определяем питоновский оператор для airflow и передаем параметры в функцию
	get_data = PythonOperator(
		task_id='get_data_from_DB', 
		python_callable=get_data,
		op_kwargs = {'folder': FOLDER, 'current_time': CURRENT_TIME_NUM, 'timeperiod': TIMEPERIOD_NUM})
# Определяем питоновский оператор для airflow и передаем параметры в функцию
	train_model = PythonOperator(
		task_id='train_model', 
		python_callable=train_model,
		op_kwargs = {
		'file_names': ['vol_size_used'], 
		'steps_before': STEPS_BEFORE, 'steps_ahead': STEPS_AHEAD})
# Определяем питоновский оператор для airflow и передаем параметры в функцию
	visualize = PythonOperator(
		task_id='visuzalize_data', 
		python_callable=visualize,
		op_kwargs = {'file_names': ['vol_size_used'],
					'active_plot': True})

	get_data >> train_model >> visualize  # определение последовательности выполнения функций