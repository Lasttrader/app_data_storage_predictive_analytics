import os
import pandas as pd
import psycopg2
import numpy as np
import requests	
import re
from datetime import timedelta, datetime, timezone

FOLDER = 'data_container/' # локальный путь для хранения загруженных данных
CURRENT_TIME_NUM = int((datetime.now(timezone.utc)).timestamp()) # convert to UNIX time
TIMEPERIOD_NUM = 3600*180
RESAMPLE_STEP = '5min'

SNMP_DICT = {
#'pool_size': '.1.3.6.1.4.1.%',
#'pool_free_space': '.1.3.6.1.4.1.%',
 'vol_size_total': '.1.3.6.1.4.1.%',
 'vol_size_used': '.1.3.6.1.4.1.%',
#'disk_read_io': '.1.3.6.1.4.1.%'
}

def get_data(folder, current_time, timeperiod, resample_step = RESAMPLE_STEP, snmp_dict=SNMP_DICT):
	con = psycopg2.connect(host = '__',
	                       dbname = '___',
	                       user = '___',
	                       password ='___',
	                       port = '___') # подключение к сторонней базе данных
	cursor=con.cursor()
	
	for name, snmp in snmp_dict.items(): # итерируемся по всем snmp
		with open('download_log.txt', 'a') as f:
			f.write(str(datetime.now(timezone.utc)) + ' downloading {}...\n'.format(name)) # делаем запись в лог о загрузке
		print(datetime.now(timezone.utc), 'downloading {}...'.format(name))
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
		data = data[['value', 'name']]
		data.sort_index(ascending=False)

		if name == 'vol_size_total': # если загружаем полный объем, то сохраняем в переменную, для вычисления занятой доли
			vol_total = data
		elif name == 'vol_size_used': # если загружаем текущий объем
			data_new = pd.DataFrame()
			for name in set(data['name']): # для каждого диска
				v_max = vol_total.loc[vol_total['name'] == 'Volsize by volume: '+ name.split(': ')[1],'value'].astype(
					float).resample(resample_step).mean().fillna(method='ffill') # делаем ресампл и извлекаем полный объем диска
				v_cur = data.loc[data['name']==name,'value'].astype(
					float).resample(resample_step).mean().fillna(method='ffill') # делаем ресампл и извлекаем текущее заполнение диска
				data_new = data_new.append(pd.DataFrame({
					'value': v_cur.values/v_max.values, 
					'name': name}, index = v_cur.index)) # определяем заполненную долю каждого диска в текущий момент
			data = data_new
			data['value'] = np.round(data['value'].astype(float), 3)

		if os.path.exists(folder+filename): # запись в файл, если не существует - создаем
			data.to_csv(folder+filename, mode='a', header=False)
		else:
			data.to_csv(folder+filename, mode='w')



get_data(FOLDER, CURRENT_TIME_NUM, TIMEPERIOD_NUM)
