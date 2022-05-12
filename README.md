# data_storage_predictive_analytics_app
predictive analytics app for data storage systems (UDS) 


Authors: Grigoriy Sokolov, Michail Zaytcev, Andrey Gantimurov
BAUM Ai team 2021

System description: the module of predictive analytics will be able to warn in advance, for a time interval defined by the user, about exceeding the set thresholds of parameters and potential failure of elements of the system, which will allow to take necessary measures to stabilize the equipment and replace the failing parts in advance. 


virtualenv -p <path to python>/bin/python3
env_name

$ pip3 install apache-airflow

$ export AIRFLOW_HOME=~/airflow

$ airflow initdb (airflow db init)

$ airflow webserver -p 8080

$ airflow scheduler

System installation:
The module requires Python version 3.6+
Installation of additional libraries to run the system:

$ pip install numpy
$ pip install pandas
$ pip install psycopg2-binary
$ pip install tensorflow (version 2.0+)
$ pip install apache-airflow
$ pip install plotly

Third-party libraries used:
Numpy - library for working with numeric and matrix data
Pandas - library for work with structured data
Psycopg2 - library for connecting to the database
Tensorflow - library for building deep learning models
ApacheAirflow - a library for implementing a pipeline and setting up the schedule of its execution
Plotly - library for interactive visualization

Connection to the virtual machine via ssh:
ssh ... , password ...

Setting up Apache Airflow:
Move the shd_predict.py module to the airflow/dags folder 
Airflow db init - initialize local database

$ airflow users create \
          --username admin \
          --firstname FIRST_NAME \
          --lastname LAST_NAME
          --role Admin \
          --E-mail admin@example.org - user creation
$ airflow webserver --port 8080 - startup web server
$ airflow scheduler (-D) - launches the scheduler (runs in the background)
$ airflow dags list - checks for dags
$ airflow dags unpause shd_predict - start dag


How the system works:
The get_data.py file is run to load the available historical data:
the system uses the psycopg2 package to connect to the Postgres database
sends an SQL query to unload the required data from the database (in this case - the information on filling the disks in time)
the information is downloaded and saved to the local disk


The system is cyclically managed through Apache Airflow. Cyclicity is set by the user. Each cycle of the system consists of the following steps:
the system connects to the Postgres database using the psycopg2 package
sends an SQL query to unload the required data from the database (in this particular case - the information about the disks' occupancy in time)
the information is uploaded to a local disc, where it is processed and analyzed
if critical values are detected, the system sends a notification to Telegram using the chatbot
the new and existing data is glued together
based on all available data, an artificial intelligence model is trained/trained to work with time-series and generate a forecast
if the forecast values exceed the admissible values, the system sends a notification via chatbot to Telegram
the system rewrites the AI model
the system builds interactive visualizations of current and predicted indicators

AI model information:
The model is based on a neural network consisting of a combination of convolutional, recurrent, and fully connected layers. The system inputs a time series with one or more parameters, the output is a forecast of these parameters for a certain number of time steps ahead. The system is able to make a forecast for one or more steps ahead, however a longer forecast will be less accurate. 

The chatbot was configured as shown in https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e
