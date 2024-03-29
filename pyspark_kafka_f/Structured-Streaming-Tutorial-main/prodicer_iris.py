# -*- coding: utf-8 -*-
"""prodicer_iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Siddharth1698/Structured-Streaming-Tutorial/blob/main/prodicer_iris.ipynb
"""

import pandas as pd
from kafka import KafkaProducer
from datetime import datetime
import time
import random
import numpy as np

# pip install kafka-python

#KAFKA_TOPIC_NAME_CONS = "TutorialTopic_v2.1"
KAFKA_TOPIC_NAME_CONS = "pkttest"
KAFKA_BOOTSTRAP_SERVERS_CONS = 'localhost:9092'

if __name__ == "__main__":
    print("Kafka Producer Application Started ... ")

    kafka_producer_obj = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS_CONS,
                                       value_serializer=lambda x: x.encode('utf-8'))
    
    filepath = "IRIS.csv"
    
    flower_df = pd.read_csv(filepath)
  
    flower_df['order_id'] = np.arange(len(flower_df))

    
    flower_list = flower_df.to_dict(orient="records")
       

    message_list = []
    message = None
    for message in flower_list:
        
        message_fields_value_list = []
               
        message_fields_value_list.append(message["order_id"])
        message_fields_value_list.append(message["sepal_length"])
        message_fields_value_list.append(message["sepal_width"])
        message_fields_value_list.append(message["petal_length"])
        message_fields_value_list.append(message["petal_width"])
        message_fields_value_list.append(message["species"])

        message = ','.join(str(v) for v in message_fields_value_list)
        print("Message Type: ", type(message))
        print("Message: ", message)
        kafka_producer_obj.send(KAFKA_TOPIC_NAME_CONS, message)
        time.sleep(1)


    print("Kafka Producer Application Completed. ")


"""
import pandas as pd
df3 = pd.read_csv('IRIS.csv')
df3.head()
"""
