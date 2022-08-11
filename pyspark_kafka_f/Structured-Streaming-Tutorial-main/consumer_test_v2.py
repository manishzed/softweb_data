import os
from kafka import KafkaProducer
from pyspark.sql import SparkSession, DataFrame
import time
from datetime import datetime, timedelta
"""
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,org.apache.spark:spark-streaming-kafka-0-8_2.12:3.30 pyspark-shell'
"""
topic_name = "pkttest"
kafka_broker = "localhost:9092"

producer = KafkaProducer(bootstrap_servers = kafka_broker)
spark = SparkSession.builder.getOrCreate()
terminate = datetime.now() + timedelta(seconds=30)

while datetime.now() < terminate:
    producer.send(topic = topic_name, value = str(datetime.now()).encode('utf-8'))
    time.sleep(1)

readDF = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("startingOffsets", "earliest") \
    .option("subscribe", topic_name) \
    .load()
    
readDF = readDF.selectExpr("CAST(key AS STRING)","CAST(value AS STRING)")

#print("1111111111111", readDF)
#readDF.writeStream.format("console").start()
#readDF.show()
query = readDF.writeStream.format("console").start()
import time
time.sleep(10) # sleep 10 seconds
query.stop()
#print("2222222222222", query)

producer.close()
