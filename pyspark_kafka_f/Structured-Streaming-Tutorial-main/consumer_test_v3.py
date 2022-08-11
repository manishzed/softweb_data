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



readDF = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("startingOffsets", "earliest") \
    .option("subscribe", topic_name) \
    .load()
    
readDF = readDF.selectExpr("CAST(key AS STRING)","CAST(value AS STRING)")

#readDF = readDF.selectExpr("CAST(value AS STRING)", "timestamp")
#query = summary \
query = readDF \
        .writeStream \
        .trigger(processingTime='15 seconds') \
        .outputMode("update") \
        .format("console") \
        .start()

### If you want to try storing the data frames in files.
#query = pkt_df3 \
#        .writeStream \
#        .trigger(processingTime='5 seconds') \
#        .outputMode("append") \
#        .format("csv") \
#        .option("checkpointLocation", "/home/raja/kafka/pkt-example") \
#        .option("path", "/home/raja/kafka/pkt-example") \
#        .start()


query.awaitTermination()
"""
#print("1111111111111", readDF)
#readDF.writeStream.format("console").start()
#readDF.show()
query = readDF.writeStream.format("console").start()
import time
time.sleep(10) # sleep 10 seconds
query.stop()
#print("2222222222222", query)

producer.close()
"""
