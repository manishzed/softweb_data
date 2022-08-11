import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.context import SparkContext
from kafka import KafkaConsumer


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

consumer = KafkaConsumer('pkttest')

for message in consumer:
    data = message
    print(data) # Printing the messages properly
    #df = data.map # am unable to convert it to a dataframe.
    print("1111", data.value)





