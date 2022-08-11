from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

from pyspark.sql.types import StructType,StructField,LongType,IntegerType,FloatType,StringType, DoubleType
from pyspark.sql.functions import split,from_json,col


odometrySchema = "order_id INT,sepal_length DOUBLE,sepal_length DOUBLE,sepal_length DOUBLE,sepal_length DOUBLE,species STRING"


            
spark = SparkSession \
    .builder \
    .appName("SSKafka") \
    .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# try this "startingOffsets" as either "earliest" or "latest"
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "pkttest") \
  .option("delimeter",",") \
  .option("startingOffsets", "latest") \
  .load() 

df.printSchema()

df1 = df.selectExpr("CAST(value AS STRING)").select(from_csv(col("value"),odometrySchema).alias("data")).select("data.*")
df1.printSchema()

df1.writeStream.trigger(processingTime='15 seconds') \
  .outputMode("update") \
  .format("console") \
  .option("truncate", False) \
  .start() \
  .awaitTermination()
  
  
  
  
 
### If you want to try storing the data frames in files.
#query = pkt_df3 \
#        .writeStream \
#        .trigger(processingTime='5 seconds') \
#        .outputMode("append") \
#        .format("csv") \
#        .option("checkpointLocation", "/home/raja/kafka/pkt-example") \
#        .option("path", "/home/raja/kafka/pkt-example") \
#        .start()

