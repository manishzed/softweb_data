from pyspark.sql import SparkSession

appName = "Kafka Examples"
master = "local"

spark = SparkSession.builder \
    .master(master) \
    .appName(appName) \
    .getOrCreate()

kafka_servers = "localhost:9092"

df = spark \
    .read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_servers) \
    .option("subscribe", "pkttest") \
    .load()
df = df.withColumn('key_str', df['key'].cast('string').alias('key_str')).drop(
    'key').withColumn('value_str', df['value'].cast('string').alias('key_str')).drop('value')
    
df.show(50)
