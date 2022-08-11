
##A SparkSession can be used create DataFrame, register DataFrame as tables, 
#execute SQL over tables, cache tables, and read parquet files.

#from pyspark.sql import SparkSession
#spark=SparkSession.builder.appName('pysparkDemo_UI_final_v3').getOrCreate()

"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.\
        master("spark://192.168.4.126:7077").getOrCreate()
print("spark session created")
"""

"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName('PySpark_Tutorial_fv1')\
        .getOrCreate()

"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[6]").appName('PySpark_Tutorial_finalv1')\
        .getOrCreate()
## Read The dataset
training = spark.read.csv('test1.csv',header=True,inferSchema=True)
training.show()


from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["age","Experience"],outputCol="Independent Features")
output=featureassembler.transform(training)
output.show()


output.columns
finalized_data=output.select("Independent Features","Salary")
finalized_data.show()



from pyspark.ml.regression import LinearRegression
##train test split
train_data,test_data=finalized_data.randomSplit([0.60,0.40])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Salary')
regressor=regressor.fit(train_data)


### Coefficients
regressor.coefficients
### Intercepts
regressor.intercept

### Prediction
pred_results=regressor.evaluate(test_data)
pred_results.predictions.show()


pred_results.meanAbsoluteError
pred_results.meanSquaredError





##############2222222222222222222
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('pysparkDemoV2').getOrCreate()


# The applied options are for CSV files. For other file types, these will be ignored.
df =spark.read.csv("tips.csv",header=True,inferSchema=True)
df.show()

### Handling Categorical Features
from pyspark.ml.feature import StringIndexer
df.show()


indexer=StringIndexer(inputCol="sex",outputCol="sex_indexed")
df_r=indexer.fit(df).transform(df)
df_r.show()



indexer=StringIndexer(inputCols=["smoker","day","time"],outputCols=["smoker_indexed","day_indexed",
                                                                  "time_index"])
df_r=indexer.fit(df_r).transform(df_r)
df_r.show()



from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=['tip','size','sex_indexed','smoker_indexed','day_indexed',
                          'time_index'],outputCol="Independent Features")
output=featureassembler.transform(df_r)


output.select('Independent Features').show()


finalized_data=output.select("Independent Features","total_bill")
finalized_data.show()




from pyspark.ml.regression import LinearRegression
##train test split
train_data,test_data=finalized_data.randomSplit([0.60,0.40])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='total_bill')
regressor=regressor.fit(train_data)




### Predictions
pred_results=regressor.evaluate(test_data)
## Final comparison
pred_results.predictions.show()




### PErformance Metrics
pred_results.r2
pred_results.meanAbsoluteError
pred_results.meanSquaredError



































