import configparser
from datetime import datetime
import os
from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, count
from pyspark.sql.functions import year, month, dayofmonth, hour, \
                                  weekofyear, date_format
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DateType, TimestampType, \
                              StructField, StringType, LongType, \
			      DoubleType, StructType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Configure and create a (or get an existing) Spark Session  

    Arguments: None
    Return: Spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Load Song dataset from AWS S3 into spark and build two dimension 
    tables (songs, artists)
    And writes tables back to S3 in parquet format

    Arguments:
        spark: Spark Session
        input_data: Input data path
        output_data: Output data path
    Return: None  
    """
    # build SQLContext
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.format("json").load(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration")

    # make sure songs table has desired schema
    songs_schema = StructType([ 
                   StructField('song_id',   StringType(), False), 
                   StructField('title',     StringType(), False), 
                   StructField('artist_id', StringType(), False),  
                   StructField('year',      LongType(),   False), 
                   StructField('duration',  DoubleType(), False) 
               ])
    songs_table = sqlContext.createDataFrame(songs_table.rdd, songs_schema)
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.repartition(col('year'), col('artist_id')) \
               .write.partitionBy('year', 'artist_id') \
               .parquet(os.path.join(output_data, 'songs'), mode='overwrite')

    # extract columns to create artists table
    artists_table = df.selectExpr("artist_id", "artist_name as name", 
                                  "artist_location as location",
                                  "artist_latitude as latitude", 
				  "artist_longitude as longitude")
    
    # make sure artists table has desired schema
    artists_schema = StructType([ 
                     StructField('artist_id', StringType(), False), 
                     StructField('name',      StringType(), False), 
                     StructField('location',  StringType(), True),  
                     StructField('latitude',  DoubleType(), True), 
                     StructField('longitude', DoubleType(), True) 
                 ])
    artists_table = sqlContext.createDataFrame(artists_table.rdd, artists_schema)
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'),
                                mode='overwrite')


def process_log_data(spark, input_data, output_data):
    """
    Load Log & Song dataset from AWS S3 into spark and build two dimension 
    tables (users, time) and a fact table (songplays) 
    And writes tables back to S3 in parquet format

    Arguments:
        spark: Spark Session
        input_data: Input data path
        output_data: Output data path
    Return: None  
    """
    # build SQLContext
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    
    # get filepath to log data file
    log_data = os.path.join(input_data, "log_data/*/*/*.json")
    
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")

    # read log data file
    df = spark.read.format("json").load(log_data)
    
    # filter by actions for song plays and keep only 'NextSong' event
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    # users table need a special treat for keeping the last level value 
    # in the log data
    # For capturing the last level value for each user, Window will be used
    win1 = Window.partitionBy('userId').orderBy('ts') \
                 .rangeBetween(Window.unboundedPreceding, 0)
    df_users = df.filter(col('userId').isNotNull()). \
                 select('userId', 'firstName', 'lastName', 
		        'gender', 'level', 'ts') \
                 .withColumn('tsSeq', count('ts').over(win1))
    win2 = Window.partitionBy("userId")
    df_users = df_users.withColumn("tsSeqMax", F.max("tsSeq").over(win2))
    users_table =  df_users.filter((df_users.tsSeq == df_users.tsSeqMax)) \
                           .groupBy('userId') \
                           .agg(F.max("firstName"), F.max("lastName"),
			        F.max("gender"), F.max("level")) \
                           .withColumnRenamed('userId', 'user_id') \
                           .withColumnRenamed('max(firstName)', 'first_name') \
                           .withColumnRenamed('max(lastName)', 'last_name') \
                           .withColumnRenamed('max(gender)', 'gender') \
                           .withColumnRenamed('max(level)', 'level') 

    # make sure users table has desired schema
    users_schema = StructType([ 
                       StructField('user_id',    StringType(), False), 
                       StructField('first_name', StringType(), False), 
                       StructField('last_name',  StringType(), False),  
                       StructField('gender',     StringType(), True), 
                       StructField('level',      StringType(), False) 
                   ])
    users_table = sqlContext.createDataFrame(users_table.rdd, users_schema)
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'),
                              mode='overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1e3), TimestampType())
    df =  df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1e3), DateType())
    df = df.withColumn("ts_date", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.groupBy('timestamp') \
                .agg({'timestamp':'max'}) \
                .withColumn('hour', F.hour('max(timestamp)')) \
                .withColumn('day', F.dayofmonth('max(timestamp)')) \
                .withColumn('week', F.weekofyear('max(timestamp)')) \
                .withColumn('month', F.month('max(timestamp)')) \
                .withColumn('year', F.year('max(timestamp)')) \
                .withColumn('weekday', F.date_format('max(timestamp)', 'E')) \
                .withColumnRenamed('max(timestamp)', 'start_time') \
                .drop('timestamp') \
                .orderBy(F.asc('start_time'))

    # make sure time table has desired schema
    time_schema = StructType([ 
                      StructField('start_time', TimestampType(), False), 
                      StructField('hour',       IntegerType(),   False), 
                      StructField('day',        IntegerType(),   False),  
                      StructField('week',       IntegerType(),   False), 
                      StructField('month',      IntegerType(),   False), 
                      StructField('year',       IntegerType(),   False), 
                      StructField('weekday',    StringType(),   False)
                  ])
    time_table = sqlContext.createDataFrame(time_table.rdd, time_schema)
    
    # write time table to parquet files partitioned by year and month
    time_table.repartition(col('year'), col('month')) \
              .write.partitionBy('year', 'month') \
              .parquet(os.path.join(output_data, 'time'), mode='overwrite')

    # read in song data to use for songplays table
    song_df = spark.read.format("json").load(song_data)
        
    t_log = df.alias('t_log')
    t_song = song_df.alias('t_song')
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = t_log.join(t_song, \
                                 (col('t_log.song') == col('t_song.title')) & \
                                 (col('t_log.artist') == col('t_song.artist_name') \
				),how='inner') \
                           .select([col('t_log.timestamp'),\
			            col('t_log.userId'), \
			            col('t_log.level'), \
				    col('t_song.song_id'), \
                                    col('t_song.artist_id'), \
				    col('t_log.sessionId'), \
				    col('t_log.location'), \
				    col('t_log.userAgent')]) \
                           .withColumnRenamed('timestamp', 'start_time') \
                           .withColumnRenamed('userId', 'user_id') \
                           .withColumnRenamed('sessionId', 'session_id') \
                           .withColumnRenamed('userAgent', 'user_agent') \
                           .orderBy(F.asc('start_time'))
    
    # make sure songplays has desired schema
    songplays_schema = StructType([ 
                           StructField('start_time', TimestampType(), False), 
                           StructField('user_id',    StringType(),    False), 
                           StructField('level',      StringType(),    True),  
                           StructField('song_id',    StringType(),    False), 
                           StructField('artist_id',  StringType(),    False),
                           StructField('session_id', LongType(),      False),
                           StructField('location',   StringType(),    True),
                           StructField('user_agent', StringType(),    False)
                       ])
    songplays_table = sqlContext.createDataFrame(songplays_table.rdd, 
                                                 songplays_schema)
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.withColumn('year', F.year('start_time')) \
                   .withColumn('month', F.month('start_time')) \
                   .repartition(col('year'), col('month')) \
                   .write.partitionBy('year', 'month') \
		   .parquet(os.path.join(output_data, 'songplays'),\
		            mode='overwrite')


def main():
    """
    Main function which performs
      1. Create a SparkSession
      2. Load and process Song dataset
      3. Load and process Log dataset
      4. Stop SparkSession which stops its underlying SparkContext

    Arguments: None
    Return: None  
    """
    # Create SparkSession
    spark = create_spark_session()
    # Set path to input/output data
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://<YOUR_S3_PATH>"

    # Process Song dataset
    process_song_data(spark, input_data, output_data)    
    # Process Log dataset
    process_log_data(spark, input_data, output_data)

    # Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
