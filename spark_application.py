import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, when, sum, expr


class SparkApplication:
    def __init__(self, appName):
        self.spark = SparkSession.builder.appName(appName).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def stop(self):
        self.spark.stop()

    def analyze_csv(self, dir_path):
        df = self.spark.read.csv(
            os.path.join(dir_path, "*.csv"), header=True, inferSchema=True
        )

        window = Window.partitionBy("video_file", "target_id").orderBy("timestamp_sec")

        df = df.withColumn("prev_state", lag("target_state").over(window))

        df = df.withColumn(
            "is_hit",
            when(
                (col("target_state") == "Broken")
                & (col("prev_state").isNotNull())
                & (col("prev_state") != "Broken"),
                1,
            ).otherwise(0),
        )

        series = df.groupBy("video_file", "target_id").agg(
            expr("sum(is_hit) as hit_count"),
            expr("min(timestamp_sec) as start_time"),
            expr("max(timestamp_sec) as end_time"),
        )

        longest = series.orderBy(col("hit_count").desc()).limit(1)

        longest.show(truncate=False)
