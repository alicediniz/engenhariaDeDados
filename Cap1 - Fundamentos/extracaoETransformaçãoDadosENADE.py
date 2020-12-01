#!/usr/local/bin/python3
import pandas as pd
import numpy as np
import zipfile
import requests
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import FloatType, StringType
from io import BytesIO


def importEnadeFile():
    # Downloading and unzipping ENADE 2019 data
    os.makedirs('./ENADE2019', exist_ok=True)
    url_enade = "http://download.inep.gov.br/microdados/Enade_Microdados/microdados_enade_2019.zip"
    filebytes = BytesIO(requests.get(url_enade).content)
    data_zip = zipfile.ZipFile(filebytes)
    data_zip.extractall("./ENADE2019")


if __name__ == '__main__':
    file_path = "./ENADE2019/microdados_enade_2019/2019/3.DADOS/microdados_enade_2019.txt"


    if(not os.path.isfile(file_path)):
        importEnadeFile();


    spark = SparkSession.builder.appName("ENADE2019").getOrCreate()
    enade_data = spark.read.option("header","true")\
                           .option("sep",";")\
                           .option("inferSchema", "true")\
                           .csv(file_path)
                           
    # Changing Decimal Separator from comma to dot
    enade_data = enade_data.withColumn('NT_GER', func.regexp_replace(func.col('NT_GER'), '\\.,', ''))\
                           .withColumn('NT_GER', func.regexp_replace(func.col('NT_GER'), ',', '.'))\
                           .withColumn('NT_GER', func.col('NT_GER').cast(FloatType()))
    
    
    # Exercise 1: getting the highest score
    enade_data.dropna().agg(func.max('NT_GER').alias("Highest Score")).show()
    
    # Exercise 2: NT_GER statistical data for students between 20yo and 50yo
    enade_data.select(func.col('NT_GER')).filter(func.col('NU_IDADE')\
              .between(20,50)).describe().show()
              
    # Exercise 2: Count number of students per gender
    enade_data.groupBy(func.col('TP_SEXO')).count().show()
    
    # Exercise 3: Create a dictionary using database content
    enade_data.groupBy(func.col('CO_REGIAO_CURSO')).agg({
        "NT_GER": "mean",
        "NT_FG": "mean",
        "NT_CE": "mean"
    }).orderBy(func.col('CO_REGIAO_CURSO')).show()
    
    # Exercise 4: Transform university type column
    ids_universidades_publicas = [93,115,116,10001,10002,10003]
    enade_data = enade_data.withColumn('CO_CATEGAD', func.when(func.col('CO_CATEGAD').isin(ids_universidades_publicas), "Pública")\
                                                                                     .otherwise("Privada"))
                                                                        
    enade_data.groupBy(func.col('CO_CATEGAD')).count().show()
    
    # Exercise 5: Transform CO_MODALIDADE so it's readable
    study_mode = {
                    "0" : "EAD",
                    "1" : "Presencial"
                 }
                 
    map_study_func = func.udf(lambda code : study_mode.get(code), StringType())
    enade_data = enade_data.withColumn('CO_MODALIDADE', func.col('CO_MODALIDADE').cast(StringType()))\
                           .withColumn('CO_MODALIDADE', map_study_func(func.col('CO_MODALIDADE')))
    
    enade_data.groupBy(func.col('CO_MODALIDADE')).count().show()
    
    
    spark.stop()
