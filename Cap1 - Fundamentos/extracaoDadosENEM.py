#!/usr/local/bin/python3
import pandas as pd
import numpy as np
import zipfile
import requests
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from io import BytesIO

# Downloading and unzipping ENEM 2019 data
os.makedirs('./ENEM2019', exist_ok=True)
url_enem = "http://download.inep.gov.br/microdados/microdados_enem_2019.zip"
filebytes = BytesIO(requests.get(url_enem).content)
data_zip = zipfile.ZipFile(filebytes)
data_zip.extractall("./ENEM2019")


spark = SparkSession.builder.appName("ENEM2019").getOrCreate()
# Accessing ENEM data
enem_data = spark.read.option("header", "true").\
            option("inferSchema","true").\
            option("sep",";").\
            option("encoding", "UTF-8").\
            csv("./ENEM2019/microdados_enem_2019/DADOS/MICRODADOS_ENEM_2019.csv")


mineiros = enem_data.filter(func.col("SG_UF_RESIDENCIA")== "MG")

# Questão 1:
# Qual é a média da nota em matemática de todos os alunos mineiros?
media_mat_mg = mineiros.filter(func.col("TP_PRESENCA_MT") == 1)\
                     .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_MG"))

media_mat_mg.show()


#Questão 2:
# Qual é a média da nota em Linguagens e Códigos de todos os alunos mineiros?
media_ling_mg = mineiros.filter(func.col("TP_PRESENCA_LC") == 1)\
                      .agg(func.round(func.avg("NU_NOTA_LC"),2).alias("AVG_LC_MG"))

media_ling_mg.show()
                      

#Questão 3:
# Qual é a média da nota em Ciências Humanas dos alunos do sexo feminino mineiros?
media_ch_mg_fem = mineiros.filter((func.col("TP_PRESENCA_CH") == 1) & (func.col("TP_SEXO") == "F"))\
                    .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_FEM_MG"))

media_ch_mg_fem.show()


#Questão 4:
# Qual é a média da nota em Ciências Humanas dos alunos do sexo MASCULINO?
media_ch_masc = enem_data.filter((func.col("TP_PRESENCA_CH") == 1) & (func.col("TP_SEXO") == "M"))\
                  .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_MASC"))

media_ch_masc.show()


# Questão 5
# Qual é a média da nota em Matemática dos alunos do sexo FEMININO que moram na cidade de Montes Claros?
mulheres_montes_claros = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA") == "Montes Claros") & (func.col("TP_SEXO") == "F"))
media_mat_mulheres_mc = mulheres_montes_claros.filter(func.col("TP_PRESENCA_MT") == 1)\
                        .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_FEM_MC"))

media_mat_mulheres_mc.show()


# Questão 6
# Qual é a média da nota em Matemática dos alunos do município de Sabará que possuem TV por assinatura na residência?
alunos_sabara_assinatura = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA").like("%Sabar%")) & (func.col("Q021") == "B"))
media_mat_sabara_tv = alunos_sabara_assinatura.filter(func.col("TP_PRESENCA_MT") == 1)\
            .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_TV_SABARA"))

media_mat_sabara_tv.show()


#Questão 7
# Qual é a média da nota em Ciências Humanas dos alunos mineiros que possuem dois fornos micro-ondas em casa?
media_ch_mg_doismicroondas = mineiros.filter((func.col("Q016") == "C") & (func.col("TP_PRESENCA_CH") == 1))\
                         .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_MG_DOISMICRO"))

media_ch_mg_doismicroondas.show()


#Questão 8
# Qual é a nota média em Matemática dos alunos mineiros cuja mãe completou a pós-graduação?
media_mat_mg_maepos = mineiros.filter((func.col("Q002") == "G") & (func.col("TP_PRESENCA_MT") == 1))\
                        .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_MAEPOS"))

media_mat_mg_maepos.show()


#Questão 9
# Qual é a nota média em Matemática dos alunos de Belo Horizonte e de Conselheiro Lafaiete?
bh_cl = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA") == "Belo Horizonte") | (func.col("NO_MUNICIPIO_RESIDENCIA") == "Conselheiro Lafaeite"))
media_mat_bh_cl = bh_cl.filter(func.col("TP_PRESENCA_MT") == 1)\
                  .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_BH_CL"))

media_mat_bh_cl.show()


#Questão 10
#Qual é a nota média em Ciências Humanas dos alunos mineiros que moram sozinhos?
media_ch_sozinhos_mg = mineiros.filter((func.col("Q005") == "1") & (func.col("TP_PRESENCA_CH") == 1))\
                       .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_SOZINHO_MG"))

media_ch_sozinhos_mg.show()


# Questão 11
# Qual é a nota média em Ciências Humanas dos alunos mineiros cujo pai completou Pós graduação e possuem renda familiar entre R$ 8.982,01 e R$ 9.980,00.
media_ch_mg_pai_pos_renda = mineiros.filter((func.col("Q001") == "G") & (func.col("TP_PRESENCA_CH") == 1) & (func.col("Q006") == "M"))\
                        .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_MG_PAI_POS_RENDA_M"))

media_ch_mg_pai_pos_renda.show()


# Questão 12
# Qual é a nota média em Matemática dos alunos do sexo Feminino que moram em Lavras e escolheram “Espanhol” como língua estrangeira?
media_mat_lavras_espanhol = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA") == "Lavras") & \
                                  (func.col("TP_PRESENCA_MT") == 1) & \
                                  (func.col("TP_LINGUA") == 1) & \
                                  (func.col("TP_SEXO") == "F"))\
                  .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_LAVRAS_ESPANHOL"))

media_mat_lavras_espanhol.show()


# Questão 13
# Qual é a nota média em Matemática dos alunos do sexo Masculino que moram em Ouro Preto?
media_mat_ouropreto_masc = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA") == "Ouro Preto") & \
                                 (func.col("TP_PRESENCA_MT") == 1) & \
                                 (func.col("TP_SEXO") == "M"))\
                           .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_OP_MASC"))

media_mat_ouropreto_masc.show()


# Questão 14
# Qual é a nota média em Ciências Humanas dos alunos surdos?
media_ch_surdos = enem_data.filter((func.col("TP_PRESENCA_CH") == 1) & ((func.col("IN_SURDEZ") == 1) | (func.col("IN_SURDO_CEGUEIRA") == 1)) )\
            .agg(func.round(func.avg("NU_NOTA_CH"),2).alias("AVG_CH_SURDEZ"))

media_ch_surdos.show()


# Questão 15
# Qual é a nota média em Matemática dos alunos do sexo FEMININO, que moram em Belo Horizonte, Sabará, Nova Lima e Betim e possuem dislexia?
moradores_grande_BH = mineiros.filter((func.col("NO_MUNICIPIO_RESIDENCIA").like("%Belo%Horizonte")) | (func.col("NO_MUNICIPIO_RESIDENCIA").like("%Sabar%")) | \
                                        (func.col("NO_MUNICIPIO_RESIDENCIA").like("%Nova%Lima%")) | \
                                        (func.col("NO_MUNICIPIO_RESIDENCIA").like("%Betim%")))
moradores_grande_BH_dislexas = moradores_grande_BH.filter((func.col("IN_DISLEXIA") == 1) & (func.col("TP_SEXO") == "F"))
media_mat_moradores_grande_BH_dislexas = moradores_grande_BH_dislexas.filter(func.col("TP_PRESENCA_MT") == 1)\
                                         .agg(func.round(func.avg("NU_NOTA_MT"),2).alias("AVG_MAT_FEM_BH"))

media_mat_moradores_grande_BH_dislexas.show()

spark.stop()

