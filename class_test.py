import pandas as pd
import numpy as np
import glob

file_list = glob.glob('/Volumes/VINCE/OAC/class_test/*.txt*')[:-2]
result = pd.read_csv('/Volumes/VINCE/OAC/result.csv')

df=pd.DataFrame()
for item in file_list:
	df_tmp = pd.read_csv(item, header=None)
	df = df.append(df_tmp, ignore_index=True)

df[0] = df[0].str[0:9]
df.columns = ['obj_ID','cl']

grouped = df.groupby(1)
grouped.count().plot.pie(subplots=True,figsize=(4,4),
	autopct='%.2f', fontsize=20,legend=False)

merged = pd.concat([df,result], axis=1)
tmp = merged[(merged[1] == 'ok') & (merged['VREL_helio']< 2200) & (merged['VREL_helio']> 450)]
print len(tmp)

tmp = merged[(merged[1] == 'ok') & (merged['VREL_helio']< 450) & (merged['VREL_helio']> -450)]
print len(tmp)

good = merged[(merged[1] == 'ok')]
galaxies = merged[(merged[1] == 'gal')]


Nicola = pd.read_csv('/Volumes/VINCE/OAC/class_test/class_NRN.txt', header=None)
Crescenzo = pd.read_csv('/Volumes/VINCE/OAC/class_test/class_Tortora.txt', header=None

sum((Nicola[1] == 'ok'))
sum((Crescenzo[1] == 'ok'))

sum((Crescenzo[1] == 'ok') & (Nicola[1] == Crescenzo[1]))