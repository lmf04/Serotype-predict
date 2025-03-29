import re
import pandas as pd

path="D:/R/randomForest/htmp210seq/region0 .txt"
excel=pd.read_excel("D:/Python/hmtp210.xlsx",header=None)
value = excel.loc[0, 0]

with open(path,"r" ) as f:
    content=f.read()
    content=content.replace("/n","")
    titles=re.findall(r">([^>]*?\.1)",content)
    seqs = re.split(r'>[A-Z0-9]+\.\d+/\d+-\d+', content)
    del seqs[0]
    count=0
    for seq in seqs:
        print(seq)
        seq=seq.replace("\n","")
        for index in range(1, len(excel)):
            if excel.iloc[index, 0]==titles[count]:
                excel.iat[index,3]=seq
        count=count+1

excel.to_csv("D:/Python/hmtp210.csv",index=False,header=None)
