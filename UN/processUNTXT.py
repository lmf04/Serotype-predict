import re
import pandas as pd
import numpy as np

path="C:/Users/36147/Desktop/UNhmtp210.fasta"
excel = np.zeros((346, 2))
excel =pd.DataFrame(excel)

with open(path,"r" ) as f:
    content=f.read()
    content=content.replace("/n","")
    titles=re.findall(r">([^>]*?\.1)",content)
    seqs = re.split(r'>[A-Z0-9]+\.\d+/\d+-\d+', content)
    del seqs[0]
    count=0
    index=0
    for seq in seqs:
        print(seq)
        seq=seq.replace("\n","")
        excel.iat[index,1]=seq
        excel.iat[index,0]=titles[count]     
        count=count+1
        index=index+1

excel.to_csv("D:/Python/UNhmtp210.csv",index=False,header=None)
