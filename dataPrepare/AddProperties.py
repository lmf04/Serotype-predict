import numpy as np
import pandas as pd


hmtp210 = pd.read_csv('D:\Python\hmtp210.csv',header=None)
#下面这个用来做多维数组
numhmtp210 = hmtp210.to_numpy()

aaindex = pd.read_csv("D:/Python/RF/aaindex1.csv",header=None)
#制作字典
Hydrophobicity= aaindex[(aaindex[0] == "ARGP820101")]

aaNameH=aaindex.iloc[0, 1:]
aaNameH=aaNameH.values
aaNameH = [str(item) for item in aaNameH]
aaValueH = Hydrophobicity.iloc[0, 1:]
aaValueH  = aaValueH .tolist()
aaValueH = [float(item) for item in aaValueH]

HyMap= {key: value for key, value in zip(aaNameH, aaValueH)}

#遍历序列并制作替换表
HydroList = np.zeros((54, 361))
x=0
for seq in hmtp210.iloc[1:,3]:
    seq=list(seq)
    y=0
    for aa in seq:
        replaced_value = HyMap.get(aa, None)
        if replaced_value is not None:
            HydroList[x, y] = replaced_value
        else:
            HydroList[x, y] = 0
        y=y+1
    x=x+1



polarity = aaindex[(aaindex[0]== "GRAR740102")]
aaNameP=aaindex.iloc[0, 1:]
aaNameP=aaNameP.values
aaNameP = [str(item) for item in aaNameP]
aaValueP = polarity.iloc[0, 1:]
aaValueP  = aaValueP.tolist()
aaValueP = [float(item) for item in aaValueP]

PoMap= {key: value for key, value in zip(aaNameP, aaValueP)}

#遍历序列并制作替换表
PolaList = np.zeros((54, 361))
x=0
for seq in hmtp210.iloc[1:,3]:
    seq=list(seq)
    y=0
    for aa in seq:
        replaced_value = PoMap.get(aa, None)
        if replaced_value is not None:
            PolaList[x, y] = replaced_value
        else:
            PolaList[x, y] = 0
        y=y+1
    x=x+1



numhmtp210= np.concatenate((numhmtp210, PolaList,HydroList),axis=1)

np.save('numhmtp2101.npy', numhmtp210)