import pandas as pd
import sys
from tqdm import tqdm
from interruptingcow import timeout

sys.path.append('./')

from utils.URLFeaturizer import UrlFeaturizer


l = ['DefacementSitesURLFiltered.csv','phishing_dataset.csv','Malware_dataset.csv','spam_dataset.csv','Benign_list_big_final.csv']
for ind, url in enumerate(l):
    l[ind]="FinalDataset/URL/"+url
emp = UrlFeaturizer("").run().keys()

A = pd.DataFrame(columns = emp)
t=[]
for j in l:
    print(j)
    d=pd.read_csv(j,header=None).to_numpy().flatten()
    for i in tqdm(d):
        try: 
            with timeout(30, exception = RuntimeError):
                temp=UrlFeaturizer(i).run()
                temp["File"]=j.split(".")[0]
                t.append(temp)
        except RuntimeError: 
            pass 
A=A.append(t)
A.to_csv("FinalDataset/final/features.csv")