import pandas as pd
import math
import numpy as np
from scipy.stats.stats import pearsonr
# User-Based Neighborhood 
example = {'U_1': [7.0,6.0,7.0,4.0,5.0,4.0], 
           'U_2': [6,7,None,4,3,4],
           'U_3': [None,3,3,1,1,None],
           'U_4': [1,2,2,3,3,4],
           'U_5': [1,None,1,2,3,3]}

df = pd.DataFrame(data=example).T

  
def findMean(dataF):
    # Transposes to get Item rating
    mean_list,dataF = [],dataF.T
    for items in dataF:
        summen,nr = 0,0
        for value in dataF[items].tolist():
            if not math.isnan(value):
                nr += 1
                summen+= value
        mean_list.append(summen/nr)
    return (mean_list)

def Pearson(df,user_u,user_v):
    user_u, user_u_mean = df[user_u].tolist()[:-1], df[user_u].tolist()[-1]
    user_v, user_v_mean = df[user_v].tolist()[:-1], df[user_v].tolist()[-1]
    snitt_index = []
    for index in range(len(user_u)):
        if math.isnan(user_v[index]) or math.isnan(user_u[index]):
            snitt_index.append(index)
    
    for index in snitt_index[::-1]:
        user_u.pop(index)
        user_v.pop(index)
    user_u = [(r - user_u_mean) for r in user_u]
    user_v = [(r - user_v_mean )for r in user_v]
    # print(user_u, user_v)
    return pearsonr(user_u,user_v)[0]

def predict_rank(df, user_u, itemNR):
    users = [user for user in df.T]
    print(users)
    u = df.T[user_u]["Mean"]
    for user in users:
        if not math.isnan(df.T[user][itemNR]) :
            over = df.T[user]["Sim(i,3)"]*(df.T[user][itemNR]-df.T[user]["Mean"])
    under = sum([abs(i) for i in df["Sim(i,3)"].tolist()])
    print(u + (over/under))

df['Mean'] = findMean(df)
users = [i for i in df.T]
sim_list = []
for user in users:
    sim_list.append(Pearson(df.T, user,"U_3"))
df["Sim(i,3)"] = sim_list

predict_rank(df, "U_3", 5)

print(df)
#np.corrcoef([1,2,3], [3,3,3])
