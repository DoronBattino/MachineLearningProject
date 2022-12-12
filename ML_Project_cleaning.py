#!/usr/bin/env python
# coding: utf-8

# # import Libraries

# In[121]:


import numpy as np 
import pandas as pd 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
import time
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
users_df = pd.read_csv("C:/Users/pavel/Downloads/Technion/6-Spring 2021/097209 - למידה חישובית 2/Project/users.csv",index_col = False,nrows=150000)


# # Import Data

# In[122]:


places_df  = pd.read_csv("C:/Users/pavel/Downloads/Technion/6-Spring 2021/097209 - למידה חישובית 2/Project/places.csv",index_col = False,nrows=150000)


# In[123]:


caracteristics_df = pd.read_csv("C:/Users/pavel/Downloads/Technion/6-Spring 2021/097209 - למידה חישובית 2/Project/caracteristics.csv", encoding="latin-1",index_col = False,nrows=150000)


# In[171]:


new_df = pd.read_csv("C:/Users/pavel/100000_full_features.csv",index_col = False,nrows=150000)


# # Preprocessing 

# ## Users 

# In[124]:


users_df = users_df.drop(["place", "catu", "sexe", "trajet", "secu", "locp", "actp", "etatp", "an_nais", "num_veh"], axis=1)


# ## properly order the grav measure from 1 to 4 

# In[125]:


users_df = users_df.replace(2,4)
users_df = users_df.replace(4,2)


# In[126]:


users_df


# ## create an additional "Total Gravity" column

# In[127]:


dict = {}
for x in users_df.Num_Acc:   
    s = users_df[users_df.Num_Acc == x].grav
    dict[x] = list(s)


# In[128]:


dict


# In[129]:


users_df=users_df.drop(["grav"],axis = 1)
users_df=users_df.drop_duplicates(subset="Num_Acc")
users_df['Total_grav'] = users_df['Num_Acc'].map(dict)


# In[130]:


users_df['Total_grav'].astype(str)


# In[131]:


# users_df = users_df['Total_grav'].replace('[^\d.]', '', regex=True).astype(str)


# In[132]:


# users_df['Num_Acc'].astype(str).astype(int)


# In[133]:


users_df.info()


# ## create an additional "num_of_inj" column

# In[134]:


num_of_inj_list = []
avg_of_grav = []
for cell in users_df['Total_grav']:
    num_of_inj = 0
    sum_of_grav = 0
    for i in cell:
        sum_of_grav += int(i)
        num_of_inj += 1
    num_of_inj_list.append(num_of_inj)


# In[135]:


users_df = users_df.assign(num_of_inj=num_of_inj_list)


# In[136]:


users_df


# ## Caracteristics 

# In[137]:


caracteristics_df = caracteristics_df[["Num_Acc","jour","mois","hrmn","lum","agg","int","atm","col","dep"]]


# In[138]:


caracteristics_df["atm"] = caracteristics_df["atm"].fillna(1)
caracteristics_df["atm"] = caracteristics_df["atm"].astype(int)


# In[139]:


caracteristics_df["int"] = caracteristics_df["int"].replace(0,1)


# In[140]:


caracteristics_df["hrmn"] = caracteristics_df["hrmn"].div(100).apply(np.floor)
caracteristics_df["hrmn"] = caracteristics_df["hrmn"].astype(int)


# In[141]:


caracteristics_df["dep"] = caracteristics_df["dep"].div(10).apply(np.floor)
caracteristics_df["dep"] = caracteristics_df["dep"].astype(int)


# In[142]:


caracteristics_df["jour"] = caracteristics_df["jour"].astype(int)
caracteristics_df["jour"].mean()
caracteristics_df["jour"].fillna(15)


# In[143]:


caracteristics_df["mois"] = caracteristics_df["mois"].astype(int)
caracteristics_df["mois"].mean()
caracteristics_df["mois"].fillna(6)


# In[144]:


caracteristics_df["lum"] = caracteristics_df["lum"].astype(int)
caracteristics_df["lum"].mean()
caracteristics_df["lum"].fillna(1)


# In[145]:


caracteristics_df["agg"] = caracteristics_df["agg"].astype(int)
caracteristics_df["agg"].mean()
caracteristics_df["agg"].fillna(1)


# In[146]:


caracteristics_df["col"].mean()
caracteristics_df["col"].fillna(4)
caracteristics_df["col"] = caracteristics_df["col"].where(caracteristics_df["col"] > 7,4)
caracteristics_df["col"] = caracteristics_df["col"].astype(int)


# ## Places

# In[147]:


places_df = places_df[["Num_Acc","catr","circ","nbv","vosp","prof","plan","surf"]]


# In[148]:


places_df["catr"].isna().sum()
places_df["catr"] = places_df["catr"].fillna(9)
places_df["catr"] = places_df["catr"].astype(int)


# In[149]:


places_df["circ"] = places_df["circ"].fillna(2)
places_df["circ"] = places_df["circ"].replace(0,2)
places_df["circ"] = places_df["circ"].astype(int)


# In[150]:


places_df["nbv"] = pd.to_numeric(places_df["nbv"], downcast="float")
places_df["nbv"] = places_df["nbv"].where(places_df["nbv"] < 6,0)
places_df["nbv"] = places_df["nbv"].fillna(2)
places_df["nbv"] = places_df["nbv"].replace(0,2)
places_df["nbv"] = places_df["nbv"].astype(int)


# In[151]:


places_df["vosp"].isna().sum()
places_df["vosp"].fillna(0)
places_df["vosp"] = places_df["vosp"].where(places_df["vosp"] > 3,0)
places_df["vosp"] = places_df["vosp"].astype(int)


# In[152]:


places_df["prof"].isna().sum()
places_df["prof"].fillna(0)
places_df["prof"] = places_df["prof"].where(places_df["prof"] > 4,0)
places_df["prof"] = places_df["prof"].astype(int)


# In[153]:


places_df["plan"].isna().sum()
places_df["plan"].fillna(0)
places_df["plan"] = places_df["plan"].where(places_df["plan"] > 4,0)
places_df["plan"] = places_df["plan"].astype(int)


# In[154]:


places_df["surf"] = places_df["surf"].fillna(1)
places_df["surf"] = places_df["surf"].replace(0,1)
places_df["surf"] = places_df["surf"].astype(int)


# # merging the tables

# In[155]:


new_df = users_df.merge(places_df, how='inner', on='Num_Acc')


# In[156]:


new_df = new_df.merge(caracteristics_df, how='inner', on='Num_Acc')


# ###### we now drop the "Num_Acc" column for it is unnecesserty for the classification  

# In[172]:


new_df = new_df.drop("Total_grav", axis=1)


# In[173]:


new_df = new_df.drop("Num_Acc", axis=1)


# # Split Train/Test/Validation

# ### the regular dataset 

# In[174]:


from sklearn.model_selection import train_test_split
#
X_training, X_test, y_training, y_test = train_test_split(new_df.drop("num_of_inj",axis=1), new_df["num_of_inj"], test_size=0.2, random_state=42)


# ### 4 worst features removed

# In[160]:


# worst_4_removed_df = new_df.drop(['col','plan','prof','vosp'], axis=1)


# In[161]:


# from sklearn.model_selection import train_test_split

# X_training, X_test, y_training, y_test = train_test_split(worst_4_removed_df.drop("num_of_inj",axis=1), worst_4_removed_df["num_of_inj"], test_size=0.2, random_state=42)


# #### Time taken : 2.561527729034424, 
# #### Accuracy : 0.5499859194593072

# ### 5 best dataframe 

# In[162]:


# best_5_df = new_df[["num_of_inj","dep","jour","hrmn","mois","nbv"]]


# In[163]:


# from sklearn.model_selection import train_test_split

# X_training, X_test, y_training, y_test = train_test_split(best_5_df.drop("num_of_inj",axis=1), best_5_df["num_of_inj"], test_size=0.2, random_state=42)


# In[183]:


X_train, X_Val, y_train, y_Val = train_test_split(X_training, y_training, test_size=0.2, random_state=42)


# In[184]:


t0=time.time()
model_rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)
model_rf.fit(X_train,y_train)
print('Time taken :' , time.time()-t0)


# In[185]:


y_pred = model_rf.predict(X_Val)
score_rf = accuracy_score(y_Val,y_pred)
print('Accuracy :',score_rf)


# In[186]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model_rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')


# In[187]:


importances


# ### without new split

# In[176]:


t0=time.time()
model_rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)
model_rf.fit(X_training,y_training)
print('Time taken :' , time.time()-t0)


# In[178]:


y_pred = model_rf.predict(X_test)
score_rf = accuracy_score(y_test,y_pred)
print('Accuracy :',score_rf)


# In[181]:


importances = pd.DataFrame({'feature':X_training.columns,'importance':np.round(model_rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')


# In[182]:


importances


# ### export to_csv

# In[170]:


new_df.to_csv('150000_full_features.csv',index=False)


# In[120]:


# best_5_df.to_csv('best_5_dataframe.csv',index=False)


# In[ ]:


t0=time.time()
model_rf = RandomForestClassifier(n_estimators=100, max_depth= 5, max_features= 3, random_state=0, n_jobs=-1)
model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_Val)
score = accuracy_score(y_Val,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[ ]:


t0=time.time()
model_rf = RandomForestClassifier(n_estimators=50, max_depth= 5, max_features= 3, random_state=0, n_jobs=-1)
model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_Val)
score = accuracy_score(y_Val,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[ ]:


t0=time.time()
model_rf = RandomForestClassifier(n_estimators=7, max_depth= 15, max_features= 6, random_state=0, n_jobs=-1)
model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_Val)
score = accuracy_score(y_Val,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)

