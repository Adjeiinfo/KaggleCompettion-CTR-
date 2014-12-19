# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:34:11 2014

@author: KKouassi2
"""

"""
This is basically to play with the data using pandas

"""

import pandas as pd
import matplotlib.pyplot as plt

###################Define the global variables#####################
TRAIN_DATA = "train_1_1.csv"


broken_df = pd.read_csv(TRAIN_DATA, low_memory=False)


#get the names
colnumames = broken_df.columns.values

#colnumames output is :
#['id' 'click' 'hour' 'C1' 'banner_pos' 'site_id' 'site_domain'
# 'site_category' 'app_id' 'app_domain' 'app_category' 'device_id'
# 'device_ip' 'device_model' 'device_type' 'device_conn_type' 'C14' 'C15'
# 'C16' 'C17' 'C18' 'C19' 'C20' 'C21']
#
colnumames = colnumames[4:15]
##################################plotting ##########################
row = 3
col = 4
f, arr = plt.subplots(row,col)



##############################Helper function #######################

def exploredata(colname):
    colcount = broken_df[colname].value_counts()
    #colcount.plot(kind='bar')
    return colcount
    
    

#################Play with the data #################################
def plotingdata():
#broken_df["C1"].plot()

#appid= broken_df['app_id'].value_counts()
#appdomain = broken_df["app_domain"].value_counts()


#appdomain.plot(kind= 'bar')

#sitcategory = broken_df['site_category'].value_counts()
#sitcategory.plot(kind='bar')

    for i, col in enumerate(colnumames):
        plt.figure()
        val = exploredata(col)
        # print i
       # r = i % 3
        #c = i % 4
        # print i, col   
        #val.plot(ax=arr[r,c], kind='bar')
        val.plot()
        plt.title(col)
    
        plt.show()
   
   
##############################Concatonate some cols and see how to it works 
###concat all site relate[site_id' 'site_domain'# 'site_category']
site = broken_df['site_id']+ broken_df['site_domain'] + broken_df['site_category'] 

site.name = "Site"
###concat all app related ['app_id' 'app_domain' 'app_category']
appcol = broken_df['app_id']+ broken_df['app_domain'] + broken_df['app_category'] 

appcol.name = "Application"
#concat device related [device_id' 'device_ip' 'device_model' 'device_type']
#devicecol =  broken_df['device_id']+ broken_df['device_ip'] + broken_df['device_model']+ broken_df['device_type'] 
#devicecol.name = "Device"



siteval=  site.value_counts()
siteval.plot(kind='bar')

  




    
