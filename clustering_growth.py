import sys
sys.path.append('/home/moneshsharma/')

from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from dateutil.relativedelta import *
from redshift_creds.database import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
# from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import datetime
import datetime as dt
from datetime import timedelta,date
import time
import warnings
from functools import reduce
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',30)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def dataPrep(dataframe):
    
    prep_df = dataframe.copy(deep=True)
    
    prep_df = prep_df.rename(columns={'age_group_category':'age_bucket','location_type':'location','duration_gold':'gold_duration','cl_sol_freq':'cl_sol'})
    cat_brand_rename = ['sun_freq','cl_freq', 'eye_freq', 'loyalty_freq', 'npr_freq',
    'hto_freq', 'prog_freq', 'online_excl_freq', 'lk_air_freq', 'kids_freq','vc_freq', 'jj_freq', 'od_freq', 'huslr_freq']
    rename_dict = {col:col.split('_')[0]  for col in cat_brand_rename}
    prep_df = prep_df.rename(columns=rename_dict)

    prep_df['gold_duration'] = np.where(prep_df['gold_duration'].isin(['2 year','1 year','Expired'])==False,'More than 2 years',prep_df['gold_duration'])
    prep_df['gold_duration'] = np.where((prep_df['gold_start_date'].isnull()) & (prep_df['gold_duration']=='Expired'),'Not Bought',prep_df['gold_duration'])
    prep_df['frequency'] = np.where(prep_df['frequency']<=3,'freq_1_to_3','freq_above_3')
    
    prep_df['acquisition_date']= pd.to_datetime(prep_df['acquisition_product'],errors='coerce')
    prep_df['today'] = pd.to_datetime(datetime.date.today())
    prep_df['customer_age'] = (prep_df['today'] - prep_df['acquisition_date']).dt.days/365
    
    
    cat_col = ['frequency','recency_bucket','age_bucket','gender','location','gold_duration']
    num_col = ['customer_phone','customer_category','customer_age','gross_orders','atv','basket_size','return_perc','eye', 'sun', 'npr', 'cl', 'cl_sol', 'loyalty', 'hto', 'prog', 'vc',
               'jj', 'lk', 'kids', 'huslr','online_freq','online_freq_30', 'session_cnt', 'session_cnt_30', 'offline_freq',
               'offline_freq_30','qms_count', 'qms_count_30', 'eye_count',
               'eye_count_12m', 'eye_count_30']

    cat_brand_col = ['eye', 'sun', 'npr', 'cl', 'cl_sol', 'loyalty', 'hto', 'prog', 'vc',
                     'jj', 'lk', 'kids', 'huslr']
    activity_col = ['online_freq_30','offline_freq_30','qms_count_30','eye_count_30','session_cnt_30']

    data_ = prep_df[num_col+cat_col]


    for col in cat_col[1:]:
        data_[col] = np.where(data_[col].isnull(),'Unknown',data_[col])

    for col in cat_brand_col:
        data_[col] = data_[col]/prep_df['item_count']
        data_[col] = np.where(data_[col].isnull(),0,data_[col])
        data_[col] = data_[col].replace(np.inf,np.nan)
        data_[col] = np.where(data_[col].isnull(),0,data_[col])

    for col in activity_col:
        data_[col] = np.where(data_[col]>0,1,0)

    data_['atv'] = np.where(data_['atv'].isnull(),0,data_['atv'])
    data_['basket_size'] = np.where(data_['basket_size'].isnull(),0,data_['basket_size'])
    
    ## separating into numerical and categorical 
    
    # numerical
    numerical = data_[num_col]
    num_drop_col = ['prog','hto']
    numerical = numerical.drop(columns=num_drop_col)
    
    numerical_cube_trans = numerical.iloc[:,4:].apply(lambda x:np.sign(x)*(abs(x)**0.33))
    
    # categorical
    categorical = data_[cat_col[:]]
    one_hot_encode = pd.get_dummies(categorical,drop_first=False,dtype=int)
    
    # cluster data
    cluster_data = pd.concat([numerical_cube_trans.iloc[:,:],one_hot_encode],axis=1)
    
    # numerical not tranformed and categorical one encoded
    not_numerical_tranformed_cat_ohe = pd.concat([data_[num_col],one_hot_encode],axis=1)

    return data_,not_numerical_tranformed_cat_ohe,cluster_data

def create_importance_dataframe(pca, original_num_df):
    
    importance_df  = pd.DataFrame(pca.components_)
    importance_df.columns  = original_num_df.columns
    importance_df =importance_df.apply(np.abs)
    importance_df=importance_df.transpose()
    num_pcs = importance_df.shape[1]
    new_columns = [f'PC{i}' for i in range(1, num_pcs + 1)]
    importance_df.columns  =new_columns

    return importance_df

def dimensionalityReduction(cluster_data:pd.DataFrame) -> pd.DataFrame:
    
    X = cluster_data.to_numpy()
    pca = PCA(n_components=12)
    pca.fit(X)
    pca_transformed = pca.transform(X)
    
    print("Explained Variance:",pca.explained_variance_ratio_)
    print("Singular Values:",pca.singular_values_)
    
    print("total_variance:",reduce(lambda a,b:a+b,pca.explained_variance_ratio_))
    sns.barplot(x=np.arange(1,13),y=np.cumsum(pca.explained_variance_ratio_))
    
    importance_df  =create_importance_dataframe(pca,cluster_data)

    print(importance_df)

    ## PC1 top 10 important features
    pc1_top_10_features = importance_df['PC1'].sort_values(ascending = False)[:10]
    print(), print(f'PC1 top 10 feautres are \n')

    ## PC2 top 10 important features
    pc2_top_10_features = importance_df['PC2'].sort_values(ascending = False)[:10]
    print(), print(f'PC2 top 10 feautres are \n')

    ## PC3 top 10 important features
    pc3_top_10_features = importance_df['PC3'].sort_values(ascending = False)[:10]
    print(), print(f'PC3 top 10 feautres are \n')
    
    pca_col_list =  ['col'+str(i) for i in range(1,13)]
    pca_df = pd.DataFrame(pca_transformed,columns=pca_col_list)
    
    return pca_df

def kmeansClustering(data:pd.DataFrame):
    
    model = KMeans(n_clusters = 4, init = "k-means++")
    label = model.fit_predict(data)

    return label

def clusterSummary(dataframe):
    
    mean_sample = dataframe.groupby(['label']).agg('mean')
    mean_pop = dataframe.mean().iloc[:-1]
    pop_df = pd.DataFrame(mean_pop,columns=['population_mean'])
    
    perc_col = ['return_perc', 'eye', 'sun', 'npr', 'cl', 'cl_sol', 'loyalty', 'hto',
                'prog', 'vc', 'jj', 'lk', 'kids', 'huslr','online_freq_30','offline_freq_30',
                'qms_count_30','eye_count_30','eye_count_12m','session_cnt_30',
                'age_bucket_10-20', 'age_bucket_18-30','age_bucket_30-40', 'age_bucket_40-60', 'age_bucket_Above 60',
                'age_bucket_Other', 'age_bucket_Under 10', 'age_bucket_Unknown',
                'gender_Female', 'gender_Kid', 'gender_Male', 'gender_Online',
                'gender_Unknown', 'location_Metro', 'location_Non-Metro',
                'location_Unknown', 'gold_duration_1 year', 'gold_duration_2 year',
                'gold_duration_Expired','gold_duration_Not Bought','gold_duration_More than 2 years',
                'recency_bucket_0 to 12 Months','recency_bucket_12 to 24 Months','recency_bucket_24+ Months']
    concatenated = pd.concat([mean_sample.T,pop_df],axis=1)

    for idx in perc_col:
        try:
            concatenated.loc[idx,:]=concatenated.loc[idx,:]*100
            concatenated.rename(index={idx:idx+' '+'%'},inplace=True)
        except Exception as e:
            print(e)
    
    return concatenated

if __name__ == '__main__':
    
    start_time = time.time()

    date_today = str(date.today())
    print('Date Today:',date_today)

    # query = f"""select * from customercharter.customer_charter_lifetime where date(ingested_at)='{date_today}' 
    #             and customer_category not in ('Unknown Category','Only Gold') """
    # data = mySQLRead(query)

    query = f"""select cast(customer_phone as text), gender, state, city, country,
            acquisition_date, gold_0_acq_date, satisfaction, location_type,
            customer_profile, customer_score, gv_discount, return_orders,
            cancelled_orders, last_trans_date, last_order_id,
            acquisition_product, gross_revenue, gross_orders, qty,
            gross_purchase_freq, item_count, exchange_orders_freq,
            exchange_item_freq, reverse_orders_freq, reverse_item_freq,
            rto_orders_freq, rto_item_freq, net_qty, net_revenue,
            net_orders, sun_freq, cl_freq, cl_sol_freq, eye_freq,
            loyalty_freq, npr_freq, hto_freq, prog_freq, online_excl_freq,
            lk_air_freq, kids_freq, vc_freq, jj_freq, od_freq,
            huslr_freq, online_freq, online_freq_30, offline_freq,
            offline_freq_30, atv, asp, basket_size,
            session_cnt, session_cnt_30, gold_start_date, gold_expiry_date,
            status, duration_gold, qms_count, qms_count_30, eye_count,
            eye_count_30, eye_count_12m, frequency, recency,
            recency_bucket,age_group_category,customer_category, return_perc,
            ingested_at from customercharter.customer_charter_lifetime where date(ingested_at)='{date_today}' 
            and customer_category not in ('Unknown Category','Only Gold') """
    data = mySQLRead(query)
    print('Query Run Successful:',data.head(10))

    input_data = pd.DataFrame()
    complete_summary = pd.DataFrame()

    for cat in data['customer_category'].unique():
        temp = data.loc[data['customer_category']==cat].reset_index(drop=True)
        
        data_original,data_post_cluster,cluster_df = dataPrep(temp)
        dim_red_df = dimensionalityReduction(cluster_df)

        assign_labels = kmeansClustering(dim_red_df)
        
        cluster_df['label'] = assign_labels
        data_original['label'] = assign_labels
        data_post_cluster['label'] = assign_labels
        temp['label'] = assign_labels
        
        # fixing the customer behavior key [ 0:low,2:Moderate,1:High,3:Elite]
        nature_df = data_post_cluster.groupby(['label']).agg(mean=('atv','mean')).reset_index()
        nature_df = nature_df.sort_values('mean').reset_index(drop=True)
        nature_df['label_updated'] = pd.Series(data=[0,2,1,3])
        label_dict = dict(zip(nature_df['label'],nature_df['label_updated']))

        data_post_cluster['label'] = data_post_cluster['label'].map(label_dict) 
    
        input_data = pd.concat([input_data,data_post_cluster],axis=0,ignore_index=True)

        summary = clusterSummary(data_post_cluster.iloc[:,2:])
        summary['segment'] = cat
    
        complete_summary = pd.concat([complete_summary,summary],axis=0,ignore_index=False)

    complete_summary = complete_summary.reset_index()
    complete_summary = complete_summary.rename(columns={'index':'features'})
    
    ## Dataframe to be ingested prod
    prod = input_data[['customer_phone','customer_category','label']]
    prod['label'] = np.where(prod['label']==0,'low',
                            np.where(prod['label']==2,'moderate',
                            np.where(prod['label']==1,'high','elite')))
    prod = prod.rename(columns={'customer_category':'segment','label':'cluster'})
    print('Total Customers Labelled:',prod.shape[0])
    print('Labelled DataFrame:',prod.head())
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time Elapsed in Minutes:",time_elapsed/60)