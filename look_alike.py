import sys
sys.path.append('/home/moneshsharma/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from redshift_creds.database import *

import pickle
# from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import datetime
import datetime as dt
from datetime import timedelta,date
from dateutil.relativedelta import *
import time
import warnings
from functools import reduce
from scipy.special import rel_entr
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',60)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def dataPrep(dataframe):

    '''
    Preprocesses the input dataframe to prepare it for further operations.

    This function performs necessary data cleaning and transformation steps such as:
    - Handling missing values
    - Encoding categorical variables
    - Transforming numerical features and categorical features
    - Removing duplicates
    - Other preprocessing steps as required by the processes

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe that needs to be preprocessed.

    Returns:
    --------
    pandas.DataFrame
        A cleaned and preprocessed dataframe ready for further operations.
        returns three dataframes 
        - original dataframe where no tranformations are applied
        - dataframe with no numerical transformation but one hot encoded categorical tranformation
        - dataframe with tranformations applied on both numerical and categorical features
    '''
    prep_df = dataframe.copy(deep=True)
    
    prep_df = prep_df.rename(columns={'age_group_category':'age_bucket','location_type':'location','duration_gold':'gold_duration','cl_sol_freq':'cl_sol'})
    cat_brand_rename = ['sun_freq','cl_freq', 'eye_freq', 'loyalty_freq', 'npr_freq',
    'hto_freq', 'prog_freq', 'online_excl_freq', 'lk_air_freq', 'kids_freq','vc_freq', 'jj_freq', 'od_freq', 'huslr_freq']
    rename_dict = {col:col.split('_')[0]  for col in cat_brand_rename}
    prep_df = prep_df.rename(columns=rename_dict)
    prep_df = prep_df.rename(columns={'non_powered_sun_freq':'non_powered_sun','powered_sun_freq':'powered_sun'})

    prep_df['gold_duration'] = np.where(prep_df['gold_duration'].isin(['2 year','1 year','Expired'])==False,'More than 2 years',prep_df['gold_duration'])
    prep_df['gold_duration'] = np.where((prep_df['gold_start_date'].isnull()) & (prep_df['gold_duration']=='Expired'),'Not Bought',prep_df['gold_duration'])
    prep_df['frequency'] = np.where(prep_df['frequency']<=3,'freq_1_to_3','freq_above_3')
    
    prep_df['acquisition_date']= pd.to_datetime(prep_df['acquisition_product'],errors='coerce')
    prep_df['today'] = pd.to_datetime(datetime.date.today())
    prep_df['customer_age'] = (prep_df['today'] - prep_df['acquisition_date']).dt.days/365
    
    
    cat_col = ['frequency','recency_bucket','age_bucket','gender','location','gold_duration']
    num_col = ['customer_phone','customer_category','customer_age','gross_orders','atv','basket_size','return_perc','eye', 'sun', 'npr', 'cl', 'cl_sol', 
               'branded','non_powered_sun','powered_sun','loyalty', 'hto', 'prog', 'vc',
               'jj', 'lk', 'kids', 'huslr','online_freq','online_freq_30', 'session_cnt', 'session_cnt_30', 'offline_freq',
               'offline_freq_30','qms_count', 'qms_count_30', 'eye_count',
               'eye_count_12m', 'eye_count_30']

    cat_brand_col = ['eye', 'sun', 'branded', 'non_powered_sun', 'powered_sun', 'npr', 'cl', 'cl_sol', 'loyalty', 'hto', 'prog', 'vc',
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
    
    # numerical features not tranformed and categorical features one hot encoded
    not_numerical_tranformed_cat_ohe = pd.concat([data_[num_col],one_hot_encode],axis=1)

    return data_,not_numerical_tranformed_cat_ohe,cluster_data

def cosineSimilarity(features_centre,scaled,num_col):

    """ A function to calculate cosine simialrity """

    weighted_compare = pd.DataFrame(np.multiply(np.array(features_centre[['weights']].T),np.array(scaled.iloc[:,1:])),columns=num_col)
    weighted_compare['compare_norm'] = np.sqrt(np.sum(weighted_compare**2,axis=1))
    
    thetax = np.matmul(features_centre[['weighted_mean']].T,weighted_compare.iloc[:,:-1].T).T
    # thetax = np.matmul(weighted_compare.iloc[:,:-1],features_centre[['weighted_mean']])
    thetax = thetax.rename(columns={'weighted_mean':'dot_prod'})
    
    norm_multiply = features_centre['centre_norm'][0]*weighted_compare['compare_norm']
    
    thetax = pd.merge(thetax,norm_multiply,left_index=True,right_index=True)
    thetax = thetax.rename(columns={'compare_norm':'prod_norm'})
    thetax['cosine']  = thetax['dot_prod']/thetax['prod_norm']

    return thetax

def jsDivergences(data_post_cluster,prod_cat):

    """
    target - category buyers
    compare - category non buyers

    Calculates weights for numerical features through JS divergences which will give us the measure 
    how two distributions (target and compare) are different. Higher the magnitude of a feature the 
    higher is its importance in distinguishing target and compare i.e buyers and non buyers.

    This function processes the computed JS divergence scores to derive feature importance weights, 
    which are subsequently used to determine the weighted mean of target features. The function performs 
    the following steps:

    1. Normalize JS Divergences:  
       - Computes the relative contribution (weights) of each numerical feature by normalizing the JS divergence values.

    2. Sort Features by Importance:
       - Sorts features in descending order of their JS divergence values to prioritize highly divergent features.

    3. Define Target Feature Center:
       - Computes the mean value of each numerical feature within the target group.
       - Merges the mean values with the computed feature weights.
       - Calculates the weighted mean and determines the center norm used for cosine similarity calculations.

    4. Calculating the cosine similarity between calculated target centre and for each customer present in compare group 
       to generate the similarity score

    Parameters:
    -----------
    data_post_cluster : pandas.DataFrame
        A DataFrame containing numeircal and categorical features both. Categorica features are one hot encoded.

    prod_cat : string
        A product category on which similarity score needs to be assigned.

    Returns:
    --------
    pandas.DataFrame
        features_centre - A DataFrame containing the feature mean values, calculated weights, weighted means, 
        and the computed center norm for cosine similarity.

        final_res - A DataFrame containing similarity scores for customers
        
    """
    
    filtered = data_post_cluster.copy(deep=True)
    
    target = filtered.loc[(filtered[prod_cat]!=0)].reset_index(drop=True)
    compare = filtered.loc[filtered['customer_phone'].isin(target['customer_phone'])==False].reset_index(drop=True)
    
    if prod_cat in ['cl','npr','jj','kids','sun','branded']:
        # features used in calulcating similarity scores
        num_col = ['customer_age', 'gross_orders','atv', 'basket_size', 'return_perc', 'eye', 'sun','cl','npr','branded',
                   'loyalty', 'vc', 'jj', 'lk', 'kids', 'huslr','online_freq',
                   'offline_freq','session_cnt','qms_count','eye_count', 'eye_count_12m']
    else:
        num_col = ['customer_age', 'gross_orders','atv', 'basket_size', 'return_perc', 'eye','branded','non_powered_sun',
                   'powered_sun','cl','npr','loyalty', 'vc', 'jj', 'lk', 'kids', 'huslr','online_freq',
                   'offline_freq','session_cnt','qms_count','eye_count', 'eye_count_12m']

    num_col.remove(prod_cat)
    
    cat_col = ['online_freq_30','offline_freq_30','qms_count_30','session_cnt_30','eye_count_30',
          'frequency_freq_1_to_3','frequency_freq_above_3','recency_bucket_0 to 12 Months','recency_bucket_24+ Months', 'recency_bucket_12 to 24 Months',
          'age_bucket_10-20', 'age_bucket_18-30','age_bucket_30-40', 'age_bucket_40-60', 'age_bucket_Above 60',
          'age_bucket_Other', 'age_bucket_Under 10', 'gender_Female','gender_Kid', 'gender_Male', 'gender_Online', 'location_Metro',
          'location_Non-Metro', 'gold_duration_1 year', 'gold_duration_2 year','gold_duration_Expired', 
          'gold_duration_More than 2 years','gold_duration_Not Bought']
    
    ## calculating js divergences
    eps = 1e-10
    num_div = {}
    error_col = []
    for col in num_col:
        try:
            plt.figure()
            dist_1 = sns.displot(target[col]**0.33,kind='kde')
            ax = dist_1.ax
            line = ax.get_lines()[0]
            dist_1_ = line.get_ydata()

            # dist_1 = sns.displot(target[col]**0.33,kind='kde').get_lines()[0]
            # dist_1_ = dist_1.get_ydata()

            p = dist_1_/dist_1_.sum() + eps

            # compare
            plt.figure()
            dist_2 = sns.displot(compare[col]**0.33,kind='kde')
            ax = dist_2.ax
            line = ax.get_lines()[0]
            dist_2_ = line.get_ydata()
            
            # dist_2 = sns.displot(compare[col]**0.33,kind='kde').get_lines()[0]
            # dist_2_ = dist_2.get_ydata()

            q = dist_2_/dist_2_.sum() + eps

            # mixture
            m = (p+q)/2

            kl_pm = np.sum(rel_entr(p,m))
            kl_qm = np.sum(rel_entr(q,m))

            js_div = 0.5*(kl_pm) + 0.5*(kl_qm)
            print(f'js div {col}:',js_div)
            num_div[col] = [js_div]
            
        except Exception as e:
            error_col.append(col)
            

    num_div_df = pd.DataFrame(num_div).T
    num_div_df = num_div_df.rename(columns={0:'js_div'})
    print(num_div_df)
    num_div_df['weights'] = num_div_df['js_div']/num_div_df['js_div'].sum()
    num_div_df = num_div_df.sort_values('js_div',ascending=False)
    num_div_df['segment'] = filtered['customer_category'].unique()[0]
    
    ## calculating similarity distances
    
    num_col = [col for col in num_col if col not in error_col]
    target_num = target[['customer_phone']+num_col]
    compare_num = compare[['customer_phone']+num_col]
    
    compare_scaled = compare_num.copy(deep=True)
    target_scaled = target_num.copy(deep=True)

    for col in num_col[:]:
        
        ## tranformation
        target_scaled[col] = target_scaled[col]**0.33
        compare_scaled[col] = compare_scaled[col]**0.33

        ## min max normalisation
        target_scaled[col] = (target_scaled[col] - target_scaled[col].min())/(target_scaled[col].max() - target_scaled[col].min())
        compare_scaled[col] = (compare_scaled[col] - compare_scaled[col].min())/(compare_scaled[col].max() - compare_scaled[col].min())
        
    features_centre = pd.DataFrame(target_scaled.iloc[:,1:].mean())
    features_centre = features_centre.rename(columns={0:'mean'})
    features_centre = pd.merge(features_centre,num_div_df,left_index=True,right_index=True)
    features_centre['weighted_mean'] = features_centre['mean']*features_centre['weights']
    features_centre['centre_norm'] = np.sqrt(np.sum(features_centre['weighted_mean']**2))
    
    thetax_compare = cosineSimilarity(features_centre,compare_scaled,num_col)
    thetax_target = cosineSimilarity(features_centre,target_scaled,num_col)
    
    final_res_compare = pd.merge(compare_num,thetax_compare,left_index=True,right_index=True)
    final_res_compare = pd.merge(compare,final_res_compare[['customer_phone','cosine']],how="inner",on=['customer_phone'])
    final_res_compare['group'] = 'compare'
    
    final_res_target = pd.merge(target_num,thetax_target,left_index=True,right_index=True)
    final_res_target = pd.merge(target,final_res_target[['customer_phone','cosine']],how="inner",on=['customer_phone'])
    final_res_target['group'] = 'target'
    
    final_res = pd.concat([final_res_compare,final_res_target],axis=0,ignore_index=True)

    return final_res,features_centre

if __name__ == '__main__':

    start_time = time.time()
    date_today = str(date.today())
    print('Date Today:',date_today)

    # query = f"""select * from customercharter.customer_charter_lifetime where date(ingested_at)='{date_today}' 
    #             and customer_category not in ('Unknown Category','Only Gold') """
    # data = mySQLRead(query)
    
    ## Query for fetching data
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
            lk_air_freq, kids_freq, vc_freq, jj_freq, od_freq, branded_freq,
            powered_sun_freq, non_powered_sun_freq,
            huslr_freq, online_freq, online_freq_30, offline_freq,
            offline_freq_30, atv, asp, basket_size,
            session_cnt, session_cnt_30, gold_start_date, gold_expiry_date,
            status, duration_gold, qms_count, qms_count_30, eye_count,
            eye_count_30, eye_count_12m, frequency, recency,
            recency_bucket,age_group_category,customer_category, return_perc,
            ingested_at from customercharter.customer_charter_lifetime where date(ingested_at)='{date_today}' 
            and customer_category not in ('Unknown Category','Only Gold')
            """
    # customer_category not in ('Unknown Category','Only Gold') 
    data = mySQLRead(query)
    print('Query Run Successful:',data.head(10))

    centres = pd.DataFrame()
    similarities = pd.DataFrame()

    ## similarity scores are calculated for each customer category
    for prod_cat in ['cl','npr','kids','sun','jj','branded','powered_sun','non_powered_sun']:
        print('Product Category:',prod_cat)
        for cat in data['customer_category'].unique()[:]:
            print('Customer Category:',cat)
            temp = data.loc[data['customer_category']==cat].reset_index(drop=True)
            data_original,data_post_cluster,cluster_df = dataPrep(temp)

            ### similarity part 
            data_post_cluster = data_post_cluster.fillna(0)
            col_to_remove = [col for col in data.columns if col.find("Unknown")!=-1]

            similarity_score_tagged,features_centre = jsDivergences(data_post_cluster,prod_cat)
            similarity_score_tagged['product_category'] = prod_cat
            features_centre['product_category'] = prod_cat

            similarities = pd.concat([similarities,similarity_score_tagged],axis=0,ignore_index=True)
            centres = pd.concat([centres,features_centre],axis=0,ignore_index=False)
        
    prod = []
    for prod_cat in similarities['product_category'].unique():
        cat_df = similarities.loc[similarities['product_category']==prod_cat]
        cat_df.rename(columns={'group':'group_'+prod_cat,'cosine':'similarity_score_'+prod_cat},inplace=True)
        col = ['customer_phone', 'customer_category', 'group_'+prod_cat,'similarity_score_'+prod_cat]
        cat_df = cat_df[col]
        prod =  prod + [cat_df]
    
    ## final dataframe csv to be uploaded
    prod = reduce(lambda df1,df2:pd.merge(df1,df2,how="inner",on=['customer_phone','customer_category']),prod)
    print('Total Customers Scored:',prod.shape[0])
    print('Similarity Scores DataFrame:',prod.head())
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time Elapsed in Minutes:",time_elapsed/60)
    