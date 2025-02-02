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
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,power_transform
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pickle
# from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import datetime
import datetime as dt
from datetime import timedelta
import time
import warnings
from functools import reduce
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',30)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def getData(start_date,end_date) -> pd.DataFrame:

    """Query for the retrieval of raw data"""

    query = f"""with Customer_base as
                (
                        select rom.customer_phone,rom.increment_id,b.revenue_without_vat,date(rom.created_at) as created_at,item_id,
                        CUSTOMER_ID,b.revenue_with_vat,rom.order_type,
                        case 
                        when rom.increment_id in (Select exchange_increment_id from wh_operations.customer_order_returns cor  where exchange_increment_id is not null)
                        then 1 else 0 end as exchange_flag
                        from flat_reports_adoc.rs_order_master rom
                        join flat_reports_adoc.rs_order_item  b on rom.increment_id=b.increment_id
                        where date(rom.created_at)>'{start_date}' and date(rom.created_at)<='{end_date}'
                        and rom.order_type not in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service')
                        and lower(rom.inv_status_type) not in ('pre')     
                        and rom.exclude=1 and b.classification not in ('loyalty_services')
                        and country='IN'                     
                )
                , exchange_ids as
                (Select display_order_code,item_id from wh_operations.customer_order_returns cor  where exchange_increment_id is not null)

                ,return_metrics AS
                (      
                        Select distinct return_id,return_item_id, increment_id,refund_amount,qc_status_type
                        FROM 
                        (        
                                select distinct cor.return_id,cor.display_order_code as increment_id,
                                case when co.INCREMENT_id is not null then null else cor.item_id end as return_item_id,
                                case when co.INCREMENT_id is not null then  0 else rf.revenue_without_vat end as refund_amount,
                                case
                                when lower(inventory_type) like '%bad%' then 'Bad Inventory'
                                when lower(inventory_type) like '%good%' then 'Good Inventory'
                                else 'other' end as qc_status_type,
                                row_number() over(partition by item_id order by return_create_datetime desc) AS RNK 
                                from wh_operations.customer_order_returns cor 
                                left join flat_reports_adoc.rs_order_master rf 
                                on cor.display_order_code = rf.INCREMENT_id
                                left join inventory.canceled_orders co  on  co.type in ('cancel') and co.increment_id = rf.INCREMENT_id 
                                where classification <> 'prescription_lens'
                                and cor.return_status not in ('return_rejected','customer_cancelled','cancelled','return_rejected_handover_pending')
                                and display_order_code not in (select distinct rom.increment_id from flat_reports_adoc.rs_order_master rom 
                                where rom.order_type in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service'))

                        )Q WHERE RNK=1        
                )
                
                ,tickets_cust_phone AS
                (Select case_associated_customer_id,sum(ticketing_touch_point) as ticket_touch_phone from  customer_service.sprinklr_tickets b
                group by 1)

                , tickets_inc_id as
                (Select ticket_order_number_case as increment_id,sum(ticketing_touch_point) as ticket_touch_id from  customer_service.sprinklr_tickets b
                group by 1)
                
                ,Latest_csat_YES_NO AS
                (        
                        select customer_phone,CSAT_YES_NO as Latest_csat
                        from 
                        (
                                select a.customer_phone,a.increment_id,was_our_agent_from_customer_delight_able_to_resolve_your_concerns AS CSAT_YES_NO,
                                response_dt,        row_number() over(partition by customer_phone order by response_dt desc) AS RNK
                                from flat_reports_adoc.rs_order_master a
                                join
                                (
                                        select a.case_id,was_our_agent_from_customer_delight_able_to_resolve_your_concerns ,ticket_order_number_case ,
                                        date(survey_response_time) as response_dt 
                                        from customer_service.csat a
                                        join customer_service.sprinklr_tickets b
                                        on a.case_id=b.case_id
                                        where lower(was_our_agent_from_customer_delight_able_to_resolve_your_concerns) in ('yes','no')
                                ) b
                                on a.increment_id=b.ticket_order_number_case
                        ) where RNK=1
                ),
                MEAN_CSAT AS
                (
                        select customer_phone,
                        ROUND((COUNT(CASE WHEN LOWER(CSAT_YES_NO) = 'yes' THEN 1 END)::numeric / COUNT(CASE WHEN LOWER(CSAT_YES_NO) IN ('yes', 'no') THEN 1 END)::numeric), 4) AS AVG_CSAT
                        from 
                        (
                                select a.customer_phone,a.increment_id,was_our_agent_from_customer_delight_able_to_resolve_your_concerns AS CSAT_YES_NO,
                                response_dt,        row_number() over(partition by customer_phone order by response_dt desc) AS RNK
                                from flat_reports_adoc.rs_order_master a
                                join
                                (
                                        select a.case_id,was_our_agent_from_customer_delight_able_to_resolve_your_concerns ,ticket_order_number_case ,
                                        date(survey_response_time) as response_dt 
                                        from customer_service.csat a
                                        join customer_service.sprinklr_tickets b
                                        on a.case_id=b.case_id
                                        where lower(was_our_agent_from_customer_delight_able_to_resolve_your_concerns) in ('yes','no')
                                ) b
                                on a.increment_id=b.ticket_order_number_case
                        ) Q
                        GROUP BY 1
                )
                ,LATEST_NPS AS
                (                
                        SELECT         customer_phone,nps_score as RECENT_NPS
                        FROM         
                        (
                                Select customer_phone,a.increment_id,nps_score,response_date,
                                row_number() over(partition by customer_phone order by response_date desc) AS RNK
                                FROM
                                (
                                        select rom.customer_phone,rom.increment_id,b.revenue_without_vat,b.discount_amount,date(rom.created_at) as created_at,item_id,
                                        CUSTOMER_ID        
                                        from flat_reports_adoc.rs_order_master rom
                                        join flat_reports_adoc.rs_order_item  b
                                        on rom.increment_id=b.increment_id
                                        where date(rom.created_at)>'{start_date}'  and date(rom.created_at)<='{end_date}'
                                        and rom.order_type not in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service')
                                        and lower(rom.inv_status_type) not in ('pre') and rom.order_state <> 'cancelled' 
                                        and rom.exclude=1 and rom.order_components not in ('loyalty_services')
                                        AND country='IN'
                                        and rom.increment_id not in (Select exchange_increment_id from wh_operations.customer_order_returns cor  where exchange_increment_id is not null)                                
                                )a
                                join 
                                (        
                                        select a.increment_id, a.nps_tag, a.response_date,nps_score
                                        from
                                        (
                                                select distinct lf.increment_id, 
                                                coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale) as nps_score,
                                                case 
                                                when coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)>=0 
                                                and coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)<=6 then 'Detractor' 

                                                when coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)>=7 
                                                and coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)<=8 then 'Passive' 

                                                when coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)>=9 
                                                and coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale)<=10 then 'Promoter' end as nps_tag,

                                                coalesce(c.date,n.date) as response_date,
                                                row_number() over(partition by lf.increment_id order by coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale) desc)
                                                as row_num
                                                from 
                                                (
                                                        select DISTINCT rom.increment_id
                                                        from flat_reports_adoc.rs_order_master rom
                                                        join flat_reports_adoc.rs_order_item  b
                                                        on rom.increment_id=b.increment_id
                                                        where date(rom.created_at)>'{start_date}' and date(rom.created_at)<='{end_date}'
                                                         and rom.order_type not in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service')
                                                        and lower(rom.inv_status_type) not in ('pre') and rom.order_state <> 'cancelled' 
                                                        and rom.exclude=1 and rom.order_components not in ('loyalty_services')                                
                                                )lf 
                                                left join flat_reports_adoc.cloud_cherry_apix c on c.order_id=cast(lf.increment_id as varchar)
                                                left join scm.nps_response_summary n on n.order_id=lf.increment_id
                                                where coalesce(c.question_recommend_nps_scale,n.question_recommend_nps_scale) is not null
                                        )a
                                                where a.row_num=1
                                )b
                                on a.increment_id=b.increment_id
                        )Q WHERE RNK=1

                )
                ,cancelled AS
                (
                        Select DISTINCT a.increment_id from 
                        (
                                (Select increment_id from  inventory.canceled_orders c  where  c.type in ('cancel') )A 
                                JOIN
                                (
                                        select DISTINCT rom.increment_id        
                                        from flat_reports_adoc.rs_order_master rom
                                        join flat_reports_adoc.rs_order_item  b
                                        on rom.increment_id=b.increment_id
                                        where date(rom.created_at)>'{start_date}' and date(rom.created_at)<='{end_date}'
                                                                and rom.order_type not in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service')
                                        and rom.exclude=1 and rom.order_components not in ('loyalty_services')
                                        AND country='IN'
                                        and rom.increment_id not in (Select exchange_increment_id from wh_operations.customer_order_returns cor  where exchange_increment_id is not null)
                                )B
                                ON A.increment_id=B.increment_id
                        )
                )

                ,MERGE_1 AS
                (
                Select A.customer_phone,RECENT_NPS,I.AVG_CSAT,max(date(a.created_at)) as last_txn_date,
                count(DISTINCT a.increment_id) as total_orders,
                count(distinct(case when lower(a.order_type) in ('online','jj_online') then increment_id else null end)) as online_orders,
                count(distinct(case when lower(a.order_type) not in ('online','jj_online') then increment_id else null end)) as offline_orders,
                sum(a.revenue_without_vat) as total_sales_wo_vat,sum(a.revenue_with_vat) as total_sales_with_vat,
                count(a.item_id) as total_items,count(distinct a.item_id) as total_unique_items

                from customer_base a
                left join MEAN_CSAT I on a.customer_phone=I.customer_phone
                left join LATEST_NPS f on a.customer_phone=f.customer_phone
                where exchange_flag=0
                GROUP BY 1,2,3
                )
                ,MERGE_2 AS
                (
                Select distinct A.customer_phone,
                COUNT (DISTINCT CASE WHEN C.INCREMENT_id IS NOT NULL and exchange_flag=0 THEN return_item_id END) AS RETURN_ID_ORDERS ,
                SUM(case when exchange_flag=0 then C.refund_amount end) AS REFUND_AMOUNT,
                COUNT (DISTINCT CASE WHEN C.qc_status_type = 'Bad Inventory' and exchange_flag=0 THEN return_item_id END) AS RETURN_QC_FAIL,
                COUNT (DISTINCT CASE WHEN C.INCREMENT_id IS NOT NULL AND C.refund_amount IS NOT NULL and exchange_flag=0 THEN return_item_id END) AS REFUND_ITEM_IDS,
                count(distinct a.item_id) as total_unique_return_items_with_exchange,
                COUNT (DISTINCT CASE WHEN C.qc_status_type = 'Bad Inventory' THEN return_item_id END) AS RETURN_QC_FAIL_With_Exchange,
                COUNT (DISTINCT CASE WHEN C.INCREMENT_id IS NOT NULL AND C.refund_amount IS NOT NULL THEN return_item_id END) AS REFUND_ITEM_IDS_with_exchange

                from customer_base a join return_metrics c 
                on a.INCREMENT_id=c.INCREMENT_id and a.item_id=c.return_item_id
                GROUP BY 1

                )
                ,MERGE_3 AS
                (
                select cont.customer_phone,
                (case 
                when cont.ticket_touch_phone is null then cont.ticket_touch_inc_id
                when cont.ticket_touch_inc_id is null then cont.ticket_touch_phone else cont.ticket_touch_inc_id end) as ticket_contacts
                from 
                (Select cb.customer_phone,t.ticket_touch_phone,sum(t_id.ticket_touch_id) as ticket_touch_inc_id
                from customer_base cb
                left join tickets_cust_phone t on cb.customer_phone=t.case_associated_customer_id
                left join tickets_inc_id t_id on cb.increment_id = t_id.increment_id
                where exchange_flag=0
                group by 1,2) cont
                )
                ,MERGE_4 AS
                (

                Select distinct A.customer_phone,
                COUNT (DISTINCT CASE WHEN G.INCREMENT_id IS NOT NULL THEN G.increment_id END) AS CANCELLED_ORDERS ,
                COUNT (DISTINCT CASE WHEN G.INCREMENT_id IS NOT NULL THEN a.item_id END) AS CANCELLED_ITEMS ,
                SUM(CASE WHEN G.INCREMENT_id IS NOT NULL THEN a.revenue_without_vat END) AS CANCELLED_AMT_wo_VAT,
                SUM(CASE WHEN G.INCREMENT_id IS NOT NULL THEN a.revenue_with_vat END) AS CANCELLED_AMT_WITH_VAT
                from customer_base a
                join cancelled g on a.increment_id=g.increment_id
                where exchange_flag=0
                GROUP BY 1

                )
                ,MERGE_5 AS
                (

                Select distinct A.customer_phone,
                sum((gnw.grand_total + gnw.shipping_charges  + gnw.tax_collected ) - gnw.item_total_after_discount)  as total_discount,
                SUM(gnw.bogo_discount) AS bogo_discount,
                SUM(gnw.sc_discount) as store_credit_discount,
                SUM(gnw.gv_discount) as gv_discount,
                SUM(gnw.wallet_discount) as wallet_discount,
                SUM(gnw.wallet_plus_discount) as wallet_plus_discount,
                sum(auto_discount) as auto_discount
                from customer_base a
                join flat_reports_adoc.gross_net_walk gnw on a.item_id=gnw.item_id
                where exchange_flag=0
                GROUP BY 1

                )
                ,MERGE_6 AS
                (
                Select distinct A.customer_phone,count(distinct b.item_id ) as exchange_item_ids
                from customer_base a
                join exchange_ids b  on a.increment_id=b.display_order_code and a.item_id=b.item_id
                where exchange_flag=0
                GROUP BY 1
                )
                SELECT cast(MERGE_1.customer_phone as text),RECENT_NPS,AVG_CSAT,total_orders,online_orders,offline_orders,total_sales_wo_vat,total_sales_with_vat,last_txn_date,total_items, 
                total_unique_items,RETURN_ID_ORDERS,REFUND_AMOUNT,REFUND_ITEM_IDS,RETURN_QC_FAIL_With_Exchange,CANCELLED_ORDERS,
                CANCELLED_AMT_WITH_VAT,CANCELLED_AMT_wo_VAT,CANCELLED_ITEMS,total_discount,bogo_discount,store_credit_discount,gv_discount,auto_discount,wallet_discount,wallet_plus_discount,
                exchange_item_ids,total_unique_return_items_with_exchange,REFUND_ITEM_IDS_with_exchange,ticket_contacts
                FROM MERGE_1  
                LEFT JOIN  MERGE_2 ON MERGE_1.customer_phone=MERGE_2.customer_phone
                LEFT JOIN  MERGE_3 ON MERGE_1.customer_phone=MERGE_3.customer_phone
                LEFT JOIN  MERGE_4 ON MERGE_1.customer_phone=MERGE_4.customer_phone
                LEFT JOIN  MERGE_5 ON MERGE_1.customer_phone=MERGE_5.customer_phone
                LEFT JOIN  MERGE_6 ON MERGE_1.customer_phone=MERGE_6.customer_phone
                """
    data = mySQLRead(query)
    data = data.loc[data['customer_phone'].isnull()==False]
#     data['customer_phone'] = data['customer_phone'].astype(str)
    
    query = f"""select cast(rom.customer_phone as text),rom.pincode,rfmt.city as franchise_city,rfmt2.city as pincode_city,
           case when  coalesce(rfmt.metropolitan, rfmt2.metropolitan)  = 'Yes' then 'Metro' else 'Non-Metro' end as location_type
           from flat_reports_adoc.rs_order_master rom
           left join flat_reports_adoc.rs_franchise_master_test rfmt on rom.franchise_id = rfmt.franchise_id
           left join flat_reports_adoc.rs_franchise_master_test rfmt2 on rom.pincode  = rfmt2.pincode 
           where date(rom.created_at)>'{start_date}' and date(rom.created_at)<='{end_date}'
           and rom.country='IN' and (rfmt.city is not null or rfmt2.city is not null)
           group by 1,2,3,4,5 """
    geo_df = mySQLRead(query)
    geo_df = geo_df.drop_duplicates('customer_phone').reset_index(drop=True)
#     geo_df['customer_phone'] = geo_df['customer_phone'].map(str)
    
    data = pd.merge(data,geo_df,how="left",on=['customer_phone'])

    return data


def variables(data:pd.DataFrame) -> pd.DataFrame:

    """From the raw data this functions creates new features to be used for defining the customer
       behaviour from clustering"""

    data['recent_nps'] = np.where(data['recent_nps'].isnull(),"Unknown",
                        np.where(data['recent_nps']>=9,'Promotor',
                        np.where(data['recent_nps']>=7,"Passive","Detractor")))

    # data['recency'] = (datetime.date(2024,2,1) - pd.to_datetime(data['last_txn_date']).dt.date)
    # data['recency'] = data['recency'].map(lambda x : x.days)

    data['order_behavior'] = np.where(data['total_orders']==1,0,1)
    data['channel'] = np.where(data['online_orders']>0,1,0)
    data['location_type'] = data['location_type'].fillna('Non-Metro')
    data['satisfaction'] = np.where(data['recent_nps'].isin(['Promotor','Passive','Unknown'])==True,'Non-Detractor','Detractor')

    null_col = ['total_sales_wo_vat','total_sales_with_vat','total_discount', 'bogo_discount',
                'store_credit_discount', 'gv_discount', 'auto_discount','wallet_discount', 'wallet_plus_discount',
                'refund_item_ids','return_id_orders','refund_amount','exchange_item_ids','ticket_contacts','cancelled_items',
                'cancelled_orders','cancelled_amt_with_vat','cancelled_amt_wo_vat','avg_csat','total_unique_return_items_with_exchange','return_qc_fail_with_exchange']
    for col in null_col:
        data[col] = data[col].fillna(0)

    data['total_sales_with_vat'] = np.where(data['total_sales_with_vat']<0,0,data['total_sales_with_vat'])
    data['total_sales_wo_vat'] = np.where(data['total_sales_wo_vat']<0,0,data['total_sales_wo_vat'])
    
    data['net_total_discount'] = data['total_discount'] - data['bogo_discount']
    
    # imputing total discount with where total discount is less than bogo discount
    data['net_total_discount'] = np.where(data['net_total_discount']<0,0,data['net_total_discount'])

    # data['aov'] = round(data['total_sales']/data['total_orders'],2)
    data['asp'] = round((data['total_sales_wo_vat']-data['refund_amount']-data['cancelled_amt_wo_vat'])/(data['total_items']-data['refund_item_ids']-data['cancelled_items']))

    # replacing negative asp with 0
    data['asp'] = np.where(data['asp']<0,0,data['asp'])


    data['refund_per_sales'] = round(data['refund_amount']/data['total_sales_wo_vat'],2).fillna(0)
    data['return_items_frac'] = round(data['return_id_orders']/data['total_items'],2)
    data['contacts_per_order'] = round(data['ticket_contacts']/data['total_orders'],2) 
    data['qc_fail_per_return'] = round(data['return_qc_fail_with_exchange']/data['total_unique_return_items_with_exchange'],2).fillna(0)
    data['items_per_order'] = round(data['total_items']/data['total_orders'],2)  
    data['cancelled_orders_frac'] = round(data['cancelled_orders']/data['total_orders'],2)
    data['exchange_items_frac'] = round(data['exchange_item_ids']/data['total_items'],2)

    # combining return and refund

    # data['exc_refund_cancelled_frac'] = round((data['return_id_orders'] + data['exchange_item_ids'] + data['cancelled_items'])/data['total_items'],2)

    ## removing rows with zero discount and zero sales for now
    # total_sales zero and total_discount also zero -> but order was there
    # data = data.iloc[data.index.isin(data.loc[(data['total_discount']==0) & (data['total_sales_wo_vat']==0)].index)==False]

    ## removing rows with negative sales
    # data = data.loc[(data['total_sales_with_vat']>=0) & (data['total_sales_wo_vat']>=0)].reset_index(drop=True)
    
    data['discount_per_sales'] = round(data['net_total_discount']/data['total_sales_with_vat'],2) 

    ## removing rows with infinite values for now 
    data['discount_per_sales'] = data['discount_per_sales'].replace(np.inf,1)
    data = data.replace(np.inf,np.nan).fillna(0)
    
    return data


def splitData(data:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    np.random.seed(2)
    n = data.shape[0]
    frac = 0.7
    shuffle_index = np.random.permutation(data.index)
    train_index = shuffle_index[:int(frac*n)]
    test_index = shuffle_index[int(frac*n):]

    # splitting into train and test
    train_data = data.iloc[train_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)
    
    return train_data,test_data


def dataPrep(data:pd.DataFrame) -> pd.DataFrame:

    """Function performs numerical and categorical feature transformation and
       reuturns the transformed dataframe for clustering""" 

    num_col = ['asp', 'return_items_frac', 'contacts_per_order', 'qc_fail_per_return',
               'discount_per_sales', 'cancelled_orders_frac', 'exchange_items_frac', 'items_per_order', 'avg_csat']
    numerical = data[num_col]
    
    # cube transformation of right skewed numerical features
    numerical_cube_trans = numerical.apply(lambda x:np.sign(x)*(abs(x)**0.33))
    
    # outlier clipping of transformed features
    for i in numerical.columns:
        p_01 = numerical_cube_trans[i].quantile(0.01)
        p_99 = numerical_cube_trans[i].quantile(0.99)
        numerical_cube_trans[i] = numerical_cube_trans[i].clip(p_01,p_99)
    
    # categorical columns
    cat_col = ['recent_nps']
    categorical = data[cat_col]
    one_hot_encode = pd.get_dummies(categorical,columns=['recent_nps'],drop_first=False,dtype=int).iloc[:,:-1]
    
    # final data used for clustering
    cluster_data = pd.concat([numerical_cube_trans,one_hot_encode],axis=1)
    
    return cluster_data


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
    
    """Function performs PCA and also give insights related to feature importances across
       the principal compents dimensions """
    
    X = cluster_data.to_numpy()
    pca = PCA(n_components=6)
    pca.fit(X)
    pca_transformed = pca.transform(X)
    
    print("Explained Variance:",pca.explained_variance_ratio_)
    print("Singular Values:",pca.singular_values_)
    
    print("total_variance:",reduce(lambda a,b:a+b,pca.explained_variance_ratio_))
    sns.barplot(x=np.arange(1,7),y=np.cumsum(pca.explained_variance_ratio_))
    
    importance_df  =create_importance_dataframe(pca,cluster_data)

    print(importance_df)

    ## PC1 top 10 important features
    pc1_top_10_features = importance_df['PC1'].sort_values(ascending = False)[:20]
    print(), print(f'PC1 top 10 feautres are \n')
    print(pc1_top_10_features)

    ## PC2 top 10 important features
    pc2_top_10_features = importance_df['PC2'].sort_values(ascending = False)[:20]
    print(), print(f'PC2 top 10 feautres are \n')
    print(pc2_top_10_features)

    pc3_top_10_features = importance_df['PC3'].sort_values(ascending = False)[:20]
    print(), print(f'PC3 top 10 feautres are \n')
    print(pc3_top_10_features)
    
    pca_df = pd.DataFrame(pca_transformed,columns=['col1','col2','col3','col4','col5','col6'])
    
    return pca_df

def kmeansClustering(data:pd.DataFrame):

    """Function performs Kmeans Clusteirng"""

    model = KMeans(n_clusters = 3, init = "k-means++")
    label = model.fit_predict(data)
    
    return label


def clusterPopMap(data:pd.DataFrame) -> pd.DataFrame:
    
    """
    This function generates a cluster-to-population heatmap, providing a summary of clustering results. 
    It calculates the ratio of the cluster mean to the overall population mean for each feature across 
    all identified clusters. 
    The resulting heatmap offers a high-level overview, highlighting which features differ significantly 
    across clusters, helping to interpret and compare cluster characteristics effectively.

    """

    mean_sample = data.groupby(['labels']).agg('mean')
    mean_pop = data.mean().iloc[:-1]
    
    comb = pd.concat([mean_sample,pd.DataFrame(mean_pop,columns=['pop_mean']).T])

    for i in range(comb.shape[0]):
        comb.iloc[i] = comb.iloc[i]/comb.iloc[-1]
        
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(comb.iloc[:-1,:].T,cmap='RdYlGn_r',annot=True,annot_kws={"fontsize":15},cbar_kws={'label': 'factor'})
    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.figure.axes[-1].tick_params(labelsize=10)
    plt.xticks(fontsize=10,rotation=0)
    plt.yticks(fontsize=10,rotation=45)
    
    return comb


def randomForest(X_train:pd.DataFrame,y_train:pd.Series):

    """
    Classification Model is trained on the labelled customers from clutering 
    to classify new unseen custoemrs and to get the feature importances that would 
    be further used for scoring the customers.
    
    """
    
    rf = RandomForestClassifier(n_estimators=150,max_depth=5, min_samples_split=2,random_state=2)
    rf.fit(X_train,y_train)
    
    name = 'model/random_forest_recent.pkl'
    dt_model_pkl = open(name,'wb')
    pickle.dump(rf,dt_model_pkl)
    dt_model_pkl.close()
    
    avg_gini_importance = pd.DataFrame(data=rf.feature_importances_,index=list(X_train.columns),columns=['importance']).reset_index()
    avg_gini_importance = avg_gini_importance.sort_values('importance',ascending=False)
    avg_gini_importance.rename(columns={'index':'features'},inplace=True)

    fig, ax = plt.subplots(figsize=(60, 100))
    ax = sns.barplot(x="importance", y="features",data=avg_gini_importance)

    # grouped bars will have multiple containers
    for container in ax.containers:
        ax.bar_label(container,size=30)

    plt.xticks(fontsize=40,rotation=0)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    
    return rf,avg_gini_importance

def accuracyMatrix(model,X_train,X_test,y_train,y_test):
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    acc_test = accuracy_score(y_test,y_pred_test)
    acc_train = accuracy_score(y_train,y_pred_train)
    
    print("Testing Data Accuracy Decision Trees:",acc_test)
    print("Training Data Accuracy Decision Trees:",acc_train)
    
    arr = confusion_matrix(y_test, y_pred_test)

    total = np.sum(arr,axis=1)
    total = np.reshape(total,(-1,1))
    per = np.divide(arr,total)

    plt.figure(figsize=(15,10))
    # ax = sns.heatmap(per,cmap='Blues_r',annot=True,annot_kws={"fontsize":10},cbar_kws={'label': 'percent'})
    ax = sns.heatmap(arr,cmap='Blues_r',annot=True,annot_kws={"fontsize":10},cbar_kws={'label': 'number'},fmt='g')
    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.figure.axes[-1].tick_params(labelsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Predicted",fontsize=15)
    plt.ylabel("Actual",fontsize=15)
    
    return arr,y_pred_train,y_pred_test

def scoring(data:pd.DataFrame,weights:pd.DataFrame,p_01=0,p_99=0) -> pd.Series:
    
    """
    Function generates the scores of customers on the basis of their features and their importances
    between 0 and 100

    """
    scaled_df = data.copy()
    for col in scaled_df.columns[:-1]:
        factor =  list(weights.loc[weights['features']==col]['importance'])[0]
        scaled_df[col] = factor*scaled_df[col]
    
    scaled_df['pos'] = scaled_df['asp']
    scaled_df['neg'] = scaled_df['return_items_frac']+scaled_df['exchange_items_frac']+scaled_df['discount_per_sales']+scaled_df['cancelled_orders_frac']+scaled_df['qc_fail_per_return']+scaled_df['contacts_per_order']
    scaled_df['score'] = scaled_df['pos'] - scaled_df['neg']
    
    if (p_01==0) & (p_99==0):
        p_01 = scaled_df['score'].quantile(0.01)
        p_99 = scaled_df['score'].quantile(0.99)
    else:
        p_01 = p_01
        p_99 = p_99
    print("p_01:",p_01)
    print("p_99:",p_99)
    
    perc_df = pd.DataFrame(data=[p_01,p_99],index=['p_01','p_99'],columns=['value'])
    perc_df.to_csv('model/perc_df.csv')
    
    scaled_df['score_clip'] = scaled_df['score'].clip(p_01,p_99)
    
    # translating score in the range from 0 to 100
    
    new_min = 0
    new_max = 100

    old_min = scaled_df['score_clip'].min()
    old_max = scaled_df['score_clip'].max()

    # scaling factor
    a = (new_max - new_min)/(old_max - old_min)

    # offset
    b = new_min - (a*old_min)

    # transformed
    scaled_df['score_trans'] = a*scaled_df['score_clip'] + b
    
    return scaled_df['score_trans']


if __name__ == '__main__':
    
    start_time = time.time()
    
    ## clustering
    end_date = datetime.date.today() - timedelta(days=1)
    start_date = end_date - relativedelta(months=18)
#     start_date_cp = end_date - timedelta(days=4)
    
    profile_dump = getData(start_date,end_date)
    
    query = f"""select cast(customer_phone as text) as customer_phone_text,* from scm.customer_profiling"""
    initial = mySQLRead(query)
    initial = initial.drop(columns=['customer_phone'])
    initial.rename(columns={'customer_phone_text':'customer_phone'},inplace=True)
#     initial['customer_phone'] = initial['customer_phone'].map(str)
    
    variables_define = variables(profile_dump)
    
    present = initial.loc[initial['customer_phone'].isin(variables_define['customer_phone'])==True].reset_index(drop=True)
    print('Total Customers Relabelled:',present.shape[0])
    
    not_present = initial.loc[initial['customer_phone'].isin(variables_define['customer_phone'])==False].reset_index(drop=True)
    print('Total Customers Not Present:',not_present.shape[0])
    
    cluster_df = dataPrep(variables_define)
    dim_red_df = dimensionalityReduction(cluster_df)
    
    assign_labels = kmeansClustering(dim_red_df)
    cluster_df['labels'] = assign_labels
    variables_define['labels'] = assign_labels
    
    clus_char = clusterPopMap(cluster_df.loc[:,'asp':])
    
    ## classification
    keep_features = ['asp', 'return_items_frac', 'contacts_per_order', 'qc_fail_per_return','discount_per_sales', 'cancelled_orders_frac', 'exchange_items_frac','labels']
    classification_df = cluster_df[keep_features]

    features = classification_df.iloc[:,:-1]
    target = classification_df.iloc[:,-1]

    indices = np.arange(features.shape[0])
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, target, indices,stratify=target,test_size=0.30,shuffle=True)
    
    model,weights = randomForest(X_train,y_train)
    weights.to_csv('model/weights.csv',index=False)
    
    conf_matrix_df,model_labels_train,model_labels_test = accuracyMatrix(model,X_train,X_test,y_train,y_test)
    
    X_train['customer_profile'] = model_labels_train
    X_test['customer_profile'] = model_labels_test
    
    # scoring
    after_pred_df = pd.concat([X_train,X_test],axis=0)
    after_pred_df['customer_score'] = scoring(after_pred_df,weights)

    variables_define = pd.merge(variables_define,after_pred_df[['customer_profile','customer_score']],left_index=True,right_index=True)
    variables_define.drop(columns=['labels'],inplace=True)
    
    # fixing the customer behavior key [ 0:Best,2:Good,1:Worst]
    nature_df = variables_define.groupby(['customer_profile']).agg(median=('customer_score','median')).reset_index()
    nature_df['customer_profile_updated'] = np.where(nature_df['median']==nature_df['median'].max(),0,
                                                     np.where(nature_df['median']==nature_df['median'].min(),1,2))
    
    variables_define['customer_profile'] = variables_define['customer_profile'].map(nature_df['customer_profile_updated'])
    
#     query = """select * from scm.customer_profiling"""
#     initial = mySQLRead(query)
    
    present = initial.loc[initial['customer_phone'].isin(variables_define['customer_phone'])==True].reset_index(drop=True)
    print('Total Customers Relabelled:',present.shape[0])
    
    not_present = initial.loc[initial['customer_phone'].isin(variables_define['customer_phone'])==False].reset_index(drop=True)
    
    variables_define = pd.concat([variables_define,not_present.iloc[:,:-1]],axis=0,ignore_index=True)
#     variables_define.to_csv('Backup_Clustering/clustering_algorithm_15_April.csv',index=False)

    col = ['customer_phone','channel','location_type','satisfaction','customer_profile','customer_score']
    prod = variables_define[col]
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time Elapsed in Minutes:",time_elapsed/60)