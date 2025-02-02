import sys
sys.path.append('/home/moneshsharma/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn
from redshift_creds.database import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import pickle
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

def getData(start_date,end_date,l_30):

    """
    Retrieves relevant product data for the pricing recommendation project.

    This function queries the database to identify product IDs that meet specific criteria 
    for pricing recommendation. Products are considered if they belong to the 'Core', 
    'Continuity', or 'Sizing' categories and have at least 4 or 5 unique price points 
    within the given date range. The function performs the following steps:

    1. Volume Calculation:
       - Extracts product sales data within the specified date range and its other details 
         such as revenue (overall and last 30 days)
       - Filters products based on their 'core' classification.
       - Computes the volume contribution of each product to prioritize significant ones.

    2. Daily Sales Data:
       - Retrieves daily sales transactions for selected product IDs within the same period.
       - Collects item counts and special prices to analyze pricing trends over time.

    3. Cost Data Calculation:
       - Calculates the average cost of selected products over the last 30 days to get cost price
         of a product id.

    Parameters:
    -----------
    start_date : str
        The start date for data extraction in 'YYYY-MM-DD' format.

    end_date : str
        The end date for data extraction in 'YYYY-MM-DD' format.

    l_30 : str
        The date threshold for filtering recent sales in 'YYYY-MM-DD' format (last 30 days)

    Returns:
    --------
    vol : pandas.DataFrame
        A DataFrame containing product sales volume, number of unique price counts and filtered product IDs 
        ready for recommendation.

    daily_sales : pandas.DataFrame
        A DataFrame containing daily sales data, including order dates, product IDs, and sales volume.

    cost : pandas.DataFrame
        A DataFrame containing average product costs from financial records.
    """
    
    query = f"""select rpm.core,rpm.product_id,roi.brand,count(roi.item_id) as total_items,count(distinct(pas.special_price)) as count_unique_price,
           sum(roi.revenue_without_vat) as total_sales,sum(case when date(roi.created_at)>'{l_30}' then roi.revenue_without_vat else 0 end) as total_sales_l30,
           count(case when date(roi.created_at)>'{l_30}' then roi.item_id else null end) as total_items_l30
           from flat_reports_adoc.rs_order_item roi
           left join flat_reports_adoc.rs_order_master rom on roi.increment_id = rom.increment_id
           left join flat_reports_adoc.rs_products_master rpm on roi.product_id = rpm.product_id
           inner join merchandising.products_attribute_snapshot pas on pas.product_id=roi.product_id and date(pas.snapshot_date)=date(rom.created_at)
           where lower(rom.inv_status_type) not in ('pre','ppc')     
           and rom.exclude=1 and rom.country='IN' 
           and date(rom.created_at) between '{start_date}' and '{end_date}'
           and roi.classification in ('eyeframe') 
           and rom.order_type not in ('franchise_bulk','Marketplace','tbyp_fup','JJ_Bulk','Franchise_OTC','SIS','hto_fup','hto_service','tbyb_service')
           and rpm.core not in ('Discontinued')
           group by 1,2,3
           order by 4 desc"""
    vol = mySQLRead(query)
    vol['volume_contribution'] = (vol['total_items']/vol['total_items'].sum())*100
    vol = vol.loc[(vol['count_unique_price']==5) | (vol['count_unique_price']==4)].sort_values('total_items',ascending=False).reset_index(drop=True)
    vol = vol.loc[vol['core'].isin(['Core','Continuity','Sizing'])].reset_index(drop=True)
    # vol['sales_ratio'] = vol['total_sales']/vol['total_sales_l30']
    
    pid = vol['product_id'].unique()
    
    query = f"""select date(rom.created_at) as order_date,roi.product_id,pas.special_price,count(roi.item_id) as total_items
                from flat_reports_adoc.rs_order_item roi
                left join flat_reports_adoc.rs_order_master rom on roi.increment_id = rom.increment_id
                left join flat_reports_adoc.rs_products_master rpm on roi.product_id = rpm.product_id
                inner join merchandising.products_attribute_snapshot pas on pas.product_id=roi.product_id and date(pas.snapshot_date)=date(rom.created_at)
                where lower(rom.inv_status_type) not in ('pre','ppc')     
                and rom.exclude=1 and rom.country='IN' 
                and date(rom.created_at) between '{start_date}' and '{end_date}'
                and roi.product_id in {tuple(pid)}
                group by 1,2,3
                order by 2,1"""
    daily_sales = mySQLRead(query)
    
    query = f"""select sku_code as product_id,avg(cost) as avg_cost from
                (select sku_code,net_cogs_pl as cost from finance.nexs_sales_report where currency_code ='INR' and remark='Sale' and 
                sku_code in {tuple(pid)} and date(date)>'{l_30}' 
                union all
                select sku_code,net_cogs_pl as cost from finance.sales_report where currency_code ='INR' and remark='Sale' and 
                sku_code in {tuple(pid)} and date(date)>'{l_30}'
                )
                group by 1"""
    cost = mySQLRead(query)
    
    return vol,daily_sales,cost

# best parameters calculation for every product id through cross_validation
def getBaseSales(details,data):

    """
    Calculates the base sales trend for each product ID using the Prophet time series forecasting model. 
    The function performs hyperparameter tuning using a cross-validation approach to determine the best 
    parameters that minimize RMSE (Root Mean Squared Error).

    Parameters:
    - details (DataFrame): Contains aggregated product-level information, including total sales count.
                           Only products with total_items > 8800 are considered.
    - data (DataFrame): Contains daily sales data for each product with associated special prices.

    Workflow:
    1. Filters products with total items > 8800 (this can be varied and changed).
    2. Iterates over each product ID and performs:
       - Time series preparation with missing dates filled.
       - Cross-validation to find optimal `changepoint_prior_scale` and `seasonality_prior_scale` by 
         evaluating various parameter combinations using RMSE and MAPE (Mean Absolute Percentage Error).
       - Forecasting future sales based on the best parameters.
    3. Compares forecasted values with actual values using accuracy metrics.
    4. Returns:
       - A DataFrame containing accuracy metrics (MAPE, RMSE) and the best hyperparameters for each product.
       - A DataFrame containing forecasted sales values along with trend and price data for further regression analysis.

    Returns:
    - final_errors (DataFrame): Performance metrics including RMSE, MAPE, and the best hyperparameters for each product.
    - final_reg_df (DataFrame): The dataset containing forecasted trends and actual sales values for price elasticity analysis.

    Metrics Evaluated:
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Comparison against a naive forecasting model for benchmarking
    
    Note - the values period, horizon and initial can be iterated too. I took a rough value 
    of period equal to 10 and horizon equals to 20. We can experiment with these parameters too.
    """
    
    final_errors = pd.DataFrame()
    final_reg_df = pd.DataFrame()

    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.3, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    details_ = details.loc[details['total_items']>8800].reset_index(drop=True)

    for pid in details_['product_id'][0:]:
        print('Running pid:',pid)
        prod = data.loc[data['product_id']==pid]
        prod['special_price_o'] = prod['special_price'].astype('O')
        prod['order_date'] = pd.to_datetime(prod['order_date'])
        prod = prod.sort_values('order_date').reset_index(drop=True)


        date_series_df = pd.DataFrame(pd.date_range(prod['order_date'].min(),prod['order_date'].max()),columns=['order_date'])
        prod = pd.merge(date_series_df,prod,how="left",on=['order_date'])

        model_df = prod[['order_date','total_items']]
        model_df.rename(columns={'order_date':'ds','total_items':'y'},inplace=True)
        model_df['ds'] = pd.to_datetime(model_df['ds'])

        n = 20
        X_train = model_df.iloc[:-n,:]
        X_test = model_df.iloc[-n:,:]

        rmses = []  # Store the RMSEs for each params here
        mapes = []


        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params,interval_width=0.95).fit(X_train)  # Fit model with given params

        #     m = Prophet(**params,interval_width=0.95,holidays=holidays_df)
        #     m.add_country_holidays(holidays_df).fit(model_df) 
            if X_train.shape[0]<200:
                initial = '150 days'
            else:
                total = 200+10+20
                if total>X_train.shape[0]:
                    initial = '150 days'  
                else:
                    initial = '200 days'

            df_cv = cross_validation(m, initial=initial, period='10 days', horizon = '20 days',parallel='processes')
            df_p = performance_metrics(df_cv, rolling_window=0.1)
            rmses.append(df_p['rmse'].values[0])
            mapes.append(df_p['mape'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        tuning_results['mape'] = mapes

        # forecasting using best parameters
        best_df = tuning_results.loc[tuning_results['rmse']==tuning_results['rmse'].min()]
        changepoint_prior_scale = best_df['changepoint_prior_scale'].iloc[0]
        seasonality_prior_scale = best_df['seasonality_prior_scale'].iloc[0]

    #     changepoint_prior_scale = 0.3
    #     seasonality_prior_scale = 0.1

        model = Prophet(interval_width=0.95,changepoint_prior_scale=changepoint_prior_scale,seasonality_prior_scale=seasonality_prior_scale)
        model.fit(X_train)

        future = model.make_future_dataframe(periods=20)
        forecast = model.predict(future)

        # comparison with baseline
        comp_train = pd.merge(X_train,forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],how="left",on=['ds'])
        comp_train['yhat'] = np.where(comp_train['yhat']<0,0,comp_train['yhat'])
        comp_test = pd.merge(X_test,forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],how="left",on=['ds'])
        comp_test['yhat'] = np.where(comp_test['yhat']<0,0,comp_test['yhat'])

        comp_train['y_hat_naive'] = comp_train['y'].shift(1)
        comp_test['y_hat_naive'] = comp_test['y'].shift(1)

        naive_train = comp_train.loc[(comp_train['y'].isnull()==False) & (comp_train['y_hat_naive'].isnull()==False)].reset_index(drop=True)
        naive_test =  comp_test.loc[(comp_test['y'].isnull()==False) & (comp_test['y_hat_naive'].isnull()==False)].reset_index(drop=True)

        from sklearn.metrics import mean_squared_error

        y_train = comp_train['y'].loc[comp_train['y'].isnull()==False].reset_index(drop=True)
        y_train_hat = comp_train['yhat'].loc[comp_train['y'].isnull()==False].reset_index(drop=True)


        y_test = comp_test['y'].loc[(comp_test['y'].isnull()==False) & (comp_test['yhat'].isnull()==False)].reset_index(drop=True)
        y_test_hat = comp_test['yhat'].loc[(comp_test['y'].isnull()==False) & (comp_test['yhat'].isnull()==False)].reset_index(drop=True)

        rmse_train = np.sqrt(mean_squared_error(y_train,y_train_hat))
        error_percentage_train = rmse_train/np.mean(y_train)
        mape_train = np.mean(abs((y_train - y_train_hat)/y_train))

        rmse_train_naive = np.sqrt(mean_squared_error(naive_train['y'],naive_train['y_hat_naive']))
        error_percentage_train_naive = rmse_train_naive/np.mean(naive_train['y'])

        rmse_test= np.sqrt(mean_squared_error(y_test,y_test_hat))
        error_percentage_test = rmse_test/np.mean(y_test)
        mape_test = np.mean(abs((y_test - y_test_hat)/y_test))

        rmse_test_naive = np.sqrt(mean_squared_error(naive_test['y'],naive_test['y_hat_naive']))
        error_percentage_test_naive = rmse_test_naive/np.mean(naive_test['y'])

        temp = pd.DataFrame({'pid':[pid],'changepoint_prior_scale':[changepoint_prior_scale],'seasonality_prior_scale':[seasonality_prior_scale],'train_rmse':[rmse_train],'test_rmse':[rmse_test],'train_mape':[mape_train],'test_mape':[mape_test],'naive_train_rmse':[rmse_train_naive],'naive_test_rmse':[rmse_test_naive],
                            'rmse_train_error_percentage':[error_percentage_train],'rmse_test_error_percentage':[error_percentage_test],'rmse_naive_train_error_percentage':[error_percentage_train_naive],'rmse_naive_test_error_percentage':[error_percentage_test_naive]})

        final_errors = pd.concat([final_errors,temp],axis=0,ignore_index=True)

        prod.rename(columns={'order_date':'ds','total_items':'y'},inplace=True)
        prod['ds'] = pd.to_datetime(prod['ds'])
        reg_df = pd.merge(forecast[['ds','trend','yhat','weekly']],prod,how="left",on=['ds'])
        final_reg_df = pd.concat([final_reg_df,reg_df],axis=0,ignore_index=True)
        
    return final_errors,final_reg_df

# Ordinary Least Squares Regression
def build_model(X,y):

    X = sm.add_constant(X) # Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    r2 = lm.rsquared_adj   # model summary
    return r2,lm

def getElasticities(reg_data):

    """
    Calculates price elasticities for each product by performing regression analysis on trend and price data 
    obtained from the forecasting step. The function identifies price changes and evaluates the relationship 
    between price and sales trend using logarithmic regression.
    Confidence intervals, pvalues and r2 are reported too. 
    
    """

    elasticity_df = pd.DataFrame()
    current_price_df = pd.DataFrame()
    product_ids = reg_data['product_id'].loc[reg_data['product_id'].isnull()==False].unique()
    
    for pid in product_ids:
        prod = reg_data.loc[reg_data['product_id']==pid].reset_index(drop=True)
        prod['shift'] = prod['special_price'].shift(1)
        prod['is_change'] = np.where(prod['special_price_o']!=prod['shift'],1,0)
        prod['cum_sum'] = prod['is_change'].cumsum()
        print(prod)

        large = prod['cum_sum'].max()
        
        # this block is to find the latest price of a pid 
        if prod.loc[prod['cum_sum']==large].shape[0]>10:
            n_rows = prod.loc[prod['cum_sum']==large].shape[0]
            base_units_trend = prod.loc[prod['cum_sum']==large]['trend'].mean()
            base_units_pred = prod.loc[prod['cum_sum']==large]['yhat'].mean()
            base_units_count = prod.loc[prod['cum_sum']==large]['y'].mean()
            current_price = prod.loc[prod['cum_sum']==large]['special_price'].iloc[0]
            print("Current_Price:",current_price)
        else:
            n_rows = prod.loc[prod['cum_sum']==large-1].shape[0]
            base_units_trend = prod.loc[prod['cum_sum']==large-1]['trend'].mean()
            base_units_pred = prod.loc[prod['cum_sum']==large-1]['yhat'].mean()
            base_units_count = prod.loc[prod['cum_sum']==large-1]['y'].mean()
            current_price = prod.loc[prod['cum_sum']==large-1]['special_price'].iloc[0]

        temp = pd.DataFrame([base_units_trend,base_units_pred,base_units_count,current_price,n_rows],index=['base_units_trend','base_units_pred','base_units_count','current_price','n_rows']).T
        temp['product_id'] = pid
        current_price_df = pd.concat([current_price_df,temp],axis=0,ignore_index=True)

        model_obj = build_model(np.log(prod['special_price']),np.log(prod['trend']))[1]

        ci = model_obj.conf_int(alpha=0.05)
        conf_int_df = pd.DataFrame(ci)
        conf_int_df.rename(columns={0:'ci_0.025',1:'ci_0.975'},inplace=True)

        coeff_df = pd.DataFrame(model_obj.params,columns=['weights'])
        p_df = pd.DataFrame(model_obj.pvalues,columns=['p_value'])

        merged = pd.merge(coeff_df,p_df,left_index=True,right_index=True)
        merged = pd.merge(merged,conf_int_df,left_index=True,right_index=True).reset_index().rename(columns={'index':'feature'})
        merged['r2_adj'] = model_obj.rsquared_adj
        merged['product_id'] = pid

        elasticity_df = pd.concat([elasticity_df,merged],axis=0,ignore_index=True)
    ## optimization 
    result = pd.merge(current_price_df,elasticity_df.loc[elasticity_df['feature']!='const'].iloc[:,1:],how="left",on=['product_id'])

    return result

def getOptimisedPrice(to_optimise_df,cost,errors):

    '''
    this function gives us the optimal price of the product ids according to the elasticities obtained
    from regression analysis.

    Equation followed for optimisation is given below

    optimised_revenue = (base_units + delta_base_units)*(optimised_price - cost_price)

    where
    delta_base_units = base_units*price_elasticity*price_change
    price_change = (optimised_price/current_price) - 1
    
    '''

    result = pd.merge(to_optimise_df,cost,how="left",on=['product_id'])

    result['recommended_price'] = (result['current_price']*(result['weights']-1))/(2*result['weights']) + (result['avg_cost']/2)
    result['lb_price'] = (result['current_price']*(result['ci_0.025']-1))/(2*result['ci_0.025']) + (result['avg_cost']/2)
    result['ub_price'] = (result['current_price']*(result['ci_0.975']-1))/(2*result['ci_0.975']) + (result['avg_cost']/2)

    result['price_change'] = (result['recommended_price'] - result['current_price'])/result['current_price']
    result['delta_base_units'] = result['base_units_trend']*result['weights']*result['price_change']
    result['optimised_units'] = result['base_units_trend'] + result['delta_base_units']
    result['optimised_profit'] = result['optimised_units']*(result['recommended_price']-result['avg_cost'])
    result['initial_profit'] = result['base_units_trend']*(result['current_price']-result['avg_cost'])
    result['net_gain'] = result['optimised_profit'] - result['initial_profit']
    result['net_gain_%'] = (result['optimised_profit'] - result['initial_profit'])/result['initial_profit']
    
    errors.rename(columns={'pid':'product_id'},inplace=True)
    result = pd.merge(errors,result,how="left",on=['product_id'])
    
    col = ['product_id','weights','ci_0.025', 'ci_0.975','current_price','recommended_price', 'lb_price', 'ub_price',
           'n_rows', 'base_units_trend', 'base_units_pred', 'base_units_count','price_change',
           'delta_base_units', 'optimised_units','avg_cost','optimised_profit', 'initial_profit', 'net_gain','net_gain_%','train_rmse', 'test_rmse',
           'train_mape', 'test_mape', 'rmse_train_error_percentage','rmse_test_error_percentage', 'naive_train_rmse', 'naive_test_rmse',
           'p_value', 'r2_adj']
    result = result[col]
    
    return result

if __name__ == '__main__':

    start_time = time.time()

    start_date = '2023-06-01'
    end_date = datetime.date.today() - timedelta(days=1)
    l_30 = end_date - timedelta(days=30)
    
    details,data,cost = getData(start_date,end_date,l_30)
    print("Details:",details.head())

    errors,reg = getBaseSales(details,data)
    print("Errors:",errors)
    
    to_optimise_df = getElasticities(reg)
    final = getOptimisedPrice(to_optimise_df,cost,errors)
    print("Result:",final)

