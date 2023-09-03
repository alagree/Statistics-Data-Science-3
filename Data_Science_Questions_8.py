import scipy
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols 


dir_path = '.../Assignment4_linear_regresion_data.xlsx'

x_file = pd.ExcelFile(dir_path)
s_names = x_file.sheet_names

#Question 1a) Visually inspect the plots to determine if linear model is appropriate
for idx in s_names:
    df = pd.read_excel(dir_path, sheet_name=idx)
    plt.figure()
    sns.scatterplot(data=df, x='x', y='y')
    plt.title(idx)
    plt.show()

#Question 1b) Transform the non-linear relation accordingly 
non_linear = ['Set 3', 'Set 4']
for idx in non_linear:
    df = pd.read_excel(dir_path, sheet_name=idx)
    yt = stats.boxcox(df['y'])[0]
    plt.figure()
    sns.scatterplot(x=df['x'], y=yt)
    plt.title(idx + 'transformed')
    plt.show()   


#Question 1c) Visually review the data again and check for outliers
df_with_outliers = []
for idx in s_names:
    if idx not in non_linear:
        df = pd.read_excel(dir_path, sheet_name=idx)
        model = ols("y ~ x", data=df).fit()
        sm.graphics.influence_plot(model) 
        plt.title(idx)
        plt.show()
        sm.graphics.plot_regress_exog(model, 'x')
        plt.show()
        
        outlier_measures = model.get_influence()
        leverage = outlier_measures.hat_matrix_diag
        student_resid = outlier_measures.resid_studentized_external
        if any(student_resid > 3) or any(student_resid < -3) or any(leverage > (3 * (2/len(leverage)))):
            df_with_outliers.append(idx)
            outlier_measures = model.get_influence()
            student_resid = outlier_measures.resid_studentized_external
            sr_index = np.where((student_resid > 3) | (student_resid < -3))[0]
            leverage = outlier_measures.hat_matrix_diag
            lev_index = np.where(leverage > (3 * (2/len(leverage))))[0] 
            #Check if any of the indexs are the same
            idx_to_remove = list(np.append(sr_index, lev_index))
            idx_to_remove = list(dict.fromkeys(idx_to_remove))
            #Remove outlier from df
            df = df.drop(df.index[idx_to_remove])
            model = ols("y ~ x", data=df).fit()
            sm.graphics.influence_plot(model) 
            plt.title(idx + ' outliers removed')
            plt.show()
            sm.graphics.plot_regress_exog(model, 'x')
            plt.show()
        
    else:
        df = pd.read_excel(dir_path, sheet_name=idx)
        model = ols("y ~ x", data=df).fit()
        sm.graphics.influence_plot(model) 
        plt.title(idx)
        plt.show()
        sm.graphics.plot_regress_exog(model, 'x')
        plt.show()
        
        yt = stats.boxcox(df['y'])[0]
        df['yt'] = yt
        model = ols("yt ~ x", data=df).fit()
        sm.graphics.influence_plot(model) 
        plt.title(idx + ' linear transform')
        plt.show()
        sm.graphics.plot_regress_exog(model, 'x')
        plt.show()
        
        outlier_measures = model.get_influence()
        leverage = outlier_measures.hat_matrix_diag
        student_resid = outlier_measures.resid_studentized_external
        if any(student_resid > 3) or any(student_resid < -3) or any(leverage > (3 * (2/len(leverage)))):
            df_with_outliers.append(idx)
            outlier_measures = model.get_influence()
            student_resid = outlier_measures.resid_studentized_external
            sr_index = np.where((student_resid > 3) | (student_resid < -3))[0]
            leverage = outlier_measures.hat_matrix_diag
            lev_index = np.where(leverage > (3 * (2/len(leverage))))[0] 
            #Check if any of the indexs are the same
            idx_to_remove = list(np.append(sr_index, lev_index))
            idx_to_remove = list(dict.fromkeys(idx_to_remove))
            #Remove outlier from df
            df = df.drop(df.index[idx_to_remove])        
            model = ols("yt ~ x", data=df).fit()
            sm.graphics.influence_plot(model) 
            plt.title(idx + ' outliers removed')
            plt.show()
            sm.graphics.plot_regress_exog(model, 'x')
            plt.show()
    
#Question 1d) Create the OLS reports
for idx in s_names:
    if idx not in non_linear:
        df = pd.read_excel(dir_path, sheet_name=idx)
        model = ols("y ~ x", data=df).fit()
        print(f'>>>OLS report for {idx}<<<')
        print(model.summary())
        
        outlier_measures = model.get_influence()
        leverage = outlier_measures.hat_matrix_diag
        student_resid = outlier_measures.resid_studentized_external
        if any(student_resid > 3) or any(student_resid < -3) or any(leverage > (3 * (2/len(leverage)))):
            outlier_measures = model.get_influence()
            student_resid = outlier_measures.resid_studentized_external
            sr_index = np.where((student_resid > 3) | (student_resid < -3))[0]
            leverage = outlier_measures.hat_matrix_diag
            lev_index = np.where(leverage > (3 * (2/len(leverage))))[0] 
            #Check if any of the indexs are the same
            idx_to_remove = list(np.append(sr_index, lev_index))
            idx_to_remove = list(dict.fromkeys(idx_to_remove))
            #Remove outlier from df
            df = df.drop(df.index[idx_to_remove])
            model = ols("y ~ x", data=df).fit()
            print(f'>>>OLS report for {idx}, outliers removed<<<')
            print(model.summary())
        
    else:
        df = pd.read_excel(dir_path, sheet_name=idx)
        model = ols("y ~ x", data=df).fit()
        print(f'>>>OLS report for {idx}<<<')
        print(model.summary())
        yt = stats.boxcox(df['y'])[0]
        df['yt'] = yt
        model = ols("yt ~ x", data=df).fit()
        print(f'>>>OLS report for {idx}, data transformed<<<')
        print(model.summary())
        
        outlier_measures = model.get_influence()
        leverage = outlier_measures.hat_matrix_diag
        student_resid = outlier_measures.resid_studentized_external
        if any(student_resid > 3) or any(student_resid < -3) or any(leverage > (3 * (2/len(leverage)))):
            outlier_measures = model.get_influence()
            student_resid = outlier_measures.resid_studentized_external
            sr_index = np.where((student_resid > 3) | (student_resid < -3))[0]
            leverage = outlier_measures.hat_matrix_diag
            lev_index = np.where(leverage > (3 * (2/len(leverage))))[0] 
            #Check if any of the indexs are the same
            idx_to_remove = list(np.append(sr_index, lev_index))
            idx_to_remove = list(dict.fromkeys(idx_to_remove))
            #Remove outlier from df
            df = df.drop(df.index[idx_to_remove])        
            model = ols("yt ~ x", data=df).fit()
            print(f'>>>OLS report for {idx}, data transformed and outliers removed<<<')
            print(model.summary())


#Question 1g) Check if the outliers are influential
for idx in df_with_outliers:
    df = pd.read_excel(dir_path, sheet_name=idx)
    model = ols("y ~ x", data=df).fit()    
    summary = model.summary()
    summary = summary.tables[1].as_html()
    summary_original = pd.read_html(summary, header=0, index_col=0)[0]

    outlier_measures = model.get_influence()
    student_resid = outlier_measures.resid_studentized_external
    sr_index = np.where((student_resid > 3) | (student_resid < -3))[0]
    leverage = outlier_measures.hat_matrix_diag
    lev_index = np.where(leverage > (3 * (2/len(leverage))))[0] 
    #Check if any of the indexs are the same
    idx_to_remove = list(np.append(sr_index, lev_index))
    idx_to_remove = list(dict.fromkeys(idx_to_remove))
    #Remove outlier from df
    df = df.drop(df.index[idx_to_remove])        
    model = ols("y ~ x", data=df).fit()
    summary = model.summary()
    summary = summary.tables[1].as_html()
    summary_or = pd.read_html(summary, header=0, index_col=0)[0]

    obs_diff = summary_or['coef']['x'] - summary_original['coef']['x']
    se = math.sqrt(summary_or['std err']['x']**2 + summary_original['std err']['x']**2)
    p_value = (1 - scipy.stats.norm.cdf(obs_diff/se)) * 2


















































