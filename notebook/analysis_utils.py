import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, auc, roc_curve
from scipy.stats import ttest_ind, norm
z_constant=norm.ppf(0.025)

def find_optimal_list(cond_pred, pred_list, selected_index=[j for j in range(13)], is_lower_better=True):
    if is_lower_better is True:
        diff_mat=np.mean(np.abs(cond_pred - pred_list.reshape(-1,1,1)), axis=-1)
    else:
        diff_mat=-np.mean(np.abs(cond_pred - pred_list.reshape(-1,1,1)), axis=-1)
    
    if selected_index:
        diff_mat=diff_mat[:,selected_index]
        return np.array(selected_index)[np.argmin(diff_mat, axis=1)]
    else:
        return np.argmin(diff_mat, axis=1)
    
def compute_post_hoc_performance(cond_pred_mat, y_test, is_acc=False):
    post_hoc_acc, post_hoc_auc=[], []
    for att_ind in range(cond_pred_mat.shape[1]):
        post_hoc_acc_tmp, post_hoc_auc_tmp=[], []
        for k in range(cond_pred_mat.shape[2]):
            y_pred=cond_pred_mat[:, att_ind, k]
            post_hoc_acc_tmp.append(np.mean(y_test == (y_pred>0.5).astype(float))) 
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            post_hoc_auc_tmp.append(auc(fpr, tpr)) 
        post_hoc_acc.append(post_hoc_acc_tmp)
        post_hoc_auc.append(post_hoc_auc_tmp)
    if is_acc is True:
        return post_hoc_acc 
    else:
        return post_hoc_auc 
 
def draw_prediction_task(recovery_curve_array, attribution_list, name='keep_absolute', data_id=0, model_character='L'):
    recovery_curve_array=np.array(recovery_curve_array)
    selected_attribution_list=[0,1,2] 
    xlabel_text='Number of features added' 
    
    plt.figure(figsize=(4,4))
    recovery_curve_mean=np.mean(recovery_curve_array, axis=0)
    recovery_curve_ste=np.std(recovery_curve_array, axis=0)/np.sqrt(len(recovery_curve_array))
    n_features=recovery_curve_mean.shape[1]
    n_display_features=int(n_features*0.6)
    for att_ind in selected_attribution_list:
        if attribution_list[att_ind]=='MCI':
            label, color='MCI', 'blue'
        elif attribution_list[att_ind]=='Shapley':
            label, color='Shapley', 'green'
        elif attribution_list[att_ind]=='WeightedSHAP':
            label, color='WeightedSHAP', 'red'
            
        plt.errorbar(np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features], 
                     recovery_curve_mean[att_ind][max(1,int(n_features*0.075)):n_display_features], 
                     z_constant*recovery_curve_ste[att_ind][max(1,int(n_features*0.075)):n_display_features],
                     label=label, color=color, linewidth=2, alpha=0.6)
    plt.title(f'Prediction recovery error curve \n Dataset: {clf_datasets[ind]}', fontsize=15)
    plt.xlabel(xlabel_text, fontsize=15)
    plt.ylabel(r'$|f(x)-\mathbb{E}[f(X) \mid X_S = x_S]|$', fontsize=15)
    plt.xticks(np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6],
               np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def draw_cond_auc_task(post_hoc_auc_array, attribution_list, name='keep_absolute', 
                  data_id=0, model_character='L', is_masking=False):
    post_hoc_auc_array=np.array(post_hoc_auc_array)
    selected_attribution_list=[0,1,2]
    
    plt.figure(figsize=(4,4))
    mean_list=np.mean(post_hoc_auc_array, axis=0)
    ste_list=np.std(post_hoc_auc_array, axis=0)/np.sqrt(len(post_hoc_auc_array))
    n_features=mean_list.shape[1]
    n_display_features=int(n_features*0.6)
    for att_ind in selected_attribution_list:
        if attribution_list[att_ind]=='MCI':
            label, color='MCI', 'blue'
        elif attribution_list[att_ind]=='Shapley':
            label, color='Shapley', 'green'
        elif attribution_list[att_ind]=='WeightedSHAP':
            label, color='WeightedSHAP', 'red'
            
        plt.errorbar(np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features], 
                 mean_list[att_ind,:][max(1,int(n_features*0.075)):n_display_features], 
                 z_constant*ste_list[att_ind,:][max(1,int(n_features*0.075)):n_display_features],
                 label=label, color=color, linewidth=2, alpha=0.6)
    if name == 'keep_absolute':
        plt.title(f'Inclusion AUC \n Dataset: {clf_datasets[data_id]}', fontsize=15)
        xlabel_text='Number of features added' 
    elif name == 'remove':
        plt.title(f'Exclusion AUC \n Dataset: {clf_datasets[data_id]}', fontsize=15)
        xlabel_text='Number of features removed'
    elif name == 'masking':
        plt.title(f'Inclusion AUC - Masking \n Dataset: {clf_datasets[data_id]}', fontsize=15)
        xlabel_text='Number of features added'
    else:
        assert False, f'check name: {name}'
    
    plt.xticks(np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6],
               np.arange(n_features)[max(1,int(n_features*0.075)):n_display_features][::n_display_features//6])
    plt.xlabel(xlabel_text, fontsize=15)
    plt.ylabel('AUC', fontsize=15)
    plt.legend(fontsize=12) 
    plt.tight_layout()
    plt.show()



