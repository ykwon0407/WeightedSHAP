from tqdm import tqdm 
import numpy as np

# custom modules
import data, utils
from third_party import behavior

beta_hyper_list=[(-1,'LOO-First'), (1,32), (1,16), (1, 8), (1, 4), (1, 2), 
                    (1,1), (2,1), (4,1), (8, 1), (16, 1), (32,1), (-1,'LOO-Last')]
attribution_list=['LOO-First', 'Beta(32,1)', 'Beta(16,1)', 'Beta(8,1)', 
                 'Beta(4,1)', 'Beta(2,1)', 'Beta(1,1)', 'Beta(1,2)',
                 'Beta(1,4)', 'Beta(1,8)', 'Beta(1,16)', 'Beta(1,32)', 'LOO-Last']

def MarginalContributionValue(game, thresh=1.005, batch_size=1, n_check_period=100):
    '''Calculate feature attributions using the marginal contributions.'''

    # index of the added feature, cardinality of set
    arange = np.arange(batch_size)
    output = game(np.zeros(game.players, dtype=bool))
    MC_size=[game.players, game.players] + list(output.shape)
    MC_mat=np.zeros(MC_size) 
    MC_count=np.zeros((game.players, game.players))

    # MCI SCORE UPDATE
    MCI_score=np.zeros(game.players)

    converged = False
    n_iter = 0
    marginal_contribs=np.zeros([0, np.prod([game.players] + list(output.shape))]) 
    while not converged:
        MCI_score_converge_score=0
        for _ in range(n_check_period):
            # Sample permutations.
            permutations = np.tile(np.arange(game.players), (batch_size, 1))
            for row in permutations:
                np.random.shuffle(row)
            S = np.zeros((batch_size, game.players), dtype=bool)

            # Unroll permutations.
            prev_value = game(S)
            marginal_contribs_tmp = np.zeros(([batch_size, game.players] + list(output.shape)))
            for j in range(game.players):
                '''
                Marginal contribution estimates with respect to j samples
                j = 0 means LOO-First
                '''
                S[arange, permutations[:, j]] = 1
                next_value = game(S)
                MC_mat[permutations[:, j], j] += (next_value - prev_value)
                MC_count[permutations[:, j], j] += 1
                marginal_contribs_tmp[arange, permutations[:, j]] = (next_value - prev_value)

                # MCI SCORE UPDATE
                if MCI_score[j] < (next_value - prev_value):
                    MCI_score[j] = next_value - prev_value
                else:
                    MCI_score_converge_score+=1

                # update
                prev_value = next_value
            marginal_contribs=np.concatenate([marginal_contribs, marginal_contribs_tmp.reshape(batch_size,-1)], axis=0)
            
        if (n_iter+1) == 1000:
            converged=True
        elif (n_iter+1) >= 5:
            if MCI_score_converge_score > 0.999*(game.players*n_check_period):
                if utils.check_convergence(marginal_contribs) < thresh: 
                    print(f'MCI score: {MCI_score_converge_score}, Therehosld: {int(0.999*(game.players*n_check_period))}')
                    converged=True
        else:
            pass

        n_iter += 1

    print(f'We have seen {((n_iter+1)*n_check_period*batch_size)} random subsets for each feature.')
    if len(MC_mat.shape) != 2:
        # classification case
        MC_count=np.repeat(MC_count, MC_mat.shape[-1], axis=-1).reshape(MC_mat.shape)
    
    return MC_mat, MC_count, MCI_score

def run_attribution_core(problem, ML_model, 
                        model_to_explain, conditional_extension, 
                        X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    Compute attribution values and evaluate its performance
    '''
    pred_list, pred_masking = [], []
    cond_pred_keep_absolute, cond_pred_remove_absolute=[], []
    n_max=min(100, len(X_test))
    for ind in tqdm(range(n_max)):
        # Store original prediction 
        original_pred=utils.compute_predict(model_to_explain, X_test[ind,:].reshape(1,-1), problem, ML_model)
        pred_list.append(original_pred)

        # Estimate marginal contributions
        conditional_game=behavior.PredictionGame(conditional_extension, X_test[ind, :])
        MC_conditional_mat, MC_conditional_count, MCI_score=MarginalContributionValue(conditional_game) 
        MC_est=np.array(MC_conditional_mat/(MC_conditional_count+1e-16))
        
        # Optimize weight for WeightedSHAP (When AUP is used)
        attribution_dict_all=utils.compute_beta_dict_from_MC(MC_est, beta_hyper_list)
        cond_pred_keep_absolute_list=utils.compute_cond_pred_list(attribution_dict_all, conditional_game)
        AUP_list=np.sum(np.abs(np.array(cond_pred_keep_absolute_list)- original_pred), axis=1)
        WeightedSHAP_index=np.argmin(AUP_list)

        '''
        Evaluation
        '''
        attribution_dict_all['MCI']=MCI_score
        # Conditional prediction from most important to least important (keep absolte)
        cond_pred_keep_absolute_list=utils.compute_cond_pred_list(attribution_dict_all, conditional_game)
        cond_pred_keep_absolute.append(cond_pred_keep_absolute_list)

        cond_pred_remove_absolute_list=utils.compute_cond_pred_list(attribution_dict_all, conditional_game, more_important_first=False)
        cond_pred_remove_absolute.append(cond_pred_remove_absolute_list)

        # Prediction after masking from most important to least important (keep absolte)
        pred_masking_list=utils.compute_pred_maksing_list(attribution_dict_all, model_to_explain, X_test[ind,:].reshape(1,-1), problem, ML_model)
        pred_masking.append(pred_masking_list)

    exp_dict=dict()
    exp_dict['true_list']=np.array(y_test)[:n_max]
    exp_dict['pred_list']=np.array(pred_list)   
    exp_dict['cond_pred_keep_absolute']=np.array(cond_pred_keep_absolute)   
    exp_dict['cond_pred_remove_absolute']=np.array(cond_pred_remove_absolute)   
    exp_dict['pred_masking']=np.array(pred_masking)

    return exp_dict



