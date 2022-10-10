import numpy as np
from time import time
import torch
import torch.nn as nn
from scipy.stats import ttest_ind

from weightedSHAP.third_party import removal, surrogate
from weightedSHAP.third_party.utils import MaskLayer1d, KLDivLoss, MSELoss

def create_boosting_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model):    
    print('Train a model to explain: Boosting')
    import lightgbm as lgb
    if problem == 'classification':
        params = {
            "learning_rate": 0.005,
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error"],
            "num_threads": 1,
            "verbose": -1,
            "num_leaves":15,
            "bagging_fraction":0.5,
            "feature_fraction":0.5,
            "lambda_l2": 1e-3
        }
    else:
        params = {
            "learning_rate": 0.005,
            "objective": "mean_squared_error",
            "metric": "mean_squared_error",
            "num_threads": 1,
            "verbose": -1,
            "num_leaves":15,
            "bagging_fraction":0.5,
            "feature_fraction":0.5,
            "lambda_l2": 1e-3
        }

    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)

    callbacks=[lgb.early_stopping(25)] 
    model_to_explain=lgb.train(params, d_train, 
                              num_boost_round=1000,
                              valid_sets=[d_val], 
                              callbacks=callbacks)

    return model_to_explain

def create_linear_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model):
    print('Train a model to explain: Linear')
    from sklearn.linear_model import LinearRegression, LogisticRegression
    if problem == 'classification':
        model_to_explain=LogisticRegression()
        model_to_explain.fit(X_train, y_train)
    else:
        model_to_explain=LinearRegression()
        model_to_explain.fit(X_train, y_train)

    return model_to_explain

def create_MLP_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model):
    print('Train a model to explain: MLP')
    device = torch.device('cpu')
    num_features=X_train.shape[1]
    n_output=2 if problem=='classification' else 1
    model_to_explain = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, n_output)).to(device)

    # training part
    return model_to_explain

def check_overfitting(X_train, y_train, X_val, y_val, model_to_explain, problem, ML_model):
    '''
    Check overfitting
    '''
    if problem == 'classification':
        tmp_err=lambda y1, y2: ((y1 > 0.5) != y2) + 0.0
    else:
        tmp_err=lambda y1, y2: (y1-y2)**2

    if (ML_model == 'linear') and (problem == 'classification'):
        tr_pred_error=tmp_err(model_to_explain.predict_proba(X_train)[:,1], y_train)
        val_pred_error=tmp_err(model_to_explain.predict_proba(X_val)[:,1], y_val)
    elif ML_model == 'MLP':
        # not used 
        y_train_pred=model_to_explain(torch.from_numpy(X_train.astype(np.float32))).detach().numpy().reshape(-1)
        y_val_pred=model_to_explain(torch.from_numpy(X_val.astype(np.float32))).detach().numpy().reshape(-1)
        tr_pred_error=tmp_err(y_train_pred, y_train)
        val_pred_error=tmp_err(y_val_pred, y_val)
    else:
        tr_pred_error=tmp_err(model_to_explain.predict(X_train), y_train)
        val_pred_error=tmp_err(model_to_explain.predict(X_val), y_val) 

    p_value=ttest_ind(tr_pred_error, val_pred_error)[1]
    overfitting_check='Not overfitted' if p_value > 0.01 else 'Overfitted'

    tr_err, val_err = np.mean(tr_pred_error), np.mean(val_pred_error)
    print(f'Overfitting? / P-value: {overfitting_check} / {p_value:.4f}')
    print(f'Tr error, Val error: {tr_err:.3f}, {val_err:.3f}')
    return tr_err, val_err

def create_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model):    
    print('-'*30)
    print('Train a model')
    start_time=time()
    if ML_model=='linear':
        model_to_explain=create_linear_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model)
    elif ML_model=='boosting':
        model_to_explain=create_boosting_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model)
    elif ML_model=='MLP':
        model_to_explain=create_MLP_model_to_explain(X_train, y_train, X_val, y_val, problem, ML_model)
    else:
        raise ValueError(f'Check ML_model: {ML_model}') 

    elapsed_time_train=time()-start_time
    print(f'Elapsed time for training a model to explain: {elapsed_time_train:.2f} seconds')
    print('-'*30)
    return model_to_explain # , tr_err, val_err


def create_surrogate_model(model_to_explain, X_train, X_est, problem='classification', ML_model='linear', verbose=False):
    start_time=time()
    # [Step 1] Create surrogate model
    device = torch.device('cpu')
    num_features=X_train.shape[1]
    n_output=2 if problem=='classification' else 1
    surrogate_model = nn.Sequential(
        MaskLayer1d(value=0, append=True),
        nn.Linear(2 * num_features, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, n_output)).to(device)

    # Set up surrogate object
    surrogate_object = surrogate.Surrogate(surrogate_model, num_features)

    if problem=='classification':
        loss_fn=KLDivLoss()
        # predict_proba and predict
        def original_model(x):
            if ML_model == 'linear':
                pred = (model_to_explain.predict_proba(x.cpu().numpy())[:,1]).reshape(-1)
            else:
                pred = model_to_explain.predict(x.cpu().numpy())

            pred = np.stack([1 - pred, pred]).T
            return torch.tensor(pred, dtype=torch.float32, device=x.device)
    else:
        loss_fn=MSELoss()
        def original_model(x):
            pred = model_to_explain.predict(x.cpu().numpy()).reshape(-1, 1)
            return torch.tensor(pred, dtype=torch.float32, device=x.device)

    # Train
    surrogate_object.train_original_model(X_train,
                                        X_est,
                                        original_model,
                                        batch_size=64,
                                        max_epochs=100, 
                                        loss_fn=loss_fn,
                                        validation_samples=128, 
                                        validation_batch_size=(2**12),
                                        lookback=10,
                                        verbose=verbose)
    elapsed_time=time()-start_time
    print(f'Elapsed time for training a surrogate model: {elapsed_time:.2f} seconds')
    return surrogate_model


def generate_coalition_function(model_to_explain, X_train, X_est,
                                problem='classification', ML_model='linear', verbose=False):
    surrogate_model=create_surrogate_model(model_to_explain, X_train, X_est, problem, ML_model, verbose)

    device=torch.device('cpu')
    if problem == 'classification':
        model_condi_wrapper = lambda x, S: surrogate_model((torch.tensor(x, dtype=torch.float32, device=device),
           torch.tensor(S, dtype=torch.float32, device=device))).softmax(dim=-1).cpu().data.numpy().reshape(x.shape[0],-1)[:,1]
    elif problem == 'regression':
        model_condi_wrapper = lambda x, S: surrogate_model((torch.tensor(x, dtype=torch.float32, device=device),
           torch.tensor(S, dtype=torch.float32, device=device))).cpu().data.numpy().reshape(x.shape[0],-1)[:,0]
    else:
        raise ValueError(f'Check problem: {problem}')
    conditional_extension=removal.ConditionalSupervisedExtension(model_condi_wrapper)

    return conditional_extension

