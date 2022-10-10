import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

def load_data(problem, dataset, dir_path, random_factor=2, random_seed=2022):
    '''
    Load a dataset
    We split datasets as Train:Val:Est:Test=7:1:1:1
    Train: to train a model
    Val: to optimize hyperparameters
    Est: to estimate coalition functions
    Test: to evaluate performance  
    '''
    print('-'*30)
    print('Load a dataset')
    print('-'*30)
    
    if problem=='regression':
        (X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)=load_regression_dataset(dataset=dataset, dir_path=dir_path, rid=random_seed)
    elif problem=='classification':
        (X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)=load_classification_dataset(dataset=dataset, dir_path=dir_path, rid=random_seed)
    else:
        raise NotImplementedError('Check problem')

    if random_factor != 0:
        # We add noisy features to the original dataset.
        print('-'*30)
        print('Before adding noise')
        print(f'Shape of X_train, X_val, X_est, X_test: {X_train.shape}, {X_val.shape}, {X_est.shape}, {X_test.shape}')
        print('-'*30)
        dim_noise=int(X_train.shape[1]*random_factor)
        X_train=extend_dataset(X_train, dim_noise, verbose=True)
        X_val=extend_dataset(X_val, dim_noise)
        X_est=extend_dataset(X_est, dim_noise)
        X_test=extend_dataset(X_test, dim_noise)            
        print('After adding noise')
        print(f'Shape of X_train, X_val, X_est, X_test: {X_train.shape}, {X_val.shape}, {X_est.shape}, {X_test.shape}')
        print('-'*30)
    else:
        # We use the original dataset.
        print('-'*30)
        print(f'Shape of X_train, X_val, X_est, X_test: {X_train.shape}, {X_val.shape}, {X_est.shape}, {X_test.shape}')
        print('-'*30)

    return (X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)

def load_regression_dataset(dataset='abalone', dir_path='dir_path', rid=1):
    '''
    This function loads regression datasets.
    dir_path: path to regression datasets.

    You may need to download datasets first. Make sure to store in 'dir_path'.
    The datasets are avaiable at the following links.
    abalone: https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/
    airfoil: https://archive.ics.uci.edu/ml/machine-learning-databases/00291/
    whitewine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    '''
    np.random.seed(rid)

    if dataset == 'boston':
        print('-'*50)
        print('Boston')
        print('-'*50)
        data=load_boston()
        X, y=data['data'], data['target']
    elif dataset == 'abalone':
        print('-'*50)
        print('Abalone')
        print('-'*50)
        raw_data = pd.read_csv(dir_path+"/abalone.data", header=None)
        raw_data.dropna(inplace=True)
        X, y = pd.get_dummies(raw_data.iloc[:,:-1],drop_first=True).values, raw_data.iloc[:,-1].values
    elif dataset == 'whitewine':
        print('-'*50)
        print('whitewine')
        print('-'*50)
        raw_data = pd.read_csv(dir_path+"/winequality-white.csv",delimiter=";")
        raw_data.dropna(inplace=True)
        X, y = raw_data.values[:,:-1], raw_data.values[:,-1]
    elif dataset == 'airfoil':
        print('-'*50)
        print('airfoil')
        print('-'*50)
        raw_data = pd.read_csv(dir_path+"/airfoil_self_noise.dat", sep='\t', names=['X1','X2,','X3','X4','X5','Y'])
        X, y = raw_data.values[:,:-1], raw_data.values[:,-1]
    else:
        raise NotImplementedError(f'Check {dataset}')

    X = standardize_data(X)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=float(1/9))
    X_train, X_est, y_train, y_est=train_test_split(X_train, y_train, test_size=float(1/8))

    return (X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)
    
def load_classification_dataset(dataset='gaussian', dir_path='dir_path', rid=1):
    '''
    This function loads classification datasets.
    dir_path: path to classification datasets.
    '''
    np.random.seed(rid)
    
    if dataset == 'gaussian':
        print('-'*50)
        print('Gaussian')
        print('-'*50)
        n, input_dim, rho=10000, 10, 0.25
        U_cov=np.diag((1-rho)*np.ones(input_dim))+rho
        U_mean=np.zeros(input_dim)
        X=np.random.multivariate_normal(U_mean, U_cov, n)

        beta_true=(np.linspace(input_dim,(41*input_dim/50),input_dim)/input_dim).reshape(input_dim,1)
        p_true=np.exp(X.dot(beta_true))/(1.+np.exp(X.dot(beta_true)))
        y=np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'fraud':
        print('-'*50)
        print('Fraud Detection')
        print('-'*50)
        data_dict=pickle.load(open(f'{dir_path}/fraud_dataset.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        X, y=make_balance_sample(data, target)
    else:
        raise NotImplementedError(f'Check {dataset}')

    X = standardize_data(X)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=float(1/9))
    X_train, X_est, y_train, y_est=train_test_split(X_train, y_train, test_size=float(1/8))

    return (X_train, y_train), (X_val, y_val), (X_est, y_est), (X_test, y_test)


'''
Data utils
'''

def extend_dataset(X, d_to_add, verbose=False):
    n, d_prev = X.shape
    rho=(np.sum((X.T.dot(X)/n)[:d_prev,:d_prev])-d_prev)/(d_prev*(d_prev-1))

    if -1/(4*(d_prev-1)+1e-16) > rho:
        if verbose is True:
            # if rho is too small, the sigma_square defined below can be negative
            print(f'Initial rho: {rho:.4f}')
            rho=max(-1/(4*(d_prev-1)+1e-16), rho)
            print(f'After fixing rho: {rho:.4f}')
    else:
        if verbose is True:
            print(f'Rho: {rho:.4f}')
    
    for _ in range(d_to_add):
        sigma_square=1-(rho**2)*(d_prev)/(1+rho*(d_prev-1)+1e-16)
        new_X = (rho/(1+rho*(d_prev-1)+1e-16))*X.dot(np.ones((d_prev,1)))
        new_X += np.random.normal(size=(n,1))*np.sqrt(sigma_square)
        X = np.concatenate((X, new_X), axis=1)
        d_prev += 1
    return X

def make_balance_sample(data, target):
    p = np.mean(target)
    minor_class=1 if p < 0.5 else 0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target

def standardize_data(X):  
    ss=StandardScaler()
    ss.fit(X)
    try:
        X = ss.transform(X.values)
    except:
        X = ss.transform(X)
    return X



