import scanpy as sc
import pandas as pd
import numpy as np
import sklearn
import warnings
import datetime

from functools import reduce
from anndata import AnnData
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,train_test_split

from functools import partial,reduce
from hyperopt import hp, Trials, fmin, tpe
from hyperopt import space_eval
from joblib import delayed, Parallel

# Function for reading marker genes from 'data' folder
def read_marker_genes(marker_type):
    test_genes = {"fridman":["ALDH1A3", "AOPEP", "CCND1", "CD44", "CDKN1A", "CDKN1C", "CDKN2A", "CDKN2B", "CDKN2D", "CITED2",
                                "CLTB", "COL1A2","CREG1","CRYAB","CCN2","CXCL14","CYP1B1","EIF2S2","ESM1","F3","FILIP1L","FN1","GSN","GUK1","HBS1L",
                                "HPS5","HSPA2","HTATIP2","IFI16","IFNG","IGFBP1","IGFBP2","IGFBP3","IGFBP4","IGFBP5","IGFBP6","IGFBP7","IGSF3",
                                "ING1","IRF5","IRF7","ISG15","MAP1LC3B","MAP2K3","MDM2","MMP1","NDN","NME2","NRG1","OPTN","PEA15","RAB13","RAB31",
                                "RAB5B","RABGGTA","RAC1","RBL2","RGL2","RHOB","RRAS","S100A11","SERPINB2","SERPINE1","SMPD1","SMURF2","SOD1","SPARC",
                                "STAT1","TES","TFAP2A","TGFB1I1","THBS1","TNFAIP2","TNFAIP3","TP53","TSPYL5","VIM","ALDH1A1","BMI1","CCNB1","CDC25B",
                                "CKS1BP7","COL3A1","E2F4","EGR1","ID1","LAMA1","LDB2","MARCKS","CCN4"],
              "sasp2":["VEGFA", "TNFRSF12A", "TNFRSF10C", "TNFRSF10B", "TIMP2", "TIMP1", "TGFB1", "SERPINE1", "TNFRSF1A",
                                    "PLAUR", "PLAU", "MMP14", "MMP13", "MMP7", "MMP3", "MIF", "LMNA", "KITLG", "IL32", "IGFBP7", "IGFBP2",
                                     "ICAM1", "FAS", "EREG", "CXCL17", "CXCL16", "CXCL8", "CXCL1", "CTSB", "CLU", "CCL20", "CCL2", "BTC",
                                     "AREG"
                                  ],
              "senmayo":pd.read_excel("data/senescence_list.xlsx",sheet_name="SenMayo")["symbol"].tolist(),
              "cellage":pd.read_excel("data/senescence_list.xlsx",sheet_name="CellAge Senescence Genes")["Symbol"].tolist()
                }

    test_genes["union"] = reduce(np.union1d, [test_genes["fridman"],
                        test_genes["sasp2"],
                        test_genes["senmayo"],
                        test_genes["cellage"]]
          )
    
    return test_genes[marker_type]

# Function for extracting expression data
def filter_anndata(anndata, ct_column, cts, donor_column, age_column, marker_genes = None, min_cells = 50):
    
    # Keep marker genes only
    comm_genes = np.intersect1d(anndata.var_names, marker_genes)
    anndata = anndata[:, anndata.var_names.isin(comm_genes)]
    
    # Reorder the gene columns
    anndata = anndata[:,comm_genes]
    
    anndata = anndata[anndata.obs[ct_column].isin(cts), :]
    
    # Select subjects having number of cells greater than min_cells 
    subjects = anndata.obs[donor_column]
    subjects_count = subjects.groupby(subjects.values).count()
    selected_subjects = subjects_count.loc[subjects_count >= min_cells].index 

    # Further subset anndata using the selected subjects
    anndata = anndata[anndata.obs[donor_column].isin(selected_subjects),]
    donor_num = anndata.obs[donor_column].unique().shape[0]
    
    exprs = []
    for ct in cts:
        
        # get anndata for the current cell type
        ann_sub = anndata[anndata.obs[ct_column].isin([ct]), :]
        
        # Whether the current cell type is present in more than 50% of all donors?
        # ct_ratio = ann_sub.obs[donor_column].unique().shape[0]/donor_num
        # if ct_ratio >= 0.5:    
            
        # Generate filtered expression matrix and ages
        expr = ann_sub.to_df()
        expr.index = ann_sub.obs[donor_column].values
        expr.columns = ct + "--" + expr.columns
        exprs.append(expr)
        
    return exprs

# Function for extracting features from expression data
def compute_features(exprs, adata, mean_degree, var_degree):
    
    Xs = []

    # Aggregate expressions for each cell type
    for expr in exprs:

        # Get each subject/individual's mean expression and polynomials of mean expressions.
        subjects = expr.index.to_list()
        expr_mean = expr.groupby(subjects).mean()

        expr_mean_poly = np.hstack([expr_mean**i for i in range(1, mean_degree+1)])
        feature_names = np.hstack([[f"{col}_mean^{deg}" for col in expr.columns] for deg in range(1,mean_degree+1)]) # Name the polynomial features
        expr_mean_poly = pd.DataFrame(
                            expr_mean_poly,
                            index = expr_mean.index,
                            columns = feature_names 
                        )

        if var_degree > 0:

            # Get each subject/indivisual's variances.
            expr_var = expr.groupby(subjects).var()
            expr_var.fillna(value=0,inplace=True) # This should not happen 
            expr_var_poly = np.hstack([expr_var**i for i in range(1, var_degree+1)])
            feature_names = np.hstack([[f"{col}_var^{deg}" for col in expr.columns] for deg in range(1,var_degree+1)]) # Name the polynomial features
            expr_var_poly = pd.DataFrame(
                            expr_var_poly,
                            index = expr_var.index,
                            columns = feature_names 
                        )

            # Concatenate mean and var polynomial features
            X = pd.concat([expr_mean_poly,expr_var_poly],axis = 1)
        else:
            X = expr_mean_poly

        Xs.append(X)

    # Concat expressions for all cell types
    Xs = pd.concat(Xs, axis = 1)

    # Some cell types may not present in some donors. Mark these as zeros.
    Xs.fillna(0, inplace = True)
    
    # Get Y
    Y = adata.obs.loc[:,['donor_id','age']].drop_duplicates()
    Y.index = Y["donor_id"]
    del Y["donor_id"]
    Y = Y.loc[Xs.index,]
    
    return Xs,Y

# Generate expression data from anndata
def generate_data(ann, cts, ct_column, donor_column, age_column, marker_genes = 'union', min_cells=50, n_rep=5,mean_degree=2,var_degree=2,use_pca=True):
    warnings.filterwarnings("ignore")

    iterator = []
    n_rep = 5
    
    # If marker genes is not a list of gene names, read marker genes from file.
    if not isinstance(marker_genes, list):
        try:
            assert marker_genes in ['fridman', 'sasp2', 'senmayo', 'cellage', 'union', 'all']
        except AssertionError:
            print(f"'marker_genes' should be either one of 'fridman', 'sasp2', 'senmayo', 'cellage', 'union', 'all' or a list of gene names.")
        
        gene_type = marker_genes
        if marker_genes != 'all':
            marker_genes = read_marker_genes(marker_genes)
        else:
            marker_genes = ann.var_names
    else:
        gene_type = "custom"
       
    exprs = filter_anndata(ann,
                         ct_column = ct_column,
                         cts = cts,
                         donor_column = "donor_id",
                         age_column = "age",
                         marker_genes = marker_genes,
                         min_cells = 50
                        )
    
    X,Y = compute_features(exprs,
                           ann,
                           mean_degree=mean_degree,
                           var_degree=var_degree
                          )

    for rep in range(n_rep):
        iterator.append([X,Y,gene_type,use_pca,rep])
    del exprs
    
    return iterator

# Function for tunning hyperparameters
def tune(param, expr, ages):
    
    '''
    # Get Hyperparameters
    '''
    alpha = param['alpha']
    l1_ratio = param['l1_ratio']
    
    X_train = expr.copy()
    Y_train = ages.copy()
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
        
    # Center and scale Y.
    Y_train = Y_train.values
    Y_train = StandardScaler().fit_transform(Y_train.reshape(-1,1)).ravel()
    
    # Model fitting
    polyreg = make_pipeline(ElasticNet(max_iter=40000, alpha=alpha, l1_ratio=l1_ratio))
    polyreg.fit(X_train, Y_train)
        
    # Get predicted age for training data
    pred = polyreg.predict(X_train)
    
    # Return negative R2 as loss
    return(-r2_score(Y_train, pred))

# Function for train/test by leave-one-out test
def train_test_loo(X, Y, param_space, n_hyper_eval, n_components, rep, use_pca, n_jobs = 1):
    warnings.filterwarnings("ignore")
            
    subjects = X.index.unique()
    
    # Leave-one-out test
    Y_trues = []
    Y_preds = []
    
    def run_loo(subj):
        test_subjects = [subj]
        train_subjects = subjects[~subjects.isin(test_subjects)]
        
        X_train = X.loc[X.index.isin(train_subjects),]
        Y_train = Y.loc[Y.index.isin(train_subjects),]

        X_test = X.loc[X.index.isin(test_subjects),]
        Y_test = Y.loc[Y.index.isin(test_subjects),]
        
        # Note that PCA trained on training data was used on test data.
        if use_pca:
            pca_model = PCA(n_components=n_components, whiten=True)
            pca_model.fit(X_train)
            X_train = pd.DataFrame(pca_model.transform(X_train), index = X_train.index)
            X_train.columns = [f"PC{i+1}" for i in range(X_train.shape[1])]
            
            X_test = pd.DataFrame(pca_model.transform(X_test), index = X_test.index)
            X_test.columns = [f"PC{i+1}" for i in range(X_test.shape[1])]
        
        # Hyper params will not be tunned if param_space is None
        if param_space is not None:
            # Partial will freeze some arguments for tune()
            fmin_objective = partial(
                                    tune,
                                    expr=X_train,
                                    ages=Y_train
                                )

            # Search for the best hyperparameters on training data
            param_best = fmin(fmin_objective,
                                space = param_space,
                                algo=tpe.suggest,
                                max_evals=n_hyper_eval,
                                verbose = False
                             )
            param_best = space_eval(param_space, param_best)
        else:
            param_best = {'alpha':1, 'l1_ratio':0.5}
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Center and scale Y
        Y_train = Y_train.values
        Y_test = Y_test.values

        scaler = StandardScaler().fit(Y_train.reshape(-1,1))
        Y_train = scaler.transform(Y_train.reshape(-1,1)).ravel()
        Y_test = scaler.transform(Y_test.reshape(-1,1)).ravel()

        # Use the best hyperparameters to fit a model on training data
        polyreg = make_pipeline(ElasticNet(max_iter=40000, alpha=param_best['alpha'], l1_ratio=param_best['l1_ratio']))
        polyreg.fit(X_train, Y_train)

        # Get predicted ages for testing set
        Y_test = Y_test*np.sqrt(scaler.var_[0]) + scaler.mean_[0]
        Y_pred = polyreg.predict(X_test)*np.sqrt(scaler.var_[0]) + scaler.mean_[0]
        
        return Y_test, Y_pred
    
    res = Parallel(n_jobs=n_jobs)(delayed(run_loo)(subj) for subj in tqdm(subjects))
    for Y_test,Y_pred in res:
        Y_trues.extend(Y_test)
        Y_preds.extend(Y_pred)
        
    # Compute R2 and RMSE as evaluation metrics
    r2 = r2_score(Y_trues, Y_preds)
    rmse = mean_squared_error(Y_trues, Y_preds, squared = False)
    Y_trues = ",".join([str(age) for age in Y_trues])
    Y_preds = ",".join([str(age) for age in Y_preds])
    
    return r2,rmse,rep,use_pca,Y_trues,Y_preds

# Function for train/test by cross-validation
def train_test_kfold(X, Y, param_space, n_hyper_eval, n_components, rep, use_pca, k=10, val_ratio=0.2, n_jobs = 1):
    warnings.filterwarnings("ignore")
    
    subjects = X.index.unique()
    
    # K fold cross validation
    Y_trues = []
    Y_preds = []
    
    def run_kfold(train_index, test_index):
    #for train_index, test_index in tqdm(kf.split(subjects)):
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        train_subjets,val_subjects = train_test_split(train_subjects, test_size = val_ratio)
        
        X_train = X.loc[X.index.isin(train_subjects),]
        Y_train = Y.loc[Y.index.isin(train_subjects),]
        X_val = X.loc[X.index.isin(val_subjects),]
        Y_val = Y.loc[Y.index.isin(val_subjects),]

        X_test = X.loc[X.index.isin(test_subjects),]
        Y_test = Y.loc[Y.index.isin(test_subjects),]
        
        # Note that PCA trained on training data was used on test data.
        if use_pca:
            pca_model = PCA(n_components=n_components, whiten=True)
            pca_model.fit(X_train)
            X_train = pd.DataFrame(pca_model.transform(X_train), index = X_train.index)
            X_train.columns = [f"PC{i+1}" for i in range(X_train.shape[1])]
            
            X_test = pd.DataFrame(pca_model.transform(X_test), index = X_test.index)
            X_test.columns = [f"PC{i+1}" for i in range(X_test.shape[1])]
            
            X_val = pd.DataFrame(pca_model.transform(X_val), index = X_val.index)
            X_val.columns = [f"PC{i+1}" for i in range(X_val.shape[1])]

        # Partial will freeze some arguments for tune()
        fmin_objective = partial(
                                tune,
                                expr=X_val,
                                ages=Y_val,
                            )
        # Hyper params will not be tunned if param_space is None
        if param_space is not None:
            # Search for the best hyperparameters on validation data
            param_best = fmin(fmin_objective,
                                space = param_space,
                                algo=tpe.suggest,
                                max_evals=n_hyper_eval,
                                verbose = False
                             )
            param_best = space_eval(param_space, param_best)
        else:
            param_best = {'alpha':1, 'l1_ratio':0.5}

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Center and scale Y
        Y_train = Y_train.values
        Y_test = Y_test.values

        scaler = StandardScaler().fit(Y_train.reshape(-1,1))
        Y_train = scaler.transform(Y_train.reshape(-1,1)).ravel()
        Y_test = scaler.transform(Y_test.reshape(-1,1)).ravel()

        # Use the best hyperparameters to fit a model on training data
        polyreg = make_pipeline(ElasticNet(max_iter=40000, alpha=param_best['alpha'], l1_ratio=param_best['l1_ratio']))
        polyreg.fit(X_train, Y_train)

        # Get predicted ages for testing set
        Y_test = Y_test*np.sqrt(scaler.var_[0]) + scaler.mean_[0]
        Y_pred = polyreg.predict(X_test)*np.sqrt(scaler.var_[0]) + scaler.mean_[0]
        
        return Y_test, Y_pred
    
    kf = KFold(n_splits=k)
    res = Parallel(n_jobs=n_jobs)(delayed(run_kfold)(train_index, test_index) for train_index, test_index in tqdm(kf.split(subjects),total = k))
    
    for Y_test,Y_pred in res:
        Y_trues.extend(Y_test)
        Y_preds.extend(Y_pred)
        
    # Compute R2 and RMSE as evaluation metrics
    r2 = r2_score(Y_trues, Y_preds)
    rmse = mean_squared_error(Y_trues, Y_preds, squared = False)
    Y_trues = ",".join([str(age) for age in Y_trues])
    Y_preds = ",".join([str(age) for age in Y_preds])
    
    return r2,rmse,rep,use_pca,Y_trues,Y_preds

# Function for training polyEN model with full dataset
def train_full(X, Y, param_space, n_hyper_eval, n_components, rep, use_pca):
    warnings.filterwarnings("ignore")
            
    subjects = X.index.unique()
    X_train = X
    Y_train = Y
    
    if use_pca:
        pca_model = PCA(n_components=n_components, whiten=True)
        pca_model.fit(X_train)
        X_train = pd.DataFrame(pca_model.transform(X_train), index = X_train.index)
        X_train.columns = [f"PC{i+1}" for i in range(X_train.shape[1])]


    # Partial will freeze some arguments for tune()
    fmin_objective = partial(
                            tune,
                            expr=X_train,
                            ages=Y_train,
                        )
    
    # Hyper params will not be tunned if param_space is None
    if param_space is not None:
        # Search for the best hyperparameters on validation data
        param_best = fmin(fmin_objective,
                            space = param_space,
                            algo=tpe.suggest,
                            max_evals=n_hyper_eval,
                            verbose = False
                         )
        param_best = space_eval(param_space, param_best)
    else:
        param_best = {'alpha':1, 'l1_ratio':0.5}

    scaler_x = StandardScaler().fit(X_train)
    X_train = scaler_x.transform(X_train)

    # Center and scale Y
    Y_train = Y_train.values

    scaler_y = StandardScaler().fit(Y_train.reshape(-1,1))
    Y_train = scaler_y.transform(Y_train.reshape(-1,1)).ravel()

    # Use the best hyperparameters to fit a model on training data
    polyreg = polyEN([('en',ElasticNet(max_iter=40000, alpha = param_best['alpha'], l1_ratio=param_best['l1_ratio']))])
    polyreg.fit(X_train, Y_train)
    
    if use_pca:
        polyreg.add_pca_model(pca_model)
    polyreg.add_scaler_x(scaler_x)
    polyreg.add_scaler_y(scaler_y)
    
    return polyreg

# The polyEN class model inherited from Pipeline class of scikit-learn
# The predict() method is oeverriden for the original Pipeline class
class polyEN(sklearn.pipeline.Pipeline):
    
    def add_std_par(self, var, mean):
        self.var = var
        self.mean = mean
    
    def add_pca_model(self, pca):
        self.pca = pca
    
    def add_scaler_x(self, scaler):
        self.scaler_x = scaler
    
    def add_scaler_y(self, scaler):
        self.scaler_y = scaler
        
    def predict(self, X):
        X = self.scaler_x.fit_transform(X)
        if hasattr(self, 'pca'):
            X = self.pca.fit_transform(X)
        else:
            try:
                assert self['en'].coef_.shape[0] == X.shape[1]
            except AssertionError:
                print(f"Number of columns in 'X' should match the number of columns in training data")
        Y_pred = sklearn.pipeline.Pipeline.predict(self, X)
        return Y_pred * np.sqrt(self.scaler_y.var_[0]) + self.scaler_y.mean_[0]

# Train model with full dataset for the selected cell types of an anndata object
def polyEN_train_full(anndata, ct_column, cts, donor_column, age_column, param_space, mean_degree = 2, var_degree = 2, marker_genes = None, min_cells = 50, n_hyper_eval=30, n_components=10,use_pca=True):
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Generate training data ...")
    
    iterator = generate_data(ann = anndata,
                             cts = cts,
                             ct_column = ct_column,
                             donor_column = donor_column,
                             age_column = age_column,
                             marker_genes = marker_genes,
                             min_cells = min_cells,
                             n_rep = 1,
                             mean_degree=2,
                             var_degree=2,
                             use_pca=use_pca
                            )
    
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Done.")
    
    X,Y,gene_type,use_pca,rep = iterator[0]
    
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Training polyEN model...")
    model = train_full(X, Y, param_space, n_hyper_eval, n_components, rep, use_pca)
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Done.")
    return model

# Train model with kfold/leave-one-out cross validation for the selected cell types of an anndata object
def polyEN_train_cv(anndata, ct_column, cts, donor_column, age_column, param_space, mean_degree = 2, var_degree = 2, marker_genes = None, min_cells = 50, n_hyper_eval=30, n_components=10, use_pca=True, test_method = 'loo', k=10, val_ratio=0.2, n_jobs = 1):
    
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Generate training data ...")
    
    iterator = generate_data(ann = anndata,
                             cts = cts,
                             ct_column = ct_column,
                             donor_column = donor_column,
                             age_column = age_column,
                             marker_genes = marker_genes,
                             min_cells = min_cells,
                             n_rep = 1,
                             mean_degree=2,
                             var_degree=2,
                             use_pca=use_pca
                            )
    
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Done.")
    
    try:
        assert test_method in ["loo","kfold"]
    except AssertionError:
        print("'test_method' should be either 'loo' or 'kfold'")
    
    date = datetime.datetime.now()
    time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
    print(f"{time} Training and testing polyEN model...")
    
    test_results = []
    if test_method == "kfold":
        for X,Y,gene_type,use_pca,rep in iterator:
            date = datetime.datetime.now()
            time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
            print(f"{time} Training and testing for rep {rep+1}")
            r2,rmse,rep,use_pca,Y_trues,Y_preds = train_test_kfold(X,
                                                                   Y,
                                                                   param_space,
                                                                   n_hyper_eval,
                                                                   n_components,
                                                                   rep,
                                                                   use_pca,
                                                                   k=k,
                                                                   val_ratio=val_ratio
                                                                  )
            test_results.append([r2,rmse,rep,use_pca,Y_trues,Y_preds])
    else:
        for X,Y,gene_type,use_pca,rep in iterator:
            date = datetime.datetime.now()
            time = f"({date.month}/{date.day}/{date.year} {date.hour}:{date.minute}:{date.second})"
            print(f"{time} Training and testing for rep {rep+1}")
            r2,rmse,rep,use_pca,Y_trues,Y_preds = train_test_loo(X,
                                                                   Y,
                                                                   param_space,
                                                                   n_hyper_eval,
                                                                   n_components,
                                                                   rep,
                                                                   use_pca,
                                                                   n_jobs = n_jobs
                                                                  )
            test_results.append([r2,rmse,rep,use_pca,Y_trues,Y_preds])
    
    test_results = pd.DataFrame(test_results, columns = ["R2","RMSE","rep","PCA","age_true","age_pred"])
    return test_results