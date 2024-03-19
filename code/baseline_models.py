import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



seed= 1337

#initialize elastic net model with specified parameters (hawkins)
def elastic_net(a, l):
    elastic =SGDClassifier(loss='log_loss', penalty='elasticnet',alpha = a, l1_ratio = l, random_state = seed)
    return elastic

#parameter tuning with two sets of parameters and a specified model
def param_tuning_elastic(X, y, param1, param2, folds):
    grid = dict()
    grid['alpha'] = param1
    grid['l1_ratio'] = param2
    cv = RepeatedKFold(n_splits= folds, n_repeats = 3, random_state = seed)
    search = GridSearchCV(SGDClassifier(loss='log_loss', penalty='elasticnet'), grid, scoring = 'f1', cv = cv)
    result = search.fit(X,y)
    return result.best_score_, result.best_params_

#fit an initialized model
def fit_model(model,X,y):
    return model.fit(X,y)

#predict on test set
def predict(fitted, X_gold):
    return X_gold.apply(lambda x: fitted.predict(x.values[None])[0], axis = 1)

#calculate evaluation metrics on predictions
def eval_metrics(y, yhat):
    print(f'The model reaches a recall of:{recall_score(y,yhat)}')
    print(f'The model reaches a precision of:{precision_score(y,yhat)}')
    print(f'The model reaches a F1-Score of:{f1_score(y,yhat)}')
    print(f'The model reaches an accuracy of:{accuracy_score(y,yhat)}')

#implement the dictionary by rooduijn and pauwels 2011
rood_dict_ger = ['elit', 'konsens', 'undemokratisch', 'referend', 'korrupt', 'propagand', 'politiker', 'täusch', 'betrüg', 'betrug', 'verrat', 'scham', 'schäm', 'skandal', 'wahrheit', 'unfair', 'unehrlich', 'establishm', 'herrsch', 'lüge']

#check if dictionary term is present and assign binary label according to that
def check_dict(df):
    dict_score = df.copy()
    dict = rood_dict_ger
    dict_score['Label'] = df['Comment'].str.lower().str.contains('|'.join(dict))
    dict_score['Label'].loc[dict_score['Label'] == True] = 1
    dict_score['Label'].loc[dict_score['Label'] == False] = 0
    dict_score['Label'] = dict_score['Label'].astype(int)
    return dict_score