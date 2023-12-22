import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def modele_regression(X_train,y_train):
    """
    Création d'un modèle de régression logistique

    Parametres:
    ----------
    X_train : données d'entrainement pour les variables explicatives
    y_train : données d'entrainement pour la variable à expliquer
        
    Sortie :
    --------
    Modèle de régression logistique
    """

    modele = LogisticRegression(C=0.1)
    modele.fit(X_train, y_train)

    return modele

def modele_arbre(X_train,y_train):
    """
    Création d'un modèle d'arbre de décision

    Parametres:
    ----------
    X_train : données d'entrainement pour les variables explicatives
    y_train : données d'entrainement pour la variable à expliquer
        
    Sortie :
    --------
    Modèle d'un arbre de décision'
    """
    param_grid_dt = {'max_depth': [3, 5, 7]}
    modele = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
    modele.fit(X_train, y_train)

    return modele

def modele_xgboost(X_train,y_train):
    """
    Création d'un modèle xgboost

    Parametres:
    ----------
    X_train : données d'entrainement pour les variables explicatives
    y_train : données d'entrainement pour la variable à expliquer
        
    Sortie :
    --------
    Modèle d'xgboost'
    """
    param_grid_xgb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    modele = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5)
    modele.fit(X_train, y_train)

    return modele

def modele_randomforest(X_train,y_train):
    """
    Création d'un modèle de randomforest

    Parametres:
    ----------
    X_train : données d'entrainement pour les variables explicatives
    y_train : données d'entrainement pour la variable à expliquer
        
    Sortie :
    --------
    Modéle de randomforest
    """
    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    modele = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
    modele.fit(X_train, y_train)

    return modele

def modele_knn(X_train,y_train):
    """
    Création d'un modèle des k plus proches voisins

    Parametres:
    ----------
    X_train : données d'entrainement pour les variables explicatives
    y_train : données d'entrainement pour la variable à expliquer
        
    Sortie :
    --------
    Modéle des k plus proches voisins
    """
    param_grid_knn = {'n_neighbors': [3]}
    modele = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    modele.fit(X_train, y_train)

    return modele