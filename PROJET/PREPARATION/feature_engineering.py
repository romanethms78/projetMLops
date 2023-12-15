import pandas as pd

def supp(data):
    """
    Suppression des variables non nécessaires au traitement du sujet

    Parametres:
    ----------
    data : df
        dataframe

        
    Sortie :
    --------

    df : dataframe
        base avec les variables supprimées
    """
    data = data.drop(columns= "relationship")
    data = data.drop(columns= "education_num")
    data = data.drop(columns= "fnlwgt")

    return data

def dummies(data):
    """
    Transformation de la base en dummies

    Parametres:
    ----------
    data : df
        dataframe

        
    Sortie :
    --------

    df : dataframe
        base transformée en dummies
    """
    data = pd.get_dummies(data)

    return data
