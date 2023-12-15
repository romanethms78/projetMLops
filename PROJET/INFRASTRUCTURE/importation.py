import pandas as pd

def chargement(data) :
    """
    Chargement de la base de donnée pour une source fourni spécifique + définition du nom des variables

    Parametres:
    ----------
    data : str
        chemin du fichier

        
    Sortie :
    --------
    df : dataframe

    """

    df = pd.read_csv(data, header=None)

    df.columns = ["Age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "salaire"]

    return df