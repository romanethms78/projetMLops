import pandas as pd
import numpy as np


def nettoyage(data) : 
    """
    Suppression des données manquantes

    Parametres:
    ----------
    data : df
        dataframe

        
    Sortie :
    --------

    df : dataframe
        base nettoyée
    """

    data = data[data["native_country"] != ' ?']
    data = data[data["workclass"] != ' ?']
    data = data[data["occupation"] != ' ?']

    return data

def regroupement(data):
    """
    Regroupement des modalités des variables

    Parametres:
    ----------
    data : df
        dataframe

        
    Sortie :
    --------

    df : dataframe
        base regroupée
    """

    data["capital_gain"] = np.where(data["capital_gain"] > 0, '1', '0')
    data["capital_loss"] = np.where(data["capital_loss"] > 0, '1','0')



    def recode_age(age):
        if age <= 25:
            return "moins_de_26_ans"
        elif 26 <= age <= 35:
            return "26_35_ans"
        elif 36 <= age <= 45:
            return "36_45_ans"
        elif 46 <= age <= 55:
            return "46_55_ans"
        else:
            return "plus_de_55_ans"

    data["Age"] = data["Age"].apply(recode_age)


    def recode_hours_per_week(hours_per_week):
        if hours_per_week <= 15:
            return "inf15_h"
        elif 16 <= hours_per_week <= 30:
            return "entre_16_et_30h"
        elif 31 <= hours_per_week <= 45:
            return "entre_31_et_45h"
        else:
            return "plus_45h"
    data["hours_per_week"] = data["hours_per_week"].apply(recode_hours_per_week)


    data.loc[data['salaire']==' >50K', ['salaire']] =1
    data.loc[data['salaire']==' <=50K', ['salaire']] =0


    workclass_mapping = {
        " Federal-gov": "government",
        " Local-gov": "government",
        " State-gov": "government",
        " Self-emp-not-inc": "independant",
        " Self-emp-inc": "independant"
    }
    data['workclass'] = data['workclass'].apply(lambda x: workclass_mapping.get(x, x))



    education_mapping = {
        " Preschool": "Primaire",
        " 1st-4th": "Primaire",
        " 5th-6th": "College",
        " 7th-8th": "College",
        " 9th": "Lycee",
        " 10th": "Lycee",
        " 11th": "Lycee",
        " 12th": "Lycee",
        " HS-grad": "Lycee",
        " Assoc-acdm": "Etudes_sup",
        " Assoc-voc": "Etudes_sup",
        " Bachelors": "Etudes_sup",
        " Masters": "Etudes_sup",
        " Doctorate": "Etudes_sup",
        " Some-college": "Etudes_sup",
        " Prof-school" : "Etudes_sup"
    }
    data['education'] = data['education'].map(education_mapping).fillna(data['education'])



    marital_status_mapping = {
        " Married-civ-spouse": "Married",
        " Married-spouse-absent": "Married",
        " Married-AF-spouse": "Married"
    }

    # Appliquer le mappage
    data['marital_status'] = data['marital_status'].replace(marital_status_mapping)



    native_country_mapping = {
        " United-States": "North_America",
        " Outlying-US(Guam-USVI-etc)": "North_America",
        " Puerto-Rico": "North_America",
        " Canada": "North_America",
        " Cuba": "North_America",
        " Honduras": "North_America",
        " Jamaica": "North_America",
        " Mexico": "North_America",
        " Dominican-Republic": "North_America",
        " Haiti": "North_America",
        " Guatemala": "North_America",
        " Nicaragua": "North_America",
        " El-Salvador": "North_America",
        " Ecuador": "South_America",
        " Columbia": "South_America",
        " Trinadad&Tobago": "South_America",
        " Peru": "South_America",
        " Cambodia": "Asia",
        " India": "Asia",
        " Japan": "Asia",
        " China": "Asia",
        " Philippines": "Asia",
        " Vietnam": "Asia",
        " Laos": "Asia",
        " Taiwan": "Asia",
        " Thailand": "Asia",
        " Hong": "Asia",
        " Iran": "Asia",
        " England": "Europe",
        " Germany": "Europe",
        " Greece": "Europe",
        " Italy": "Europe",
        " Poland": "Europe",
        " Portugal": "Europe",
        " Ireland": "Europe",
        " France": "Europe",
        " Hungary": "Europe",
        " Scotland": "Europe",
        " Yugoslavia": "Europe",
        " Holand-Netherlands": "Europe",
        " South": "Africa",
        " ?": "North_America"
    }
    data['native_country'] = data['native_country'].map(native_country_mapping).fillna(data['native_country'])

    return data