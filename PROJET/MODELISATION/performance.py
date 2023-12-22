from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def evaluation_modele(name, model, X_test, y_test, threshold=0.5):
    """
    Mesure des performances du modèle avec différents indicateurs

    Parametres:
    ----------
    name : nom du modèle
    model : modèle crée
    X_test : données de test pour les variables explicatives
    y_test : données de test pour la variable à expliquer
    threshold : 

        
    Sortie :
    --------

    Affichage de l'Accuracy, Precision, Recall, Specificity, F1 Score, AUC, courbe roc et matrice de confusion
    """

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Rappel est synonyme de sensibilité
    f1 = f1_score(y_test, y_pred)

    # Calcul de la spécificité
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall/Sensitivity: {recall:.2f}, Specificity: {specificity:.2f}, F1 Score: {f1:.2f}, AUC: {roc_auc:.2f}\n")

    # Affichage de la matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title(f'Matrice de confusion pour {name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.show()

    # Affichage de la courbe ROC
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC pour {name}')
    plt.legend(loc="lower right")
    plt.show()




