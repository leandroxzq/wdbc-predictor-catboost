from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def evaluate_model(name, y_true, y_pred, y_probs, results):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    results.append([name, specificity, sensitivity, accuracy, auc])
    return results
