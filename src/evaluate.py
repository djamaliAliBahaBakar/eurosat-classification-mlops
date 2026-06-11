from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import itertools

def evaluate(model,test_ds_prepared, class_names):
    loss, accuracy, top2_accuracy = model.evaluate(test_ds_prepared, verbose=0)

    y_true, y_pred = get_predictions(model=model, test_ds_prepared=test_ds_prepared)
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    report = build_classification_report(y_true, y_pred, class_names)
    return {
        "loss":loss, 
        "accuracy": accuracy,
        "confusion_matrix": cm, 
        "classification_report": report,
        "top2_accuracy": top2_accuracy
    }

def get_predictions(model, test_ds_prepared):
    y_true = []
    y_pred = []

    for images, labels in test_ds_prepared:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true, y_pred


def build_classification_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names, digits=3)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion - Modèle Fine-Tuned', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Ajouter les valeurs dans les cellules
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold')

    plt.ylabel('Vraie Classe', fontsize=13, fontweight='bold')
    plt.xlabel('Classe Prédite', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()