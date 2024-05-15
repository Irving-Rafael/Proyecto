import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
zoo_data = pd.read_csv("zoo2.csv")

# Separar características y etiquetas
X = zoo_data.drop(columns=["animal_name", "class_type"])
y = zoo_data["class_type"]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Función para evaluar el modelo
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

# Regresión Logística
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)
logistic_accuracy, logistic_precision, logistic_recall, logistic_f1, logistic_confusion = evaluate_model(y_test, logistic_predictions)

# K-Vecinos Cercanos
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_accuracy, knn_precision, knn_recall, knn_f1, knn_confusion = evaluate_model(y_test, knn_predictions)

# Máquinas de Vector Soporte
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy, svm_precision, svm_recall, svm_f1, svm_confusion = evaluate_model(y_test, svm_predictions)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_predictions = nb_model.predict(X_test_scaled)
nb_accuracy, nb_precision, nb_recall, nb_f1, nb_confusion = evaluate_model(y_test, nb_predictions)

# Resultados para gráficos
models = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machines', 'Naive Bayes']
accuracies = [logistic_accuracy, knn_accuracy, svm_accuracy, nb_accuracy]
precisions = [logistic_precision, knn_precision, svm_precision, nb_precision]
recalls = [logistic_recall, knn_recall, svm_recall, nb_recall]
f1_scores = [logistic_f1, knn_f1, svm_f1, nb_f1]

# Impresión de resultados en la terminal
print("Resultados:")
for model, accuracy, precision, recall, f1 in zip(models, accuracies, precisions, recalls, f1_scores):
    print(f"\n{model}:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Determinar el método con la mayor precisión
best_method_index = accuracies.index(max(accuracies))
best_method = models[best_method_index]
print("\nEl método más eficaz fue:", best_method)

# Gráfico de resultados
barWidth = 0.2
r1 = range(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 8))
plt.bar(r1, accuracies, color='skyblue', width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, precisions, color='salmon', width=barWidth, edgecolor='grey', label='Precision')
plt.bar(r3, recalls, color='lightgreen', width=barWidth, edgecolor='grey', label='Recall')
plt.bar(r4, f1_scores, color='orange', width=barWidth, edgecolor='grey', label='F1 Score')

plt.xlabel('Modelos', fontweight='bold')
plt.xticks([r + barWidth * 1.5 for r in range(len(models))], models)
plt.title('Métricas de Evaluación para Diferentes Modelos')
plt.legend()
plt.show()
