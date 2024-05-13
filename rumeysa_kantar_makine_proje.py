import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
data = pd.read_csv('expanded_campaign_responses.csv')

# Kategorik değişkenleri sayısallaştırma
label_encoders = {}
categorical_columns = ['gender', 'employed', 'marital_status', 'responded']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Bağımsız değişkenler ve bağımlı değişkeni ayırma
X = data.drop(['customer_id', 'responded'], axis=1)
y = data['responded']

# Sayısal özellikler için normalizasyon
scaler = MinMaxScaler()
numerical_columns = ['age', 'annual_income', 'credit_score', 'no_of_children']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Özellik Seçimi
selector = SelectKBest(score_func=chi2, k=4)
X_selected_univariate = selector.fit_transform(X, y)

rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=4, step=1)
X_selected_rfe = rfe_selector.fit_transform(X, y)

forest = RandomForestClassifier()
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
X_selected_rf = X.iloc[:, indices[:4]]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_uni, X_test_uni = X_train.iloc[:, selector.get_support()], X_test.iloc[:, selector.get_support()]
X_train_rfe, X_test_rfe = X_train.iloc[:, rfe_selector.support_], X_test.iloc[:, rfe_selector.support_]
X_train_rf, X_test_rf = X_train.iloc[:, indices[:4]], X_test.iloc[:, indices[:4]]

# Model isimleri ve model nesneleri
models = {
    "Lojistik Regresyon": LogisticRegression(random_state=42),
    "Karar Ağacı": DecisionTreeClassifier(random_state=42),
    "K-NN": KNeighborsClassifier(n_neighbors=3)
}

# Performans ölçütlerini ve model bilgilerini depolamak için boş bir DataFrame oluşturma
performance_data = pd.DataFrame(columns=["Feature Set", "Model", "Accuracy", "Precision", "Recall"])

# Model eğitimi ve performans değerlendirme
for name, model in models.items():
    for X_train_set, X_test_set, set_name in zip([X_train, X_train_uni, X_train_rfe, X_train_rf],
                                                 [X_test, X_test_uni, X_test_rfe, X_test_rf],
                                                 ["All Features", "Univariate", "RFE", "Random Forest"]):
        model.fit(X_train_set, y_train)
        y_pred = model.predict(X_test_set)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        performance_row = pd.DataFrame({
            "Feature Set": [set_name],
            "Model": [name],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall]
        })
        # Boş ve tamamen NA olan sütunları her bir DataFrame oluşturulduğunda kaldırın
        clean_performance_row = performance_row.dropna(axis=1, how='all').dropna(axis=0, how='all')
        # DataFrame'e ekleyin
        performance_data = pd.concat([performance_data, clean_performance_row], ignore_index=True)

# Sonuçları yazdırma
print(performance_data)
