import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('D:\DBS Foundation Bootcamp\Machine Learning Terapan\predictive-analysis\dataset\heart_attack_prediction_indonesia.csv')

df.info()

df.head()

df = df.drop(['region', 'income_level', 'family_history', 'alcohol_consumption', 'EKG_results', 'participated_in_free_screening'], axis=1)
df.info()

df.head()

binary_categorical_cols = [
    'hypertension', 'diabetes', 'obesity', 
    'previous_heart_disease', 'medication_usage', 'heart_attack'
]

for col in binary_categorical_cols:
    df[col] = df[col].astype('category')

df.info()

# Check for missing values
df.isnull().sum()

numeric_features = [
    'age',
    'cholesterol_level',
    'waist_circumference',
    'sleep_hours',
    'blood_pressure_systolic',
    'blood_pressure_diastolic',
    'fasting_blood_sugar',
    'cholesterol_hdl',
    'cholesterol_ldl',
    'triglycerides'
]

categorical_features = [
    'gender',
    'smoking_status',
    'physical_activity',
    'dietary_habits',
    'air_pollution_exposure',
    'stress_level',
    'hypertension',
    'diabetes',
    'obesity',
    'previous_heart_disease',
    'medication_usage'
]

# Fungsi deteksi outlier dengan IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Simpan outlier untuk tiap kolom
outlier_summary = {col: detect_outliers_iqr(df, col) for col in numeric_features}

# Buat visualisasi
n_features = len(numeric_features)
n_cols = 3  # Misalnya 3 kolom per baris
n_rows = int(np.ceil(n_features / n_cols))

plt.figure(figsize=(n_cols * 5, n_rows * 4))

for i, col in enumerate(numeric_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[col], kde=True, color='skyblue', bins=30, label='All Data')

    # Plot outliers jika ada
    if not outlier_summary[col].empty:
        sns.scatterplot(
            x=outlier_summary[col][col],
            y=[0.5] * len(outlier_summary[col]),
            color='red',
            label='Outliers',
            marker='x'
        )

    plt.title(f'Distribution with Outliers: {col}')
    plt.xlabel(col)
    plt.legend()

plt.tight_layout()
plt.show()

def remove_outliers_iqr(data, columns):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

df_no_outliers = remove_outliers_iqr(df, numeric_features)

# Buat subplot untuk data sebelum & sesudah
n_features = len(numeric_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

plt.figure(figsize=(n_cols * 5, n_rows * 7))

for i, col in enumerate(numeric_features, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Histogram sebelum
    sns.histplot(df[col], kde=True, color='lightcoral', bins=30, label='Before')
    
    # Histogram sesudah
    sns.histplot(df_no_outliers[col], kde=True, color='seagreen', bins=30, label='After', alpha=0.6)

    plt.title(f'Before vs After Outlier Removal: {col}')
    plt.xlabel(col)
    plt.legend()

plt.tight_layout()
plt.show()

import math

n_features = len(categorical_features)
n_cols = 3  # kamu bisa atur sesuai lebar yang diinginkan
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
axes = axes.flatten()  # agar indexing jadi 1 dimensi

for i, col in enumerate(categorical_features):
    sns.countplot(x=col, data=df_no_outliers, palette='Set2', ax=axes[i])
    axes[i].set_title(f'Distribusi Kategori: {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Untuk subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

n_features = len(numeric_features)
n_cols = 3  # atur sesuai selera, misal 3 kolom
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.histplot(df_no_outliers[col], kde=True, bins=30, color='skyblue', ax=axes[i])
    axes[i].set_title(f'Distribusi Numerik: {col}')
    axes[i].set_xlabel(col)

# Hapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

sns.set(style="whitegrid")

n_features = len(categorical_features)
n_cols = 3  # bisa diubah
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(data=df_no_outliers, x=col, hue="heart_attack", palette='Set2', ax=axes[i])
    axes[i].set_title(f'Distribution of Heart Attack by {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)

# Hapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

n_features = len(categorical_features)
n_cols = 3  # atur sesuai keinginan
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='heart_attack', data=df_no_outliers, palette='Set1', ax=axes[i])
    axes[i].set_title(f'Heart Attack by {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Menghapus subplot kosong jika ada
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

corr = df_no_outliers.select_dtypes(include=['number']).corr()
print(corr)

plt.figure(figsize=(10, 8))
correlation_matrix = df_no_outliers[numeric_features].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix - Numeric Features')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import OneHotEncoder

# Buat encoder
ohe = OneHotEncoder(drop='first', sparse_output=False)

# Encode fitur kategorikal
encoded_cat = ohe.fit_transform(df[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(categorical_features), index=df.index)

encoded_cat_df.head()

X_raw = pd.concat([df[numeric_features], encoded_cat_df], axis=1)
y = df['heart_attack'].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# PCA pada data training (bisa juga pada gabungan train+test kalau hanya untuk visualisasi)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Buat dataframe untuk plot
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['heart_attack'] = y_train.values

# Pairplot
sns.pairplot(pca_df, hue='heart_attack', palette='Set1')
plt.suptitle("PCA Pairplot (2 Komponen Utama)", y=1.02)
plt.show()

logreg_balanced = LogisticRegression(class_weight='balanced', max_iter=500)
logreg_balanced.fit(X_pca, y_train)
y_pred_balanced = logreg_balanced.predict(X_pca)

print("Logistic Regression dengan class_weight='balanced':")
print(classification_report(y_train, y_pred_balanced))
print(f"Accuracy: {accuracy_score(y_train, y_pred_balanced):.4f}")

from sklearn.svm import SVC

# 2. Support Vector Machine (SVM)
svm = SVC(kernel='linear')
svm.fit(X_pca, y_train)
y_pred_svm = svm.predict(X_pca)
print("Support Vector Machine:")
print(classification_report(y_train, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_train, y_pred_svm):.4f}")
print("-"*40)

from sklearn.neighbors import KNeighborsClassifier

# 3. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_pca, y_train)
y_pred_knn = knn.predict(X_pca)
print("K-Nearest Neighbors:")
print(classification_report(y_train, y_pred_knn))
print(f"Accuracy: {accuracy_score(y_train, y_pred_knn):.4f}")
print("-"*40)



# 4. XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Suppress warning
xgb.fit(X_pca, y_train)
y_pred_xgb = xgb.predict(X_pca)

print("XGBoost Classifier:")
print(classification_report(y_train, y_pred_xgb))
print(f"Accuracy: {accuracy_score(y_train, y_pred_xgb):.4f}")
print("-"*40)