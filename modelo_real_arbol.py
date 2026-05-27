import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cargar_y_preparar():
    """Carga el dataset construido sobre datos REALES"""
    df = pd.read_csv('datos/transmilenio_dataset_real.csv')
    
    print(f"📋 Dataset cargado: {len(df)} viajes")
    print(f"   Variables originales: {list(df.columns)}")
    
    # Codificar target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['nivel_retraso'])
    
    # Variables numéricas REALES (extraídas del CSV oficial)
    numericas = [
        'distancia_km_real', 'area_origen_m2', 'area_destino_m2',
        'capacidad_origen', 'capacidad_destino', 'accesos_origen',
        'accesos_destino', 'misma_troncal', 'hora_viaje', 'dia_semana',
        'hora_pico', 'lluvia'
    ]
    
    # Variables categóricas (troncales reales)
    categoricas = ['troncal_origen', 'troncal_destino']
    df_enc = pd.get_dummies(df, columns=categoricas, drop_first=False)
    
    # Seleccionar features finales
    cols_cat_enc = [c for c in df_enc.columns if c.startswith(('troncal_origen_', 'troncal_destino_'))]
    X = df_enc[numericas + cols_cat_enc]
    y = df_enc['target']
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

def entrenar(X_train, y_train, X_test, y_test):
    """Entrena árbol con poda para evitar sobreajuste (Palma Méndez, cap. 17)"""
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 MÉTRICAS")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Profundidad real: {clf.get_depth()}")
    print(f"   Hojas: {clf.get_n_leaves()}")
    print(f"\n📋 REPORTE DETALLADO")
    print(classification_report(y_test, y_pred, 
                                target_names=['Bajo', 'Medio', 'Alto'], digits=3))
    return clf, y_pred

def guardar_resultados(clf, features, y_test, y_pred):
    """Genera gráficos y reglas"""
    os.makedirs('resultados', exist_ok=True)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bajo', 'Medio', 'Alto'],
                yticklabels=['Bajo', 'Medio', 'Alto'])
    plt.title('Matriz de Confusión - Dataset REAL', fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('resultados/matriz_confusion_real.png', dpi=150)
    plt.close()
    
    # Importancia de features
    plt.figure(figsize=(10, 8))
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.barh(range(len(imp)), imp[idx], align='center', color='teal')
    plt.yticks(range(len(imp)), [features[i] for i in idx])
    plt.title('Importancia de Variables (Gini) - Datos REALES', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('resultados/importancia_real.png', dpi=150)
    plt.close()
    
    # Árbol visual
    plt.figure(figsize=(18, 10))
    plot_tree(clf, feature_names=features, 
              class_names=['Bajo', 'Medio', 'Alto'],
              filled=True, rounded=True, fontsize=9, max_depth=3)
    plt.title('Árbol de Decisión - Primeros 3 Niveles', fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados/arbol_real.png', dpi=150)
    plt.close()
    
    # Reglas IF-THEN (Palma Méndez cap. 17)
    print("\n📜 REGLAS EXTRAÍDAS (Palma Méndez, Cap. 17)")
    print("=" * 60)
    reglas = export_text(clf, feature_names=features, max_depth=4)
    print(reglas)
    
    with open('resultados/reglas_extraidas.txt', 'w', encoding='utf-8') as f:
        f.write(reglas)
    
    print("\n✅ Archivos guardados en carpeta 'resultados/'")

if __name__ == "__main__":
    print("=" * 70)
    print("🚌 APRENDIZAJE SUPERVISADO - DATOS REALES TRANSMILENIO")
    print("   Fuente: Estaciones_Troncales_de_TRANSMILENIO.csv (oficial)")
    print("   Referencia: Palma Méndez (2008), Cap. 17")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, features = cargar_y_preparar()
    clf, y_pred = entrenar(X_train, y_train, X_test, y_test)
    guardar_resultados(clf, features, y_test, y_pred)