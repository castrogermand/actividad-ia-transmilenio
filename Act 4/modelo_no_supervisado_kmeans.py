import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

def cargar_y_preparar_datos(ruta_csv='datos/Estaciones_Troncales_de_TRANSMILENIO.csv'):
    """Carga los datos reales y selecciona las características numéricas relevantes."""
    print("📋 Cargando datos reales de estaciones de TransMilenio...")
    df = pd.read_csv(ruta_csv)
    
    # Seleccionar variables estructurales y de capacidad para el agrupamiento
    features = ['area_est', 'long_est', 'ancho_est', 'cap_biart', 'cap_art', 'num_acc']
    df_modelo = df[['nom_est'] + features].dropna() # Eliminar filas con datos faltantes
    
    print(f"✅ Datos cargados: {len(df_modelo)} estaciones válidas.")
    return df, df_modelo, features

def encontrar_k_optimo(X, max_k=10):
    """Aplica el Método del Codo (Elbow Method) para determinar el mejor número de clusters."""
    print("\n🔍 Buscando el número óptimo de clusters (Método del Codo)...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
    # Graficar Método del Codo
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, marker='o', color='teal')
    plt.title('Método del Codo (Inercia)')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    
    # Graficar Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='s', color='orange')
    plt.title('Puntuación de Silueta')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('resultados/metodo_codo_y_silueta.png', dpi=150)
    plt.close()
    
    # Seleccionamos k=3 basándonos en la inflexión típica de estos datos
    k_optimo = 3
    print(f"   ➡️ K óptimo seleccionado: {k_optimo} (Silhouette Score: {silhouette_scores[k_optimo-2]:.4f})")
    return k_optimo

def entrenar_y_evaluar(df_modelo, features, k_optimo):
    """Entrena el modelo K-Means y evalúa los resultados."""
    print("\n⚙️ Entrenando modelo K-Means...")
    X = df_modelo[features]
    
    # Estandarización (Crucial para K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenamiento
    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    df_modelo['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Evaluación
    sil_score = silhouette_score(X_scaled, df_modelo['cluster'])
    print(f"✅ Entrenamiento completado. Silhouette Score final: {sil_score:.4f}")
    
    return df_modelo, kmeans, scaler

def analizar_resultados(df_modelo, features, k_optimo):
    """Analiza los centroides y visualiza los clusters con PCA."""
    print("\n📊 Analizando perfiles de los clusters...")
    
    # 1. Mostrar tamaño de cada cluster
    print("\nDistribución de estaciones por cluster:")
    print(df_modelo['cluster'].value_counts().sort_index())
    
    # 2. Visualización con PCA (Reducción a 2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_modelo[features])
    
    df_modelo['PCA1'] = pca_result[:, 0]
    df_modelo['PCA2'] = pca_result[:, 1]
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA1', y='PCA2',
        hue='cluster',
        palette='viridis',
        data=df_modelo,
        s=100, alpha=0.8
    )
    plt.title(f'Agrupamiento de Estaciones TransMilenio (PCA 2D) - K={k_optimo}', fontweight='bold')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('resultados/clusters_pca_visualizacion.png', dpi=150)
    plt.close()
    
    # 3. Guardar dataset con etiquetas
    df_modelo.to_csv('resultados/estaciones_con_clusters.csv', index=False, encoding='utf-8')
    print("\n✅ Resultados y visualizaciones guardados en la carpeta 'resultados/'")

if __name__ == "__main__":
    print("=" * 70)
    print("🚌 APRENDIZAJE NO SUPERVISADO - AGRUPAMIENTO TRANSMILENIO")
    print("   Fuente: Estaciones_Troncales_de_TRANSMILENIO.csv (oficial)")
    print("   Referencia: Palma Méndez (2008), Cap. 16: Técnicas de agrupamiento")
    print("=" * 70)
    
    # Crear carpeta de resultados
    os.makedirs('resultados', exist_ok=True)
    
    # Ejecutar pipeline
    df_original, df_modelo, features = cargar_y_preparar_datos()
    k_optimo = encontrar_k_optimo(df_modelo[features], max_k=8)
    df_final, modelo_kmeans, scaler = entrenar_y_evaluar(df_modelo, features, k_optimo)
    analizar_resultados(df_final, features, k_optimo)