#!/usr/bin/env python3
"""
Fraud Detection Model - Advanced Implementation
===============================================

Autor: Data Scientist Senior
Descripción: Implementación mejorada para detección de fraude que aborda las 
            debilidades del modelo original y proporciona una solución robusta.

Este script implementa las 4 fases del análisis:
1. Crítica Constructiva y Plan de Acción
2. Preprocesamiento de Datos Avanzado y Feature Engineering  
3. Modelado y Optimización
4. Evaluación Rigurosa del Modelo y Conclusiones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, accuracy_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FRAUD DETECTION MODEL - ADVANCED IMPLEMENTATION")
print("="*80)

# =====================================================================
# FASE 1: CRÍTICA CONSTRUCTIVA Y PLAN DE ACCIÓN
# =====================================================================

print("\n" + "="*60)
print("FASE 1: CRÍTICA CONSTRUCTIVA Y PLAN DE ACCIÓN")
print("="*60)

print("""
ANÁLISIS CRÍTICO DE LA METODOLOGÍA ACTUAL:

1. MANEJO INADECUADO DE VALORES NULOS:
   ❌ Problema: Reemplazar todos los valores NaN por 0
   🔍 Por qué es problemático: 
      - Introduce sesgo artificial en el modelo
      - Puede crear patrones falsos donde no los hay
      - Diferentes variables requieren estrategias de imputación distintas
   ✅ Solución: Imputación sofisticada usando KNNImputer o estrategias específicas

2. CODIFICACIÓN INCORRECTA DE VARIABLES CATEGÓRICAS:
   ❌ Problema: Asignar "pesos" numéricos arbitrarios a variables de texto
   🔍 Por qué es problemático:
      - Crea relaciones ordinales artificiales que no existen
      - El modelo interpreta incorrectamente la distancia entre categorías
      - Puede llevr a conclusiones erróneas sobre importancia de features
   ✅ Solución: One-Hot Encoding estándar

3. MÉTRICAS DE EVALUACIÓN INSUFICIENTES:
   ❌ Problema: Solo usar precisión (accuracy) como métrica
   🔍 Por qué es problemático para fraude:
      - En datasets desbalanceados, accuracy puede ser muy engañosa
      - Un modelo que clasifica todo como "no fraude" podría tener 80% accuracy
      - No mide la capacidad real de detectar fraudes (recall)
   ✅ Solución: Métricas completas - Precision, Recall, F1-Score, ROC-AUC

4. FALTA DE MANEJO DEL DESBALANCE DE CLASES:
   ❌ Problema: No abordar el desbalance 63.5% vs 36.5%
   🔍 Por qué es problemático:
      - El modelo se sesga hacia la clase mayoritaria
      - Puede tener buen accuracy pero muy mal recall para fraudes
   ✅ Solución: SMOTE para balancear las clases

5. AUSENCIA DE PREPROCESAMIENTO AVANZADO:
   ❌ Problema: No escalar variables, no analizar correlaciones
   🔍 Por qué es problemático:
      - Algoritmos como Logistic Regression son sensibles a la escala
      - Variables con rangos diferentes dominan el modelo
   ✅ Solución: StandardScaler y análisis de features

PLAN DE ACCIÓN ESTRUCTURADO:
1. EDA completo con visualizaciones
2. Imputación avanzada de valores faltantes  
3. Codificación apropiada de categóricas
4. Escalado de features numéricas
5. Manejo del desbalance con SMOTE
6. Comparación de múltiples algoritmos
7. Optimización de hiperparámetros
8. Evaluación rigurosa con métricas de negocio
""")

# =====================================================================
# FASE 2: PREPROCESAMIENTO DE DATOS AVANZADO Y FEATURE ENGINEERING
# =====================================================================

print("\n" + "="*60)
print("FASE 2: PREPROCESAMIENTO DE DATOS AVANZADO Y FEATURE ENGINEERING")
print("="*60)

# Cargar datos
print("📊 Cargando datasets...")
df_train = pd.read_excel('entrenamiento_fraude.xlsx')
df_test = pd.read_excel('testeo_fraude.xlsx')
df_eval = pd.read_excel('base_evaluada.xlsx')

print(f"✅ Dataset entrenamiento: {df_train.shape}")
print(f"✅ Dataset testeo: {df_test.shape}")  
print(f"✅ Dataset evaluación: {df_eval.shape}")

# EDA Inicial
print("\n📈 ANÁLISIS EXPLORATORIO INICIAL:")
print(f"Variables totales: {len(df_train.columns)}")
print(f"Variables numéricas: {len(df_train.select_dtypes(include=[np.number]).columns)}")
print(f"Variables categóricas: {len(df_train.select_dtypes(include=['object']).columns)}")
print(f"Valores faltantes totales: {df_train.isnull().sum().sum()}")

# Análisis del desbalance de clases
print("\n⚖️ ANÁLISIS DEL BALANCE DE CLASES:")
class_distribution = df_train['fraude'].value_counts()
print(f"Clase 0 (No Fraude): {class_distribution[0]} ({class_distribution[0]/len(df_train)*100:.1f}%)")
print(f"Clase 1 (Fraude): {class_distribution[1]} ({class_distribution[1]/len(df_train)*100:.1f}%)")
print(f"Ratio de desbalance: {class_distribution[0]/class_distribution[1]:.2f}:1")

if class_distribution[0]/class_distribution[1] > 1.5:
    print("⚠️ DATASET DESBALANCEADO DETECTADO - Se aplicará SMOTE")
else:
    print("✅ Dataset relativamente balanceado")

# Identificar variables categóricas
categorical_features = ['descri_apli_prod_ben', 'marca_timeout', 'marca_host_no_resp']
numerical_features = [col for col in df_train.columns 
                     if col not in categorical_features + ['radicado', 'fraude']]

print(f"\n🔤 Variables categóricas identificadas: {categorical_features}")
print(f"🔢 Variables numéricas: {len(numerical_features)}")

# Análisis de valores faltantes por tipo de variable
print("\n🕳️ ANÁLISIS DE VALORES FALTANTES:")
missing_analysis = df_train.isnull().sum()
missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)

for col, missing_count in missing_analysis.head(10).items():
    missing_pct = missing_count / len(df_train) * 100
    col_type = "categórica" if col in categorical_features else "numérica"
    print(f"   {col} ({col_type}): {missing_count} ({missing_pct:.1f}%)")

# Separar features y target
X = df_train.drop(['radicado', 'fraude'], axis=1)
y = df_train['fraude']

print("\n🔧 APLICANDO PREPROCESAMIENTO AVANZADO...")

# 1. Estrategia de Imputación Sofisticada
print("1️⃣ Imputación avanzada de valores faltantes:")
print("   • Variables numéricas: KNNImputer (k=5) - más sofisticado que mediana")
print("   • Variables categóricas: Moda (valor más frecuente)")

# Para categóricas: imputar con moda
for col in categorical_features:
    if col in X.columns:
        mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
        X[col] = X[col].fillna(mode_value)
        print(f"   ✅ {col}: {X[col].isnull().sum()} valores faltantes restantes")

# Para numéricas: KNNImputer
numerical_cols_in_X = [col for col in numerical_features if col in X.columns]
if len(numerical_cols_in_X) > 0:
    knn_imputer = KNNImputer(n_neighbors=5)
    X[numerical_cols_in_X] = knn_imputer.fit_transform(X[numerical_cols_in_X])
    print(f"   ✅ Variables numéricas: KNNImputer aplicado a {len(numerical_cols_in_X)} columnas")

print(f"   ✅ Valores faltantes restantes: {X.isnull().sum().sum()}")

# 2. One-Hot Encoding para variables categóricas
print("\n2️⃣ One-Hot Encoding para variables categóricas:")
print("   🎯 Ventajas sobre asignación de pesos:")
print("      • No crea relaciones ordinales artificiales")
print("      • Cada categoría es independiente")
print("      • Interpretabilidad clara del modelo")

X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, drop_first=True)
print(f"   ✅ Dimensiones después de encoding: {X_encoded.shape}")
print(f"   ✅ Nuevas columnas categóricas creadas: {X_encoded.shape[1] - len(numerical_cols_in_X)}")

# 3. Escalado de características
print("\n3️⃣ Escalado de características numéricas:")
print("   🎯 Importancia del escalado:")
print("      • Algoritmos como Logistic Regression son sensibles a la escala")
print("      • Mejora convergencia y performance")
print("      • Evita que variables con rangos grandes dominen el modelo")

scaler = StandardScaler()
numerical_cols_encoded = [col for col in X_encoded.columns if col in numerical_cols_in_X]
X_scaled = X_encoded.copy()
X_scaled[numerical_cols_encoded] = scaler.fit_transform(X_encoded[numerical_cols_encoded])
print(f"   ✅ {len(numerical_cols_encoded)} variables numéricas escaladas")

# 4. División de datos
print("\n4️⃣ División estratificada de datos:")
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Entrenamiento: {X_train.shape[0]} muestras")
print(f"   ✅ Validación: {X_val.shape[0]} muestras")
print(f"   ✅ Distribución mantenida en ambos conjuntos")

# 5. Manejo del desbalance con SMOTE
print("\n5️⃣ Manejo del desbalance de clases con SMOTE:")
print("   🎯 SMOTE (Synthetic Minority Over-sampling Technique):")
print("      • Genera muestras sintéticas de la clase minoritaria")
print("      • Mejor que duplicar datos existentes")
print("      • Solo se aplica en entrenamiento, NO en validación")

original_distribution = pd.Series(y_train).value_counts()
print(f"   📊 Distribución original: {dict(original_distribution)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

balanced_distribution = pd.Series(y_train_balanced).value_counts()
print(f"   📊 Distribución después de SMOTE: {dict(balanced_distribution)}")
print(f"   ✅ Dataset balanceado: {X_train_balanced.shape[0]} muestras totales")

print("\n🎉 PREPROCESAMIENTO COMPLETADO:")
print(f"   • Datos finales entrenamiento: {X_train_balanced.shape}")
print(f"   • Features totales: {X_train_balanced.shape[1]}")
print(f"   • Clases balanceadas: ✅")
print(f"   • Datos escalados: ✅")
print(f"   • Variables categóricas codificadas: ✅")

# =====================================================================
# FASE 3: MODELADO Y OPTIMIZACIÓN  
# =====================================================================

print("\n" + "="*60)
print("FASE 3: MODELADO Y OPTIMIZACIÓN")
print("="*60)

print("\n🤖 SELECCIÓN Y COMPARACIÓN DE ALGORITMOS:")
print("   🎯 Por qué múltiples algoritmos:")
print("      • Diferentes algoritmos capturan diferentes patrones")
print("      • Permite encontrar el mejor para este problema específico")
print("      • Reduce overfitting y aumenta robustez")

# Definir modelos a comparar
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

print(f"\n📋 Algoritmos a evaluar:")
for name, model in models.items():
    print(f"   • {name}")

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\n🔄 Estrategia de validación cruzada: {cv_strategy.n_splits}-fold estratificada")

# Evaluar modelos base
print("\n🏃‍♂️ EVALUACIÓN INICIAL DE MODELOS (sin optimizar):")
model_results = {}

for name, model in models.items():
    print(f"\n   🔍 Evaluando {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                               cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    
    # Fit model and predict on validation
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    model_results[name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'model': model
    }
    
    print(f"      CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"      Validación - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Encontrar el mejor modelo base
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_auc_mean'])
print(f"\n🏆 MEJOR MODELO BASE: {best_model_name}")
print(f"   AUC promedio en CV: {model_results[best_model_name]['cv_auc_mean']:.4f}")

# Optimización de hiperparámetros para el mejor modelo
print(f"\n⚙️ OPTIMIZACIÓN DE HIPERPARÁMETROS - {best_model_name}:")
print("   🎯 GridSearchCV para encontrar la mejor combinación de parámetros")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
else:  # XGBoost
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

# GridSearch
print(f"   🔍 Buscando entre {len(list(ParameterGrid(param_grid)))} combinaciones...")
best_base_model = models[best_model_name]
grid_search = GridSearchCV(
    best_base_model, param_grid, cv=cv_strategy, 
    scoring='roc_auc', n_jobs=-1, verbose=0
)

grid_search.fit(X_train_balanced, y_train_balanced)
best_model_optimized = grid_search.best_estimator_

print(f"   ✅ Mejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"      {param}: {value}")
print(f"   ✅ Mejor AUC en CV: {grid_search.best_score_:.4f}")

# =====================================================================
# FASE 4: EVALUACIÓN RIGUROSA DEL MODELO Y CONCLUSIONES
# =====================================================================

print("\n" + "="*60)
print("FASE 4: EVALUACIÓN RIGUROSA DEL MODELO Y CONCLUSIONES")
print("="*60)

print("\n📊 EVALUACIÓN COMPLETA DEL MODELO OPTIMIZADO:")
print("   🎯 Métricas esenciales para detección de fraude:")
print("      • Precision: ¿Qué % de alertas de fraude son realmente fraude?")
print("      • Recall: ¿Qué % de fraudes reales detectamos?")
print("      • F1-Score: Balance entre Precision y Recall")
print("      • ROC-AUC: Capacidad discriminativa general")

# Predicciones finales
y_pred_final = best_model_optimized.predict(X_val)
y_pred_proba_final = best_model_optimized.predict_proba(X_val)[:, 1]

# Métricas finales
final_accuracy = accuracy_score(y_val, y_pred_final)
final_precision = precision_score(y_val, y_pred_final)
final_recall = recall_score(y_val, y_pred_final)
final_f1 = f1_score(y_val, y_pred_final)
final_auc = roc_auc_score(y_val, y_pred_proba_final)

print(f"\n📈 MÉTRICAS FINALES DEL MODELO OPTIMIZADO:")
print(f"   • Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")
print(f"   • Precision: {final_precision:.4f} ({final_precision*100:.1f}%)")
print(f"   • Recall: {final_recall:.4f} ({final_recall*100:.1f}%)")
print(f"   • F1-Score: {final_f1:.4f}")
print(f"   • ROC-AUC: {final_auc:.4f}")

# Matriz de confusión
print(f"\n🎯 MATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_val, y_pred_final)
tn, fp, fn, tp = cm.ravel()

print(f"                 Predicción")
print(f"               No Fraude  Fraude")
print(f"Real No Fraude    {tn:4d}    {fp:4d}")
print(f"Real Fraude       {fn:4d}    {tp:4d}")

# Interpretación de negocio
print(f"\n💼 INTERPRETACIÓN DE NEGOCIO:")
print(f"   📊 Estadísticas operacionales:")
print(f"      • Fraudes detectados correctamente: {tp} de {tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"      • Falsos positivos (falsa alarma): {fp}")
print(f"      • Fraudes no detectados (pérdida): {fn}")
print(f"      • Casos normales correctos: {tn}")

print(f"\n   💰 Impacto de negocio:")
fraud_detection_rate = tp / (tp + fn) * 100
false_positive_rate = fp / (fp + tn) * 100
print(f"      • Tasa de detección de fraude: {fraud_detection_rate:.1f}%")
print(f"      • Tasa de falsas alarmas: {false_positive_rate:.1f}%")

if final_recall >= 0.8:
    print("      ✅ EXCELENTE: Alta capacidad de detección de fraudes")
elif final_recall >= 0.6:
    print("      ⚠️ BUENO: Capacidad moderada de detección")
else:
    print("      ❌ REQUIERE MEJORA: Baja detección de fraudes")

if final_precision >= 0.7:
    print("      ✅ EXCELENTE: Pocas falsas alarmas")
elif final_precision >= 0.5:
    print("      ⚠️ ACEPTABLE: Falsas alarmas moderadas")
else:
    print("      ❌ PROBLEMÁTICO: Muchas falsas alarmas")

# Comparación con modelo original
print(f"\n📊 COMPARACIÓN CON METODOLOGÍA ORIGINAL:")
print(f"   🔄 Mejoras implementadas:")
print(f"      ✅ Imputación KNN vs llenar con 0")
print(f"      ✅ One-hot encoding vs asignación arbitraria") 
print(f"      ✅ Balanceo con SMOTE vs datos desbalanceados")
print(f"      ✅ Múltiples métricas vs solo accuracy")
print(f"      ✅ Optimización de hiperparámetros")
print(f"      ✅ Escalado de features")

# Recomendaciones finales
print(f"\n🚀 PRÓXIMOS PASOS Y RECOMENDACIONES:")
print(f"   🔧 Para mejorar el modelo:")
print(f"      • Feature engineering adicional (ratios, interacciones)")
print(f"      • Ensemble methods (combinación de modelos)")
print(f"      • Análisis de features más importantes")
print(f"      • Ajuste del threshold de clasificación según costos de negocio")

print(f"\n   🏭 Para producción:")
print(f"      • Pipeline automatizado de preprocesamiento")
print(f"      • Monitoreo de drift en datos")
print(f"      • Reentrenamiento periódico")
print(f"      • Sistema de alertas en tiempo real")
print(f"      • Feedback loop para mejorar continuamente")

print(f"\n   📊 Monitoreo en producción:")
print(f"      • Tracking de precision/recall mensual")
print(f"      • Análisis de falsos positivos por operadores")
print(f"      • Costos vs beneficios de detección")

# Feature importance (si es posible)
if hasattr(best_model_optimized, 'feature_importances_'):
    print(f"\n🎯 TOP 10 FEATURES MÁS IMPORTANTES:")
    feature_names = X_train_balanced.columns
    importances = best_model_optimized.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

print(f"\n" + "="*80)
print("✅ ANÁLISIS COMPLETO FINALIZADO")
print("🎉 MODELO SUPERIOR DE DETECCIÓN DE FRAUDE IMPLEMENTADO")
print("="*80)

if __name__ == "__main__":
    print("\n📋 Implementación completa exitosa!")
    print("💾 Guardando modelo optimizado...")
    
    # Guardar el modelo y preprocessors para uso futuro
    import joblib
    
    # Guardar modelo
    joblib.dump(best_model_optimized, 'fraud_detection_model_optimized.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Modelo guardado como 'fraud_detection_model_optimized.pkl'")
    print("✅ Scaler guardado como 'scaler.pkl'")
    
    # Crear resumen para reporte
    summary = {
        'best_model': best_model_name,
        'final_metrics': {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': final_f1,
            'roc_auc': final_auc
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'business_impact': {
            'fraud_detection_rate': f"{fraud_detection_rate:.1f}%",
            'false_positive_rate': f"{false_positive_rate:.1f}%"
        }
    }
    
    import json
    with open('model_performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✅ Resumen guardado como 'model_performance_summary.json'")