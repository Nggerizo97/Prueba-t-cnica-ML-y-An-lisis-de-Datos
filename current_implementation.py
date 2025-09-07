#!/usr/bin/env python
# coding: utf-8

# # Librerias

# In[1]:


# Tratamiento de datos
import numpy as np
import pandas as pd
import math

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')

# Estilos de las gráfica
plt.style.use('bmh') #makes plots look pretty

# Preprocesado y modelado
# Sklearn
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV, KFold, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn import datasets
from sklearn.pipeline import make_pipeline

# Escalado de datos
from sklearn.preprocessing import scale

# Configuración ignore warnings
import warnings
warnings.filterwarnings('ignore')

#import pingouin as pg #Correlación con intervalos de confianza
from scipy import stats
from scipy.stats import pearsonr
import scipy
import statsmodels.api as sm


# # Procesamiento De datos

# In[2]:


df1 = pd.read_excel('base_evaluada.xlsx')
df2 = pd.read_excel('entrenamiento_fraude.xlsx')
df3 = pd.read_excel('testeo_fraude.xlsx')


# In[3]:


df4 = pd.merge(df1,df3, left_on="radicado",right_on="radicado") # junto los data set 
df4


# In[4]:


df4=df4.fillna(0) # No puedo eliminar valores NaN, eliminaria filas. 
#Por lo que los voy a reemplzar por 0


# In[5]:


df2=df2.fillna(0)


# In[6]:


df2.describe().T#analizar las variables numericas y su comportamiento


# Analizo los valores con texto, la idea es asignarles un valor para poder identificar luego cuales son los valores más significativos en el data set. Luego se plantea el modelo para verificar que pueda funcionar correctamente

# In[7]:


df2["descri_apli_prod_ben"].value_counts()


# In[8]:


df2["marca_host_no_resp"].value_counts()


# In[9]:


df2["marca_timeout"].value_counts()


# In[10]:


d = {ni: indi for indi, ni in enumerate(set(df2["descri_apli_prod_ben"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df2["descri_apli_prod_ben"]]
df2["descri_apli_prod_ben"]=numbers


# In[11]:


d = {ni: indi for indi, ni in enumerate(set(df2["marca_host_no_resp"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df2["marca_host_no_resp"]]
df2["marca_host_no_resp"]=numbers


# In[12]:


d = {ni: indi for indi, ni in enumerate(set(df2["marca_timeout"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df2["marca_timeout"]]
df2["marca_timeout"]=numbers


# In[13]:


df_entrenar=df2.drop(labels=['radicado', 'fraude'], axis=1)
df_entrenar


# In[ ]:





# # Modelo

# In[14]:


X = df_entrenar
Y = df2['fraude']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train) #se entrena el modelo
print(rfc.score(X_test,y_test)) # se mustra el puntaje 
pred = rfc.predict(df_entrenar)


# In[15]:


pred


# Se hace un modelo de randoforestclassifier para determinar en este caso use todas las variables para ejecutar el modelo, las variables Object las converti a valores númericas para que se pudiera entrenar por completo el modelo. El modelo se puede entrenar con menos variables pero eso implicaria conocer a fondo todas las variables que peso le puede dar al modelo. Con un metodo de correlacion se pueden verificar que variables tienen mayor peso con respecto a la que se desea analizar que en este caso es fraude. Tambien es posible usar un diagrama de calor para determinar cuales tiene menos relaciones entre sí. 

# In[16]:


df2.corr()


# Todos los valores son muy cercanos a 0 por lo que tiene relaciones nulas o muy debiles 

# In[17]:


# Diagnóstico errores (residuos) de las predicciones de validación cruzada
# ==============================================================================
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import statsmodels.api as sm

# Validación cruzada
# ==============================================================================
cv = KFold(n_splits=5, random_state=123, shuffle=True)
cv_prediccones = cross_val_predict(
                    estimator = rfc,
                    X         = X_train,
                    y         = y_train,
                    cv        = cv
                  )

# Gráficos
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

axes[0, 0].scatter(y_train, cv_prediccones, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'k--', color = 'black', lw=2)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), y_train - cv_prediccones,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
axes[0, 1].tick_params(labelsize = 7)

sns.histplot(
    data    = y_train - cv_prediccones,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)

axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    y_train - cv_prediccones,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");


# In[18]:


from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)
cv_scores = cross_val_score(
                estimator = rfc,
                X         = X_train,
                y         = y_train,
                scoring   = 'neg_root_mean_squared_error',
                cv        = cv,
                n_jobs    = -1 # todos los cores disponibles
             )

print(f"Média métricas de validación cruzada: {cv_scores.mean()}")


# In[19]:


# neg_root_mean_squared_error de test
# ==============================================================================
from sklearn.metrics import mean_squared_error

predicciones = rfc.predict(X_test)
rmse = mean_squared_error(
        y_true = y_test,
        y_pred = predicciones,
        squared = False
       )
rmse


# El error cuadrático medio (RMSE) mide la cantidad de error que hay entre dos conjuntos de datos. En otras palabras, compara un valor predicho y un valor observado o conocido. Cuanto más pequeño es un valor RMSE, más cercanos son los valores predichos y observados.

# # Entrenar los datos 

# Para el que el dataset testeo_fraude pueda ser entrenado se debe proceder a hacer la limpiza de datos.

# In[20]:


df3=df3.fillna(0)


# In[21]:


d = {ni: indi for indi, ni in enumerate(set(df3["descri_apli_prod_ben"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df3["descri_apli_prod_ben"]]
df3["descri_apli_prod_ben"]=numbers


# In[22]:


d = {ni: indi for indi, ni in enumerate(set(df3["marca_host_no_resp"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df3["marca_host_no_resp"]]
df3["marca_host_no_resp"]=numbers


# In[23]:


d = {ni: indi for indi, ni in enumerate(set(df3["marca_timeout"]))}#receta para asignar un número a cada valor único en una lista
numbers = [d[ni] for ni in df3["marca_timeout"]]
df3["marca_timeout"]=numbers


# In[24]:


df_entrenar1 = df3.drop(labels=['radicado'], axis=1)


# In[25]:


pred_datos = rfc.predict(df_entrenar1)


# In[26]:


type(pred_datos)


# In[30]:


df1=df1.fillna(0)


# In[42]:


dft = pd.DataFrame(df1)


# In[46]:


dft.columns = dft.columns.str.strip()


# Se usa esto porque al parecer farude_pred no tiene atributos, se usa este metodo que perimte deshacerse del error.
# De otro modo lo más conveniente es cambiar el nombre directamente desde el excel

# In[49]:


dft['fraude_pred'] = pred_datos


# In[50]:


dft


# Con esto se deja entrenado donde el valor 1 es fraude y el valor 0 es no fraude

# In[51]:


dft = dft.to_excel('base_evaluada1.xlsx')


# In[ ]:




