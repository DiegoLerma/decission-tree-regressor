# Árbol de Decisión para Regresión: Predicción de Precios de Casas

Este proyecto implementa un modelo de **Árbol de Decisión para Regresión** utilizando Python y la biblioteca `scikit-learn` para predecir los precios de casas en función de su tamaño (en metros cuadrados). A través de este ejemplo, aprenderás cómo construir, entrenar y visualizar un árbol de decisión para realizar predicciones sobre datos continuos.

## Tabla de contenidos

- [Árbol de Decisión para Regresión: Predicción de Precios de Casas](#árbol-de-decisión-para-regresión-predicción-de-precios-de-casas)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Requisitos](#requisitos)
  - [Descripción del proyecto](#descripción-del-proyecto)
  - [Instalación](#instalación)
  - [Uso del proyecto](#uso-del-proyecto)
    - [Entrenar el modelo](#entrenar-el-modelo)
    - [Visualización de predicciones](#visualización-de-predicciones)
    - [Estructura del árbol de decisión](#estructura-del-árbol-de-decisión)
  - [Explicación técnica](#explicación-técnica)
    - [Árbol de Decisión para Regresión](#árbol-de-decisión-para-regresión)
      - [Fórmula del MSE](#fórmula-del-mse)
    - [Explicación del reshape(-1, 1)](#explicación-del-reshape-1-1)
  - [Conclusión](#conclusión)

---

## Requisitos

Este proyecto requiere las siguientes dependencias:

- **Python 3.6+**
- **scikit-learn** (para construir y entrenar el árbol de decisión)
- **numpy** (para manejo de matrices y arreglos)
- **matplotlib** (para visualizar los resultados)

Puedes instalar todas las dependencias usando `pip`:

```bash
pip install scikit-learn numpy matplotlib
```

## Descripción del proyecto

En este proyecto, utilizamos un Árbol de Decisión para Regresión para predecir el precio de una casa basado en su tamaño en metros cuadrados. Los árboles de decisión son herramientas de aprendizaje supervisado que pueden utilizarse tanto para clasificación como para regresión. En este caso, usaremos regresión, ya que nuestra variable objetivo (precio de la casa) es continua.

**Pasos clave del proyecto**:

1. Definir los datos de entrada (tamaño de las casas y precios).
2. Entrenar un árbol de decisión para ajustar el modelo a los datos.
3. Hacer predicciones para nuevos tamaños de casas.
4. Visualizar el comportamiento del árbol en los datos.
5. Interpretar la estructura del árbol de decisión.

## Instalación

Sigue estos pasos para descargar y configurar el proyecto en tu máquina local.

1. **Clona el repositorio** (si es necesario):

   ```bash
   git clone https://github.com/tuusuario/arbol-decision-regresion.git
   ```

2. **Navega al directorio del proyecto**:

   ```bash
   cd arbol-decision-regresion
   ```

3. **Instala las dependencias**:

   Asegúrate de tener un entorno virtual activado (opcional pero recomendado). Después, ejecuta:

   ```bash
   pip install -r requirements.txt
   ```

   O instala las dependencias manualmente con:

   ```bash
   pip install scikit-learn numpy matplotlib
   ```

## Uso del proyecto

El archivo principal del proyecto es `arbol_regresion.py`. A continuación se explica cómo ejecutar el script y qué hace cada sección del código.

### Entrenar el modelo

1. **Definir los datos simulados**:

   En el archivo `arbol_regresion.py`, tenemos un conjunto de datos simulados que consisten en el tamaño de varias casas (en metros cuadrados) y sus respectivos precios (en dólares).

   ```python
   # Datos simulados: tamaños de casas (m²) y sus precios ($)
   tamanos = np.array([80, 120, 95, 130, 85, 110]).reshape(-1, 1)
   precios = np.array([200000, 250000, 210000, 280000, 205000, 240000])
   ```

   El uso de `reshape(-1, 1)` es necesario para convertir el arreglo de tamaños en una matriz de 2 dimensiones, donde cada valor es una muestra con una única característica (tamaño de la casa).

2. **Inicializar y entrenar el árbol de decisión**:

   Se inicializa el modelo de Árbol de Decisión para Regresión con una profundidad máxima de 2 niveles para evitar sobreajuste.

   ```python
   arbol_regresion = DecisionTreeRegressor(max_depth=2, random_state=0)
   arbol_regresion.fit(tamanos, precios)
   ```

   Aquí, la profundidad máxima del árbol está limitada a 2 niveles para simplificar el modelo y evitar un ajuste excesivo a los datos de entrenamiento.

### Visualización de predicciones

Una vez entrenado el modelo, realizamos predicciones para un rango más amplio de tamaños de casas (de 70 a 140 m²) y visualizamos los resultados en una gráfica.

```python
# Generamos un rango de tamaños de casas
tamanos_continuos = np.linspace(70, 140, 500).reshape(-1, 1)

# Hacemos predicciones con el árbol
precios_predichos = arbol_regresion.predict(tamanos_continuos)

# Visualizamos los datos reales y las predicciones
plt.scatter(tamanos, precios, color="black", label="Datos reales")
plt.plot(tamanos_continuos, precios_predichos, color="blue", label="Predicciones del árbol")
plt.xlabel("Tamaño de la casa (m²)")
plt.ylabel("Precio de la casa ($)")
plt.title("Árbol de Decisión para Regresión: Predicción de Precios de Casas")
plt.legend()
plt.grid(True)
plt.show()
```

### Estructura del árbol de decisión

El árbol de decisión creado se puede visualizar en formato de texto para entender cómo se han realizado las divisiones basadas en el tamaño de las casas.

```python
from sklearn import tree

# Mostramos el árbol de decisión en texto
tree_text = tree.export_text(arbol_regresion, feature_names=["Tamaño (m²)"])
print(tree_text)
```

Este código imprimirá la estructura del árbol en la consola, permitiéndote ver los nodos de división y las predicciones que realiza en cada partición.

## Explicación técnica

### Árbol de Decisión para Regresión

Un Árbol de Decisión para Regresión es un modelo de aprendizaje supervisado que divide el espacio de características en regiones basadas en divisiones secuenciales (llamadas nodos). Cada nodo realiza una división en función de un umbral que minimiza una medida de error (como el Error Cuadrático Medio, MSE). Las predicciones dentro de cada región son el promedio de los valores observados en esa región.

- **Criterio de división**: En este proyecto, las divisiones se realizan basadas en la variable `Tamaño (m²)` para minimizar el MSE.
- **Predicción en las hojas**: Cuando el árbol alcanza una hoja, predice el valor promedio de los datos en esa hoja.

#### Fórmula del MSE

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^2
\]
Donde:

- \(n\) es el número de observaciones en la región,
- \(y_i\) es el valor real de la variable objetivo,
- \(\hat{y}\) es la predicción (el valor promedio en esa región).

### Explicación del reshape(-1, 1)

El método `reshape(-1, 1)` es necesario porque `scikit-learn` requiere que los datos de entrada tengan dos dimensiones: una para las muestras (filas) y otra para las características (columnas). En nuestro caso, tenemos una característica (tamaño de la casa), por lo que necesitamos que los datos estén en forma de matriz con una columna.

Sin `reshape(-1, 1)`, los datos serían un arreglo de 1D, lo que causaría un error, ya que `scikit-learn` espera una matriz de 2D para entrenar el modelo.

## Conclusión

Este proyecto muestra cómo usar un Árbol de Decisión para Regresión para predecir valores continuos, como el precio de una casa, en función de una característica. También explica cómo visualizar y entender el comportamiento del árbol. Los árboles de decisión son modelos potentes, pero pueden ser propensos al sobreajuste si no se limitan correctamente, por lo que es importante controlar la profundidad y otros hiperparámetros.
