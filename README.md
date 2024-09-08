# Árbol de Decisión para Regresión: Predicción de Precios de Casas

Este proyecto implementa un modelo de **Árbol de Decisión para Regresión** utilizando Python y la biblioteca `scikit-learn`. El objetivo es predecir el precio de una casa en función de su tamaño (en metros cuadrados). El proyecto está diseñado para ser ejecutado en un entorno de **Jupyter Notebook**, tal como se muestra en la carpeta `notebooks`.

## Tabla de Contenidos

- [Árbol de Decisión para Regresión: Predicción de Precios de Casas](#árbol-de-decisión-para-regresión-predicción-de-precios-de-casas)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Estructura del Proyecto](#estructura-del-proyecto)
  - [Requisitos](#requisitos)
  - [Instalación](#instalación)
  - [Uso del Proyecto](#uso-del-proyecto)
    - [Entrenar el modelo](#entrenar-el-modelo)
    - [Visualizar las predicciones](#visualizar-las-predicciones)
    - [Visualizar la estructura del árbol](#visualizar-la-estructura-del-árbol)
  - [Explicación Técnica](#explicación-técnica)
    - [Árbol de Decisión para Regresión](#árbol-de-decisión-para-regresión)
      - [Explicación del `reshape(-1, 1)`](#explicación-del-reshape-1-1)
  - [Conclusión](#conclusión)

---

## Estructura del Proyecto

El proyecto tiene la siguiente estructura de carpetas y archivos:

```bash
├── docs/                        # Documentación adicional (opcional)
├── notebooks/                   # Contiene los Jupyter Notebooks
│   ├── .gitkeep                 # Archivo para mantener la carpeta en Git
│   └── decision-tree-regression.ipynb # El notebook principal donde se ejecuta el código
├── .gitignore                   # Ignorar archivos y carpetas innecesarios para Git
├── environment.yml              # Especificaciones del entorno (opcional para conda)
├── LICENSE                      # Licencia del proyecto
├── Makefile                     # Automación de tareas (opcional)
├── pyproject.toml               # Configuración de proyecto (opcional)
├── README.md                    # Este archivo README explicativo
├── setup.cfg                    # Configuración adicional de Python
```

## Requisitos

Asegúrate de tener las siguientes herramientas y bibliotecas instaladas:

- **Python 3.6+**
- **Jupyter Notebook**
- **scikit-learn**
- **numpy**
- **matplotlib**

Puedes instalar las dependencias necesarias con los siguientes comandos:

```bash
pip install scikit-learn numpy matplotlib
```

Si estás utilizando un entorno de conda, puedes instalar el entorno a partir del archivo `environment.yml`:

```bash
conda env create -f environment.yml
```

## Instalación

1. **Clona el repositorio** (si es necesario):

   ```bash
   git clone https://github.com/DiegoLerma/decision-tree-regressor.git
   ```

2. **Navega al directorio del proyecto**:

   ```bash
   cd decision-tree-regressor
   ```

3. **Ejecuta Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

4. **Abre el archivo `decision-tree-regression.ipynb`** dentro del directorio `notebooks`.

## Uso del Proyecto

El código principal se encuentra en el archivo Jupyter Notebook `decision-tree-regression.ipynb`. A continuación, se explica cómo se ejecuta y cómo puedes interpretar los resultados.

### Entrenar el modelo

El archivo `decision-tree-regression.ipynb` define los datos simulados de tamaños de casas y sus precios. Primero, se entrena el árbol de decisión con estos datos utilizando `scikit-learn`.

Datos de entrada:

```python
sizes = np.array([80, 120, 95, 130, 85, 110]).reshape(-1, 1)
prices = np.array([20000, 250000, 210000, 280000, 205000, 240000])
```

Entrenar el modelo:

```python
tree_regressor = DecisionTreeRegressor(max_depth=2, random_state=0)
tree_regressor.fit(sizes, prices)
```

### Visualizar las predicciones

Una vez entrenado el árbol de decisión, generamos un rango continuo de tamaños de casas y hacemos predicciones para visualizar cómo el árbol predice los precios en función del tamaño de la casa.

```python
continuous_sizes = np.linspace(70, 140, 500).reshape(-1, 1)
predicted_prices = tree_regressor.predict(continuous_sizes)

# Visualizamos los datos reales y las predicciones
plt.scatter(sizes, prices, color="black", label="Datos reales")
plt.plot(continuous_sizes, predicted_prices, color="blue", label="Predicciones del árbol")
plt.xlabel("Tamaño de la casa (m²)")
plt.ylabel("Precio de la casa ($)")
plt.title("Árbol de Decisión para Regresión: Predicción de Precios de Casas")
plt.legend()
plt.grid(True)
plt.show()
```

El gráfico generado mostrará los puntos de datos reales y la línea de predicción del árbol. La predicción será una serie de saltos, ya que el Árbol de Decisión genera regiones planas en cada partición de datos.

### Visualizar la estructura del árbol

Puedes ver la estructura interna del árbol de decisión con el siguiente código:

```python
from sklearn import tree

# Mostramos el árbol de decisión en texto
tree_text = tree.export_text(tree_regressor, feature_names=["Tamaño (m²)"])
print(tree_text)
```

El resultado será una representación en texto del árbol, mostrando cómo el modelo toma decisiones basadas en el tamaño de la casa.

También puedes visualizar el árbol gráficamente con `plot_tree`:

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 5))
plot_tree(tree_regressor, filled=True, feature_names=["Tamaño (m²)"])
plt.show()
```

Esto generará una imagen que muestra las divisiones en los nodos y los valores de predicción en cada hoja.

## Explicación Técnica

### Árbol de Decisión para Regresión

Un Árbol de Decisión para Regresión divide el espacio de características (en este caso, el tamaño de la casa) en varias regiones, realizando predicciones basadas en el valor promedio de la variable objetivo (precio de la casa) en cada región.

En nuestro caso:

- El árbol se entrena usando el tamaño de las casas como la única característica.
- Cada división del árbol crea regiones en las que los datos son más homogéneos.
- El criterio de división utilizado es la minimización del **Error Cuadrático Medio (MSE)**, que mide la diferencia entre los precios reales y los predichos.

#### Explicación del `reshape(-1, 1)`

El uso de `reshape(-1, 1)` convierte un arreglo de una dimensión en una matriz de 2 dimensiones, donde cada valor original es una fila con una columna. Esto es necesario porque `scikit-learn` espera que los datos de entrada tengan dos dimensiones: una para las muestras y otra para las características.

```python
# Ejemplo del reshape
sizes = np.array([80, 120, 95, 130, 85, 110]).reshape(-1, 1)
```

Sin el `reshape(-1, 1)`, los datos estarían en formato de una dimensión, lo que resultaría en un error, ya que el modelo espera una matriz de entrada de 2 dimensiones.

## Conclusión

Este proyecto demuestra cómo implementar y visualizar un Árbol de Decisión para Regresión utilizando Python y `scikit-learn`. A través del análisis de los precios de casas, hemos construido un modelo simple que predice precios en función del tamaño de la casa. También se ha mostrado cómo visualizar y entender las decisiones que toma el modelo internamente.
