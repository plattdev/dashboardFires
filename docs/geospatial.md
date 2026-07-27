# 🚀 Mejores Prácticas Geoespaciales 2026 en Python

Trabajar con múltiples archivos GeoJSON masivos (como los 217 MB de Canarias y la Península) representa un desafío para el rendimiento de cualquier aplicación web. Para solucionarlo, hemos aplicado tres principios fundamentales de ingeniería de datos espaciales.

## 1. 📦 Formato GeoParquet (Rendimiento Extremo)

El formato **GeoJSON** es excelente para compartir datos porque es texto legible, pero es muy ineficiente para la lectura por parte de las máquinas. Cargar 200MB de texto en JSON cada vez que se reinicia el servidor de Streamlit puede tardar más de 20 segundos.

> [!TIP]
> **La Solución:** Convertir los datos a **GeoParquet**. Es un formato binario en columnas diseñado para el Big Data.
> 
> *   **Compresión:** Reduce drásticamente el tamaño del archivo en el disco.
> *   **Velocidad:** Los archivos Parquet se cargan en la memoria en milisegundos. Tu dashboard ahora arranca instantáneamente gracias a esto.

## 2. 🌍 Respetar los Sistemas de Coordenadas (Precisión Matemática)

En cartografía, la Tierra (esférica) debe proyectarse sobre un plano para poder medir distancias en metros. España es tan ancha que no cabe en un solo "huso" (zona de proyección) sin deformarse gravemente:
*   **Península y Baleares:** Utilizan el huso UTM 30N (`EPSG:25830`).
*   **Islas Canarias:** Al estar tan al sur y al oeste, utilizan el huso UTM 28N (`EPSG:32628`).

> [!WARNING]
> **El Error Común (Práctica Obsoleta):** Forzar a Canarias a proyectarse en el mapa de la Península (`EPSG:25830`). Esto deforma brutalmente la geometría de las islas, haciendo que cualquier cálculo de distancia (como nuestra regla de "5000 metros") devuelva resultados erróneos.

> [!IMPORTANT]
> **Nuestra Solución (Mejor Práctica):** Mantener las geometrías separadas. El código detecta los incendios de Canarias y los mide en su propio plano perfecto (`EPSG:32628`). Luego, detecta los incendios de la Península y los mide en su plano (`EPSG:25830`). Finalmente, suma los dos resultados sin comprometer la precisión matemática de ninguno.

## 3. 🧠 Uso de Índices Espaciales (R-Tree) con `sjoin_nearest`

Para saber cuántos incendios están a menos de 5 km de un Espacio Natural Protegido, podríamos dibujar un círculo (buffer) de 5 km alrededor de cada incendio y ver si se cruza matemáticamente con alguno de los 2.000 polígonos. 

> [!CAUTION]
> Dibujar buffers y cruzar geometrías complejas (con miles de vértices) una por una es computacionalmente lentísimo y puede bloquear el servidor.

> [!TIP]
> **La Solución:** Utilizar `gpd.sjoin_nearest(fires, enp, max_distance=5000)`. 
> Esta función no dibuja círculos. En su lugar, utiliza el **Índice Espacial R-Tree** que viene incorporado en GeoPandas. Es como un índice de un diccionario cartográfico que descarta instantáneamente los polígonos lejanos y solo calcula la distancia exacta con el polígono más cercano. ¡Hace millones de comprobaciones en fracciones de segundo!
