# 🗺️ Integración de Datos Espaciales (ENP)

He preparado el código de tu Dashboard para que calcule matemáticamente la distancia real entre cada incendio y el Espacio Natural Protegido (ENP) más cercano.

## Visualización Interactiva en el Mapa 3D

Atendiendo a tu última petición, he añadido un **interruptor (toggle)** redondo justo encima de tu mapa 3D con el texto: *"🌿 Mostrar Capa de Espacios Naturales Protegidos"*. 

Para lograr que tu navegador no colapse al intentar dibujar los más de 200MB de parques naturales de España, he aplicado las siguientes técnicas bajo el capó:
1.  **Simplificación de Vértices:** El código de Python toma los polígonos originales y les aplica un algoritmo de simplificación con una tolerancia de 150 metros. Es decir, borra vértices intermedios que estén a menos de 150 metros de la línea principal. Esto reduce el peso visual en un 90% pero mantiene la forma de los parques perfecta a vista de pájaro.
2.  **Reproyección en caliente:** Aunque nuestros archivos Parquet están en husos métricos (`EPSG:25830` y `EPSG:32628`), la librería `Pydeck` que usamos para el mapa 3D exige que los datos estén en latitud/longitud global (`EPSG:4326`). El sistema los reproyecta en el aire, los une, y genera el JSON especial para pintarlos.
3.  **Caché Avanzado:** Toda esta conversión matemática tarda un par de segundos, pero gracias a `@st.cache_data`, Streamlit guarda el resultado final en la memoria. Si apagas y enciendes el toggle, aparecerán de forma instantánea.

Los polígonos se dibujarán de color **verde semi-transparente** con un borde sólido verde oscuro, para que puedas ver perfectamente tanto el parque natural como si hay algún punto de incendio (rojo/naranja) dentro o cerca de él.
