# Pyrrhenius

Arrhenius tiene dos funcionalidades principales:
1. Obtener coeficiente de difusión de traza a partir del desplazamiento
   cuadrático medio.
2. A partir de datos (temperatura, coeficiente de difusión, **errores opcionales**)
   hace un ajuste tipo arrhenius, plotea y extrapola a la temperatura deseada
   (por default la ambiente).

Notas: Ver en PyPI (y su documentación)
---------------------------------------
1. Ver `chempy`: es más para enseñar química, mostrar fórmulas, etc.
2. Ver `svante`: hace más o menos esto pero está roto y no me gusta como está implementado
    (no tiene la api en la documentación, ni tutorial).
3. Ver `MDAnalysis`: brinda funcionalidades para calcular el desplazamiento cuadrático
    medio de Einstein y nombra en la documentación cómo se haría para extrapolar (los
    cuidados que hay que tener), pero pareciera decir que no piensa implementarlo 
    en su código.

Una vez visto esto:
- class Arrhenius: **ver si heredar de sklearn** Base algo...
- Aprender a usar bien MDAnalysis y usarlo como ejemplo para calcular el desplazamiento
    cuadrático medio en los tutoriales.
- Para los tests usar directamente dato (t, msd).
- Discutir con Juan.
- Además de el código escrito, considerar distintas variantes de cambio de unidades.
