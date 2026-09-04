# Leer Primero

Informes de calidad y trazabilidad de Gold. Los genera el pipeline en cada ejecucion.

## Archivos

- `Informe Calidad Gold.json`: conteos y validaciones de la tabla por ruta.
- `Informe Calidad Gold Tramos.json`: lo mismo para la tabla por tramo, que es la del modelo.
- `Informe Pipeline Completo.json`: resultado de ejecutar el pipeline entero, con el estado de
  cada paso y las validaciones agregadas de todas las capas.

## Que mirar antes de empezar a analizar

En los tres, el campo `status` debe valer `ok`. Ademas, todos los contadores cuyo nombre empieza
por `routes_without`, `tramos_sin` o `gold_segments_without` deben valer **0**: cuentan filas a
las que les falta algo, asi que cualquier valor distinto de cero indica un problema de union.

Dos validaciones merecen atencion especial:

- `routes_with_unknown_scan_status`: si deja de ser 0, ha aparecido un estado de escaneo de
  paquete que el pipeline no contempla.
- `routes_with_inconsistent_distance_split`: comprueba que la distancia total sigue siendo la
  suma de sus dos componentes.

## Uso

Consultarlos antes del EDA y despues de cualquier cambio en el pipeline.
