# Interpretabilidad del modelo final (TreeSHAP)

Generado por `entrenamiento_final.py` el 2026-09-15.

Modelo entrenado sobre el 100% de `train` (80% de las rutas), evaluado UNA VEZ
sobre `test` (20%, nunca visto antes), y explicado con TreeSHAP nativo de cada
libreria (no `shap.TreeExplainer`, solo se usa `shap` para el trazado).

## Rendimiento en test

```text
global    MAE 0.3683 min   R2 0.9398
reparto   MAE 0.3518 min   R2 0.6487   (1865 arboles)
almacen   MAE 2.7868 min   R2 0.8728   (532 arboles)
```

## Importancia de variables por SHAP (media de |valor SHAP|, minutos)

### Reparto

| Variable | \|SHAP\| medio (min) | % del total |
|---|---:|---:|
| `segment_distance_km` | 0.3966 | 55.81% |
| `station_code` | 0.0871 | 12.25% |
| `same_zone` | 0.0790 | 11.12% |
| `to_zone_id` | 0.0378 | 5.32% |
| `route_total_stops` | 0.0228 | 3.21% |
| `service_time_at_destination_seconds` | 0.0203 | 2.86% |
| `cumulative_distance_km` | 0.0161 | 2.26% |
| `segment_position` | 0.0135 | 1.90% |
| `packages_with_window_at_destination` | 0.0098 | 1.37% |
| `departure_hour` | 0.0084 | 1.18% |
| `route_package_count` | 0.0066 | 0.93% |
| `weekday` | 0.0058 | 0.82% |

### Almacen

| Variable | \|SHAP\| medio (min) | % del total |
|---|---:|---:|
| `segment_distance_km` | 7.1612 | 71.15% |
| `station_code` | 1.6063 | 15.96% |
| `departure_hour` | 0.3031 | 3.01% |
| `route_total_stops` | 0.2606 | 2.59% |
| `route_package_count` | 0.1860 | 1.85% |
| `packages_with_window_at_destination` | 0.1612 | 1.60% |
| `volume_at_destination_cm3` | 0.1510 | 1.50% |
| `service_time_at_destination_seconds` | 0.1186 | 1.18% |
| `weekday` | 0.0914 | 0.91% |
| `packages_at_destination` | 0.0261 | 0.26% |

`to_zone_id` aporta el 5.32% de la magnitud media de SHAP en reparto.

## Graficos

```text
figuras/Reparto importancia SHAP.png
figuras/Reparto distribucion SHAP.png
figuras/Almacen importancia SHAP.png
figuras/Almacen distribucion SHAP.png
```

## Ejemplos de explicacion local

### Ejemplo: tramo de reparto tipico

```text
tiempo real:        0.77 min
prediccion:         0.45 min
valor base (media): 0.92 min

mayores contribuciones (segundos):
  segment_distance_km                       -15.9 s   (valor: 0.091973)
  station_code                               -6.2 s   (valor: DLA4)
  same_zone                                  -2.4 s   (valor: 1.0)
  service_time_at_destination_seconds        -1.9 s   (valor: 20.0)
  segment_position                           +0.7 s   (valor: 142)
  to_zone_id                                 -0.7 s   (valor: B-11.1C)
```

### Ejemplo: tramo de reparto largo (atipico)

```text
tiempo real:        9.63 min
prediccion:         1.78 min
valor base (media): 0.92 min

mayores contribuciones (segundos):
  segment_distance_km                       +24.5 s   (valor: 0.270769)
  station_code                              +14.8 s   (valor: DLA7)
  to_zone_id                                +12.3 s   (valor: C-15.2B)
  same_zone                                  -3.5 s   (valor: 1.0)
  service_time_at_destination_seconds        +2.3 s   (valor: 125.0)
  cumulative_distance_km                     +1.6 s   (valor: 23.970963)
```

### Ejemplo: tramo de almacen (mediana)

```text
tiempo real:        29.91 min
prediccion:         30.24 min
valor base (media): 30.00 min

mayores contribuciones (segundos):
  segment_distance_km                       +95.5 s   (valor: 18.671928)
  station_code                              -62.2 s   (valor: DLA5)
  volume_at_destination_cm3                 -24.6 s   (valor: 2135.0)
  weekday                                   +12.1 s   (valor: 6)
  route_total_stops                          -6.5 s   (valor: 156)
  departure_hour                             -6.1 s   (valor: 15)
```

