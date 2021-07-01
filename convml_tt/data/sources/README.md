# dataset generation from `convml_tt`

Producing triplets for training and study:

```yaml
source: goes16
type: truecolor_rgb

time:
  N_days: 40
  N_hours_from_zenith: 0.2
  t_start: 2020-01-15 00:00:00

domain:
  kind: rect
  lat0: 14
  lon0: -48
  l_zonal: 3000.0e+3
  l_meridional: 1000.0e+3

projection:
  kind: triplet
  N_triplets: {study: 0, train: 0}
  tile_N: 256
  tile_size: 200000.0
```



```yaml
source: goes16
type: truecolor_rgb

time:
  N_days: 40
  N_hours_from_zenith: 0.2
  t_start: 2020-01-15 00:00:00

domain:
  kind: rect
  lat0: 14
  lon0: -48
  l_zonal: 3000.0e+3
  l_meridional: 1000.0e+3

projection:
  kind: rect
  resolution: 1000. # in m/deg
```


```bash
$> python -m luigi --module convml_tt.data.sources.pipeline GenerateAllScenes
```
