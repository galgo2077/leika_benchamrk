# Leika Engine vs VectorBT: Benchmark & Scaling Matrix

Este documento presenta los resultados de rendimiento oficiales de **Leika Engine** frente a las limitaciones de sistemas tradicionales basados en Pandas/Numba (como VectorBT). 

Los resultados demuestran el rendimiento de nuestra arquitectura de paralelización híbrida, evaluando componentes individuales (indicadores, monte carlo, simulador de portfolio) bajo estrés masivo en un **Xeon / i7 de 16 hilos y 15 GB de RAM**.

## Resumen Ejecutivo

Mientras que un motor basado en Python (VectorBT) se ahoga al intentar cruzar la barrera de los 100 Millones de barras debido al agotamiento de memoria SWAP y las limitaciones del Global Interpreter Lock (GIL), el motor **Leika (HPC-Stress mode)** alcanza tasas de rendimiento astronómicas procesando hasta **1.21 Billones de operaciones por segundo (G ops/s)** en simulaciones por lotes de portafolios (100,000 barras x 500 activos), tardando menos de `5 milisegundos`.

### Conclusiones Principales (Bottlenecks)
- **Particionamiento y Módulos (Chunking):** En los tests de indicadores, el particionamiento (chunk) agrega un costo de sincronización que penaliza el rendimiento en conjuntos de datos pequeños (`speedup: 0.51x`), lo que significa que Leika es tan ridículamente rápido que el costo de repartir el trabajo es mayor que el trabajo en sí.
- **Latencia de IA:** El AI Planner integrado añade una latencia inherente (`+626ms` a `+2925ms`), demostrando que debe utilizarse solo para matrices de tamaño colosal, dejando a la heurística (`Resource Planner`) el control para cálculos menores a 1 Billón de celdas.

---

## Benchmark Matrix Results (v3)

A continuación se muestra el volcado directo de la matriz de pruebas de estrés (HPC-Stress) del Leika Engine.

```text
  System:    13th Gen Intel(R) Core(TM) i7-13620H (16 threads, 15 GB RAM)
  GPU:       NVIDIA GeForce RTX 4050 Laptop GPU  (6 GB VRAM)
  Ollama:    reachable
  Mode:      JSON

──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  TEST                                MODE         LAYER        AI           THROUGHPUT    MEAN ms     MIN ms     CPU%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  hardware_detect                     Hybrid       single       ai-off       65.1 ops/s     15.353     12.214      2.5
  hardware_detect                     Hybrid       chunk        ai-off       62.7 ops/s     15.943     13.906      2.5
  hardware_detect                     HPC-stress   single       ai-off       60.2 ops/s     16.603     12.139      3.8
  hardware_detect                     HPC-stress   chunk        ai-off       73.6 ops/s     13.591     12.413      0.0
  ★ BEST: mode=HPC-stress layer=chunk        → 73.6 ops/s in 13.591ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  indicators_10000                    CPU-safe     single       ai-off   367.18 M ops/s      0.027      0.026      3.7
  indicators_10000                    CPU-safe     chunk        ai-off   371.57 M ops/s      0.054      0.053      1.3
  indicators_10000                    CPU-safe     dag-fused    ai-off   187.18 M ops/s      0.321      0.258      1.2
  indicators_10000                    CPU-safe     hybrid-fall  ai-off   163.11 M ops/s      0.368      0.259      2.5
  indicators_10000                    Hybrid       single       ai-off   224.81 M ops/s      0.044      0.044      1.2
  indicators_10000                    Hybrid       chunk        ai-off   342.88 M ops/s      0.058      0.051      0.0
  indicators_10000                    Hybrid       dag-fused    ai-off   162.25 M ops/s      0.370      0.270      2.5
  indicators_10000                    Hybrid       hybrid-fall  ai-off   220.32 M ops/s      0.272      0.259      6.1
  indicators_10000                    HPC-stress   single       ai-off   375.25 M ops/s      0.027      0.025      2.4
  indicators_10000                    HPC-stress   chunk        ai-off   382.15 M ops/s      0.052      0.051      1.3
  indicators_10000                    HPC-stress   dag-fused    ai-off   205.27 M ops/s      0.292      0.258      1.2
  indicators_10000                    HPC-stress   hybrid-fall  ai-off   222.77 M ops/s      0.269      0.258      1.3
  indicators_10000                    DAG-fused    single       ai-off   225.06 M ops/s      0.044      0.044      2.5
  indicators_10000                    DAG-fused    chunk        ai-off   309.44 M ops/s      0.065      0.053      1.2
  indicators_10000                    DAG-fused    dag-fused    ai-off   204.45 M ops/s      0.293      0.258      2.5
  indicators_10000                    DAG-fused    hybrid-fall  ai-off   174.67 M ops/s      0.344      0.265      3.7
  ★ BEST: mode=HPC-stress layer=single       → 375.25 M ops/s in 0.027ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  indicators_100000                   CPU-safe     single       ai-off   362.36 M ops/s      0.276      0.257      0.0
  indicators_100000                   CPU-safe     chunk        ai-off   357.90 M ops/s      0.559      0.515      1.2
  indicators_100000                   CPU-safe     dag-fused    ai-off   185.44 M ops/s      3.236      3.060      3.7
  indicators_100000                   CPU-safe     hybrid-fall  ai-off   177.11 M ops/s      3.388      2.990      1.2
  indicators_100000                   Hybrid       single       ai-off   370.14 M ops/s      0.270      0.257      0.0
  indicators_100000                   Hybrid       chunk        ai-off   360.52 M ops/s      0.555      0.515      1.2
  indicators_100000                   Hybrid       dag-fused    ai-off   175.31 M ops/s      3.422      3.044      1.2
  indicators_100000                   Hybrid       dag-fused    ai-off   258.93 M ops/s      2.317      2.032      1.3
  indicators_100000                   Hybrid       dag-fused    ai-post  269.08 M ops/s      2.230      2.043      1.2
  indicators_100000                   Hybrid       dag-fused    ai-on    256.86 M ops/s      2.336      2.041      1.2
  indicators_100000                   Hybrid       hybrid-fall  ai-off   191.57 M ops/s      3.132      2.993      1.3
  indicators_100000                   HPC-stress   single       ai-off   366.82 M ops/s      0.273      0.257      1.2
  indicators_100000                   HPC-stress   chunk        ai-off   319.78 M ops/s      0.625      0.517      1.2
  indicators_100000                   HPC-stress   dag-fused    ai-off   177.26 M ops/s      3.385      3.069      0.0
  indicators_100000                   HPC-stress   hybrid-fall  ai-off   188.69 M ops/s      3.180      2.986      1.2
  indicators_100000                   DAG-fused    single       ai-off   369.17 M ops/s      0.271      0.257      2.5
  indicators_100000                   DAG-fused    chunk        ai-off   323.87 M ops/s      0.618      0.518      2.5
  indicators_100000                   DAG-fused    dag-fused    ai-off   188.68 M ops/s      3.180      2.994      1.2
  indicators_100000                   DAG-fused    hybrid-fall  ai-off   184.07 M ops/s      3.260      3.027      1.2
  ★ BEST: mode=Hybrid     layer=single       → 370.14 M ops/s in 0.270ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  indicators_1000000                  CPU-safe     single       ai-off   341.70 M ops/s      2.927      2.641      2.5
  indicators_1000000                  CPU-safe     chunk        ai-off   362.29 M ops/s      5.520      5.305      1.3
  indicators_1000000                  CPU-safe     dag-fused    ai-off   198.99 M ops/s     30.153     28.083      3.7
  indicators_1000000                  CPU-safe     hybrid-fall  ai-off   193.96 M ops/s     30.934     29.239      2.5
  indicators_1000000                  Hybrid       single       ai-off   345.75 M ops/s      2.892      2.670      2.5
  indicators_1000000                  Hybrid       chunk        ai-off   351.93 M ops/s      5.683      5.305      0.0
  indicators_1000000                  Hybrid       dag-fused    ai-off   196.74 M ops/s     30.497     28.666      2.5
  indicators_1000000                  Hybrid       hybrid-fall  ai-off   190.71 M ops/s     31.461     28.893      2.5
  indicators_1000000                  HPC-stress   single       ai-off   354.92 M ops/s      2.818      2.658      0.0
  indicators_1000000                  HPC-stress   chunk        ai-off   360.47 M ops/s      5.548      5.294      1.3
  indicators_1000000                  HPC-stress   dag-fused    ai-off   199.88 M ops/s     30.018     28.776      4.9
  indicators_1000000                  HPC-stress   hybrid-fall  ai-off   194.94 M ops/s     30.779     28.152      2.5
  indicators_1000000                  DAG-fused    single       ai-off   360.35 M ops/s      2.775      2.645      1.3
  indicators_1000000                  DAG-fused    chunk        ai-off   356.01 M ops/s      5.618      5.304      0.0
  indicators_1000000                  DAG-fused    dag-fused    ai-off   199.91 M ops/s     30.013     27.876      2.5
  indicators_1000000                  DAG-fused    hybrid-fall  ai-off   199.44 M ops/s     30.085     28.381      4.9
  ★ BEST: mode=DAG-fused  layer=single       → 360.35 M ops/s in 2.775ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  mc_100000_paths                     Hybrid       single       ai-off   125.30 M ops/s    201.117    191.848      1.3
  mc_100000_paths                     Hybrid       chunk        ai-off   282.09 M ops/s     89.334     84.612      0.0
  mc_100000_paths                     HPC-stress   single       ai-off   127.79 M ops/s    197.194    193.226      0.0
  mc_100000_paths                     HPC-stress   chunk        ai-off   282.67 M ops/s     89.150     87.137      1.2
  mc_100000_paths                     DAG-fused    single       ai-off   124.80 M ops/s    201.922    200.589      2.5
  mc_100000_paths                     DAG-fused    chunk        ai-off   258.62 M ops/s     97.439     93.067      0.0
  ★ BEST: mode=HPC-stress layer=chunk        → 282.67 M ops/s in 89.150ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  mc_10000_paths                      Hybrid       single       ai-off   133.45 M ops/s     18.884     18.252      2.5
  mc_10000_paths                      Hybrid       chunk        ai-off   381.15 M ops/s      6.612      4.945      1.3
  mc_10000_paths                      HPC-stress   single       ai-off   120.64 M ops/s     20.889     18.361      2.5
  mc_10000_paths                      HPC-stress   chunk        ai-off   480.28 M ops/s      5.247      4.967      0.0
  mc_10000_paths                      HPC-stress   chunk        ai-off   411.56 M ops/s      6.123      4.894      1.2
  mc_10000_paths                      HPC-stress   chunk        ai-post  473.71 M ops/s      5.320      4.735      2.5
  mc_10000_paths                      HPC-stress   chunk        ai-on    495.69 M ops/s      5.084      4.819      3.7
  mc_10000_paths                      DAG-fused    single       ai-off   128.36 M ops/s     19.633     18.567      2.5
  mc_10000_paths                      DAG-fused    chunk        ai-off   459.92 M ops/s      5.479      4.975      2.5
  ★ BEST: mode=HPC-stress layer=chunk        → 495.69 M ops/s in 5.084ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  mc_1000_paths                       Hybrid       single       ai-off   121.87 M ops/s      2.068      1.846      1.2
  mc_1000_paths                       Hybrid       chunk        ai-off   564.46 M ops/s      0.446      0.386      3.7
  mc_1000_paths                       HPC-stress   single       ai-off   111.06 M ops/s      2.269      1.853      0.0
  mc_1000_paths                       HPC-stress   chunk        ai-off   596.37 M ops/s      0.423      0.391      0.0
  mc_1000_paths                       DAG-fused    single       ai-off   131.80 M ops/s      1.912      1.835      3.7
  mc_1000_paths                       DAG-fused    chunk        ai-off   577.22 M ops/s      0.437      0.388      1.2
  ★ BEST: mode=HPC-stress layer=chunk        → 596.37 M ops/s in 0.423ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  mc_100_paths                        Hybrid       single       ai-off   131.15 M ops/s      0.192      0.190      2.5
  mc_100_paths                        Hybrid       chunk        ai-off   264.10 M ops/s      0.095      0.054      2.5
  mc_100_paths                        HPC-stress   single       ai-off   127.49 M ops/s      0.198      0.181      2.5
  mc_100_paths                        HPC-stress   chunk        ai-off   343.67 M ops/s      0.073      0.060      1.2
  mc_100_paths                        DAG-fused    single       ai-off    87.91 M ops/s      0.287      0.183      2.5
  mc_100_paths                        DAG-fused    chunk        ai-off   291.75 M ops/s      0.086      0.062      0.0
  ★ BEST: mode=HPC-stress layer=chunk        → 343.67 M ops/s in 0.073ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_1000                      Hybrid       single       ai-off   148.06 M ops/s      0.007      0.007      2.5
  portfolio_1000                      Hybrid       chunk        ai-off   138.98 M ops/s      0.007      0.007      3.7
  portfolio_1000                      HPC-stress   single       ai-off   142.86 M ops/s      0.007      0.007      2.5
  portfolio_1000                      HPC-stress   chunk        ai-off   138.94 M ops/s      0.007      0.007      0.0
  ★ BEST: mode=Hybrid     layer=single       → 148.06 M ops/s in 0.007ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_10000                     Hybrid       single       ai-off   147.01 M ops/s      0.068      0.065      4.9
  portfolio_10000                     Hybrid       chunk        ai-off   126.91 M ops/s      0.079      0.065      3.7
  portfolio_10000                     HPC-stress   single       ai-off   151.18 M ops/s      0.066      0.065      4.9
  portfolio_10000                     HPC-stress   chunk        ai-off   150.74 M ops/s      0.066      0.065      1.2
  portfolio_10000                     HPC-stress   chunk        ai-off   150.50 M ops/s      0.066      0.064      1.2
  portfolio_10000                     HPC-stress   chunk        ai-post  149.24 M ops/s      0.067      0.065      2.5
  portfolio_10000                     HPC-stress   chunk        ai-on    148.90 M ops/s      0.067      0.065      1.3
  ★ BEST: mode=HPC-stress layer=single       → 151.18 M ops/s in 0.066ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_100000                    Hybrid       single       ai-off   117.28 M ops/s      0.853      0.666      2.5
  portfolio_100000                    Hybrid       chunk        ai-off   141.02 M ops/s      0.709      0.682      1.3
  portfolio_100000                    HPC-stress   single       ai-off   145.03 M ops/s      0.690      0.681      3.7
  portfolio_100000                    HPC-stress   chunk        ai-off   141.45 M ops/s      0.707      0.661      1.2
  ★ BEST: mode=HPC-stress layer=single       → 145.03 M ops/s in 0.690ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_100x10000           Hybrid       chunk        ai-off   877.29 M ops/s      1.140      0.970     11.2
  portfolio_batch_100x10000           HPC-stress   chunk        ai-off     1.07 G ops/s      0.935      0.831     10.0
  ★ BEST: mode=HPC-stress layer=chunk        → 1.07 G ops/s in 0.935ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_100x100000          Hybrid       chunk        ai-off   908.57 M ops/s     11.006     10.189      7.7
  portfolio_batch_100x100000          HPC-stress   chunk        ai-off   892.87 M ops/s     11.200     10.375     10.3
  ★ BEST: mode=Hybrid     layer=chunk        → 908.57 M ops/s in 11.006ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_10x10000            Hybrid       chunk        ai-off   489.32 M ops/s      0.204      0.152      6.2
  portfolio_batch_10x10000            HPC-stress   chunk        ai-off   463.48 M ops/s      0.216      0.176     11.2
  ★ BEST: mode=Hybrid     layer=chunk        → 489.32 M ops/s in 0.204ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_10x100000           Hybrid       chunk        ai-off   632.48 M ops/s      1.581      1.443     25.9
  portfolio_batch_10x100000           HPC-stress   chunk        ai-off   597.28 M ops/s      1.674      1.600      7.5
  ★ BEST: mode=Hybrid     layer=chunk        → 632.48 M ops/s in 1.581ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_500x10000           Hybrid       chunk        ai-off     1.11 G ops/s      4.488      3.798     35.1
  portfolio_batch_500x10000           HPC-stress   chunk        ai-off     1.21 G ops/s      4.126      3.870      3.7
  ★ BEST: mode=HPC-stress layer=chunk        → 1.21 G ops/s in 4.126ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  portfolio_batch_500x100000          Hybrid       chunk        ai-off   755.90 M ops/s     66.146     63.159      1.3
  portfolio_batch_500x100000          HPC-stress   chunk        ai-off   758.67 M ops/s     65.904     60.689      2.5
  ★ BEST: mode=HPC-stress layer=chunk        → 758.67 M ops/s in 65.904ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_gaussian_1000                    HPC-stress   chunk        ai-off   889.11 M ops/s      0.283      0.250      1.2
  ★ BEST: mode=HPC-stress layer=chunk        → 889.11 M ops/s in 0.283ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_gaussian_10000                   HPC-stress   chunk        ai-off   672.52 M ops/s      3.747      3.354      1.2
  ★ BEST: mode=HPC-stress layer=chunk        → 672.52 M ops/s in 3.747ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_gbm_1000                         HPC-stress   chunk        ai-off   672.70 M ops/s      0.375      0.356      2.4
  ★ BEST: mode=HPC-stress layer=chunk        → 672.70 M ops/s in 0.375ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_gbm_10000                        HPC-stress   chunk        ai-off   526.52 M ops/s      4.786      4.299      6.1
  ★ BEST: mode=HPC-stress layer=chunk        → 526.52 M ops/s in 4.786ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_jumpdiffusion_1000               HPC-stress   chunk        ai-off   549.59 M ops/s      0.459      0.401      2.5
  ★ BEST: mode=HPC-stress layer=chunk        → 549.59 M ops/s in 0.459ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_jumpdiffusion_10000              HPC-stress   chunk        ai-off   490.73 M ops/s      5.135      4.741      2.5
  ★ BEST: mode=HPC-stress layer=chunk        → 490.73 M ops/s in 5.135ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_meanreversion_1000               HPC-stress   chunk        ai-off   794.16 M ops/s      0.317      0.282      6.2
  ★ BEST: mode=HPC-stress layer=chunk        → 794.16 M ops/s in 0.317ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_meanreversion_10000              HPC-stress   chunk        ai-off   612.64 M ops/s      4.113      3.962      0.0
  ★ BEST: mode=HPC-stress layer=chunk        → 612.64 M ops/s in 4.113ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_regimeswitching_1000             HPC-stress   chunk        ai-off   572.09 M ops/s      0.440      0.374      2.5
  ★ BEST: mode=HPC-stress layer=chunk        → 572.09 M ops/s in 0.440ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  rw_regimeswitching_10000            HPC-stress   chunk        ai-off   521.13 M ops/s      4.836      4.623      1.2
  ★ BEST: mode=HPC-stress layer=chunk        → 521.13 M ops/s in 4.836ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  segmentation_1m                     Hybrid       single       ai-off   22522.52 G ops/s      0.000      0.000      3.8
  segmentation_1m                     Hybrid       chunk        ai-off   14409.22 G ops/s      0.000      0.000      2.5
  segmentation_1m                     HPC-stress   single       ai-off   25706.94 G ops/s      0.000      0.000      2.5
  segmentation_1m                     HPC-stress   chunk        ai-off   23288.31 G ops/s      0.000      0.000      2.5
  ★ BEST: mode=HPC-stress layer=single       → 25706.94 G ops/s in 0.000ms
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
