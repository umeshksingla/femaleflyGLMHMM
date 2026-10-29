#### Results

`python collect_k_sweep.py ./models/octladder_cv_wt_female/`

Held-out bits/s over Chance by number of states (K = 1 is the single-state GLM):
    flies  full model (median)  IQR low  IQR high  % of best K  HMM no inputs (median)  transitions gain (median)  gain vs previous K (median)  % flies better  Wilcoxon p
K                                                                                                                                                                         
1      81                    2     1.05       3.3        0.766                     NaN                        NaN                          NaN             NaN         NaN
2      81                  192      156       229         73.6                     191                      0.319                          190            98.8    5.57e-15
3      81                  219      184       262         84.1                     218                      0.277                         28.8            98.8    5.57e-15
4      81                  229      187       270         87.7                     229                      0.251                         9.51            98.8       6e-15
5      81                  239      201       277         91.7                     233                      0.345                         9.07            82.7    2.85e-10
6      81                  243      201       281         93.3                     243                      0.363                         3.49            87.7    3.22e-11
7      81                  245      203       283         93.9                     245                      0.354                         2.56            91.4    6.34e-09
8      81                  247      207       285         94.7                     247                       0.27                         2.06            75.3    2.93e-05
12     81                  253      212       300         97.2                     254                      0.377                         6.05            85.2    1.92e-10
20     81                  261      222       302          100                     258                      0.447                         5.82            96.3    3.91e-14

Each row’s p-value comes from a two-sided Wilcoxon signed-rank test on each fly’s held-out gain when going from the previous K in the table to this K, for the full model. It’s the same paired comparison as the “gain vs previous K” and “% flies better” columns.


`python collect_methods.py ./models/octladder_cv_wt_female/`

K=2: median 0.319, 73 of 81 flies better, p = 4.9e-12
K=3: median 0.277, 73 of 81 flies better, p = 5e-13
K=4: median 0.251, 70 of 81 flies better, p = 2.3e-12
K=5: median 0.345, 60 of 81 flies better, p = 3.1e-08
K=6: median 0.363, 61 of 81 flies better, p = 2.1e-07
K=7: median 0.354, 66 of 81 flies better, p = 2.6e-10
K=8: median 0.270, 63 of 81 flies better, p = 5.2e-08
K=20: median 0.447, 63 of 81 flies better, p = 2.8e-07

fold 0: reference (l1=1,l2=1e+08) +0.035,       chosen (l1=1,l2=100) +0.471,    so tr cues added +0.436 bits/s
fold 1: reference (l1=1,l2=1e+08) +0.019,       chosen (l1=1,l2=10000) +0.382,  so tr cues added +0.363 bits/s
fold 4: reference (l1=1,l2=1e+08) +0.020,       chosen (l1=1,l2=100) +0.793,    so tr cues added +0.773 bits/s
fold 2: reference (l1=1,l2=1e+08) +0.163,       chosen (l1=1,l2=100) +0.829,    so tr cues added +0.666 bits/s
fold 5: reference (l1=1,l2=1e+08) +0.027,       chosen (l1=1,l2=10000) +0.485,  so tr cues added +0.458 bits/s
fold 3: reference (l1=1,l2=1e+08) +0.022,       chosen (l1=1,l2=1000) +0.469,   so tr cues added +0.447 bits/s
fold 7: reference (l1=1,l2=1e+08) +0.026,       chosen (l1=1,l2=1000) +0.504,   so tr cues added +0.479 bits/s
fold 6: reference (l1=1,l2=1e+08) +0.018,       chosen (l1=1,l2=10000) +0.366,  so tr cues added +0.348 bits/s
fold 9: reference (l1=1,l2=1e+08) +0.019,       chosen (l1=1,l2=10000) +0.423,  so tr cues added +0.404 bits/s
fold 8: reference (l1=1,l2=1e+08) +0.016,       chosen (l1=1,l2=100) +0.335,    so tr cues added +0.319 bits/s
10 folds, largest gap 0.163 bits/s


`python collect_folds.py ./models/octladder_cv_wt_female 5`

Held-out flies, bits/s (per-fly paired differences):
                                      n  median  IQR low  IQR high  mean  % flies > 0  Wilcoxon p
id-glm-hmm - lrhmm (cue transitions) 81   0.251    0.119     0.383 0.241         86.4    2.26e-12
lrhmm - dghmm                        81   0.572    0.184      1.17 0.657         85.2    2.73e-08
id-glm-hmm - dghmm                   81   0.824    0.376      1.46 0.898         87.7    1.75e-09
glm - chance                         81       2     1.05       3.3  2.36         91.4    9.27e-14
dghmm - chance                       81     229      185       270   223          100    5.36e-15
id-glm-hmm - chance                  81     229      187       270   224          100    5.36e-15 

Wilcoxon p is for the paired diff.

Per fold (validation gains from tuning, test medians over that fold's flies), bits/s:
      ridge    L2  val lrhmm-dghmm  val id-lrhmm  flies  test id-lrhmm (median)  test lrhmm-dghmm (median)
fold                                                                                                      
0     1e+06 1e+05            0.569          0.22      9                   0.162                      0.518
1     1e+05   100             0.71         0.327      8                   0.176                      0.332
2     3e+04 1e+03            0.677         0.255      8                   0.278                      0.414
3       100 1e+03            0.814         0.311      8                   0.273                      0.943
4     3e+05 1e+04            0.726         0.379      8                   0.285                      0.421
5       100 1e+03             1.22         0.467      8                   0.251                      0.261
6     3e+05 1e+03            0.768          0.28      8                   0.234                      0.896
7     1e+05 1e+04            0.809         0.286      8                   0.379                      0.715
8     3e+04   100            0.715         0.237      8                   0.157                      0.887
9     1e+05 1e+04             0.76         0.259      8                   0.335                      0.664



`python collect_pearsoncorrelations.py ./models/octladder_cv_wt_female 5`
id-glm-hmm, K=5: 10 folds, 81 held-out flies
  measure  state  output  flies  test r (mean)  test r (SD)  test r (median)  test r^2 (mean, %)  train r (mean)  train r (SD)
  overall    NaN    None     81          0.213       0.0488            0.214                4.76           0.223        0.0478
by_output    NaN forward     81          0.529       0.0942            0.541                28.9           0.534        0.0917
by_output    NaN lateral     81         0.0663       0.0657           0.0631                0.87          0.0833        0.0607
by_output    NaN angular     81         0.0424       0.0517           0.0344               0.447          0.0507         0.054
 by_state      1    None     81         0.0874       0.0666           0.0787                1.21          0.0993        0.0629
 by_state      2    None     81          0.152       0.0652            0.165                2.73           0.156         0.056
 by_state      3    None     81         0.0642       0.0376           0.0635               0.554          0.0647        0.0364
 by_state      4    None     81          0.051       0.0509           0.0373               0.519          0.0571        0.0446
 by_state      5    None     81          0.183        0.181            0.145                6.65           0.179         0.122


`python collect_folds_auxem.py ./models/octladder_cv_wt_female`

{'chosen_C': [0.0001,
              9.999999999999999e-06,
              9.999999999999999e-06,
              1e-06,
              1e-07,
              9.999999999999999e-06,
              1e-06,
              1e-06,
              1e-06,
              1e-07],
 'cues_beyond_states': {'flies': 81,
                        'flies_better': 52,
                        'median_bits_per_s': 0.015675791598743542,
                        'wilcoxon_p': 0.024563170632345716},
 'full': {'test_average_precision': 0.07932204860597528,
          'test_bits_per_s_over_rate': 0.35835814465204324,
          'test_f1': 0.1422415839885545,
          'train_f1_mean': 0.1676024433835041},
 'states_only': {'test_average_precision': 0.068871794484513,
                 'test_bits_per_s_over_rate': 0.3457508730703292,
                 'test_f1': 0.12208844750145509,
                 'train_f1_mean': 0.1478969022004802},
 'test_flick_rate': 0.05294838591359035}


`plot_state_dwell_times()`
State 1: Mean dwell time = 128.77ms and STD = 146.56ms
State 2: Mean dwell time = 121.61ms and STD = 137.65ms
State 3: Mean dwell time = 90.42ms and STD = 99.61ms
State 4: Mean dwell time = 237.47ms and STD = 837.18ms
State 5: Mean dwell time = 232.08ms and STD = 772.56ms



`python collect_k_sweep.py ./models/octladder_cv_wt_fred_female/`
Held-out bits/s over Chance by number of states (K = 1 is the single-state GLM):
    flies  full model (median)  IQR low  IQR high  % of best K  HMM no inputs (median)  transitions gain (median)  gain vs previous K (median)  % flies better  Wilcoxon p
K                                                                                                                                                                         
1      11                 2.69     2.12      5.23         1.56                     NaN                        NaN                          NaN             NaN         NaN
2      11                  120     80.7       139         69.4                     117                     0.0726                          114             100    0.000977
3      11                  144     96.3       168         83.3                     142                     0.0328                         25.8            90.9     0.00195
4      11                  157      106       181         91.1                     154                     0.0506                         12.8             100    0.000977
5      11                  155      112       183         90.1                     155                     0.0601                         2.98            81.8      0.0322
6      11                  156      114       181         90.6                     155                     0.0451                         1.83            72.7       0.102
7      11                  167      126       183         96.7                     161                    0.00154                         1.51            81.8      0.0322
8      11                  158      126       180         91.4                     158                     0.0588                        0.109            54.5       0.898
12     11                  163      135       184         94.4                     164                   -0.00471                         4.97            90.9      0.0244
15     11                  170      139       185         98.5                     167                    -0.0147                         3.82             100    0.000977
20     11                  172      133       190          100                     175                     0.0309                         3.98            81.8       0.147


`python collect_methods.py ./models/octladder_cv_wt_fred_female/`
K=2: median 0.073, 9 of 11 flies better, p = 0.083
K=3: median 0.033, 7 of 11 flies better, p = 0.15
K=4: median 0.051, 8 of 11 flies better, p = 0.083
K=5: median 0.060, 11 of 11 flies better, p = 0.00098
K=6: median 0.045, 7 of 11 flies better, p = 0.24
K=7: median 0.002, 8 of 11 flies better, p = 0.32
K=8: median 0.059, 9 of 11 flies better, p = 0.083
K=12: median -0.005, 5 of 11 flies better, p = 0.52
K=15: median -0.015, 4 of 11 flies better, p = 0.7
K=20: median 0.031, 7 of 11 flies better, p = 0.21

fold 0: reference (l1=1,l2=1e+08) -0.012,       chosen (l1=1,l2=100000) +0.036, so tr cues added +0.048 bits/s
fold 1: reference (l1=1,l2=1e+08) -0.020,       chosen (l1=1,l2=10000) +0.058,  so tr cues added +0.078 bits/s
fold 2: reference (l1=1,l2=1e+08) +0.018,       chosen (l1=1,l2=100000) +0.241, so tr cues added +0.223 bits/s
fold 3: reference (l1=1,l2=1e+08) +0.084,       chosen (l1=1,l2=100000) +0.356, so tr cues added +0.271 bits/s
fold 4: reference (l1=1,l2=1e+08) -0.001,       chosen (l1=1,l2=10000) +0.185,  so tr cues added +0.187 bits/s
fold 5: reference (l1=1,l2=1e+08) +0.002,       chosen (l1=1,l2=1e+06) +0.015,  so tr cues added +0.013 bits/s
fold 6: reference (l1=1,l2=1e+08) -0.003,       chosen (l1=1,l2=100000) +0.042, so tr cues added +0.045 bits/s
fold 7: reference (l1=1,l2=1e+08) -0.001,       chosen (l1=1,l2=100000) +0.058, so tr cues added +0.059 bits/s
fold 8: reference (l1=1,l2=1e+08) -0.006,       chosen (l1=1,l2=10000) +0.073,  so tr cues added +0.079 bits/s
fold 9: reference (l1=1,l2=1e+08) +0.002,       chosen (l1=1,l2=1000) +0.156,   so tr cues added +0.154 bits/s
10 folds, largest gap 0.084 bits/s


`python collect_folds.py ./models/octladder_cv_wt_fred_female 5`

Held-out flies, bits/s (per-fly paired differences):
                                      n  median  IQR low  IQR high   mean  % flies > 0  Wilcoxon p
id-glm-hmm - lrhmm (cue transitions) 11  0.0601   0.0125     0.146 0.0943          100    0.000977
lrhmm - dghmm                        11  0.0989  -0.0649     0.721  0.304         72.7       0.365
id-glm-hmm - dghmm                   11   0.125   0.0447      0.79  0.398         81.8       0.175
glm - chance                         11    2.69     2.12      5.23   4.56         90.9     0.00293
dghmm - chance                       11     155      110       184    150          100    0.000977
id-glm-hmm - chance                  11     155      112       183    151          100    0.000977 

Per fold (validation gains from tuning, test medians over that fold's flies), bits/s:
      ridge    L2  val lrhmm-dghmm  val id-lrhmm  flies  test id-lrhmm (median)  test lrhmm-dghmm (median)
fold                                                                                                      
0     1e+08 1e+05          -0.0106        0.0361      2                    0.16                     0.0755
1     1e+04 1e+04             2.23        0.0576      1                   0.126                      0.388
2       100 1e+05             1.42         0.241      1                   0.153                       3.24
3     1e+06 1e+05             4.83         0.356      1                  0.0118                       1.05
4     1e+05 1e+04             2.81         0.185      1                   0.217                     -0.156
5     1e+06 1e+06            0.816        0.0149      1                  0.0133                     -0.663
6     1e+08 1e+05          -0.0103        0.0418      1                  0.0052                       0.12
7     1e+08 1e+05           0.0374        0.0584      1                 0.00222                      0.026
8     1e+05 1e+04             2.15        0.0732      1                  0.0501                       -3.1
9     1e+04 1e+03              1.8         0.156      1                   0.139                       2.28


`python collect_pearsoncorrelations.py ./models/octladder_cv_wt_fred_female 5`
id-glm-hmm, K=5: 10 folds, 11 held-out flies
  measure  state  output  flies  test r (mean)  test r (SD)  test r (median)  test r^2 (mean, %)  train r (mean)  train r (SD)
  overall    NaN    None     11          0.255       0.0857            0.236                7.24           0.298        0.0869
by_output    NaN forward     11          0.609       0.0997            0.607                38.1           0.626        0.0961
by_output    NaN lateral     11         0.0672        0.129           0.0441                2.11           0.145         0.135
by_output    NaN angular     11         0.0883        0.142           0.0237                2.81           0.121         0.124
 by_state      1    None     11          0.175        0.131            0.153                4.78           0.214         0.143
 by_state      2    None     11          0.204        0.108            0.206                5.34           0.254        0.0959
 by_state      3    None     11          0.165       0.0986            0.166                3.71           0.186        0.0899
 by_state      4    None     11         0.0434       0.0218           0.0439               0.236           0.058         0.024
 by_state      5    None     11         0.0603       0.0785           0.0281               0.981          0.0486        0.0387

