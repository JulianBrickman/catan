# Training Run Comparison

This file is auto-generated from `training_runs/` and is meant to make session-to-session comparison easy.

## Session Summary

| Session | Best Update | Eval 1st Place | Eval Truncations | Eval Final VP | Eval Turns | Resume From | Max Turns | Rollout Episodes | Eval Episodes | PPO Epochs | Reward Placement | Reward VP Gain | Reward Final VP | Reward Win |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training_session_001_20260409_141409 | 4 | 0.0000 | 2 | 3.6250 | 1000.00 | /tmp/catan_resume_test/training_session_001_20260409_135900/checkpoints/checkpoint_0001.pt | 1000 | 2 | 2 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| training_session_002_20260409_142708 | 1 | 0.0000 | 1 | 2.0000 | 10.00 |  | 10 | 1 | 1 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| training_session_003_20260409_143032 | 4 | 1.0000 | 0 | 6.8750 | 505.50 | training_runs/training_session_002_20260409_142708/checkpoints/checkpoint_0001.pt | 1000 | 2 | 2 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| training_session_005_20260409_152759 | 15 | 0.7500 | 1 | 6.1875 | 674.75 | training_runs/training_session_003_20260409_143032/checkpoints/checkpoint_0004.pt | 1000 | 8 | 4 | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| training_session_006_20260409_162036 | 7 | 0.7500 | 2 | 5.4375 | 599.88 | training_runs/training_session_003_20260409_143032/checkpoints/checkpoint_0004.pt | 1000 | 4 | 8 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| training_session_007_20260410_103344 | 11 | 1.0000 | 0 | 4.9062 | 275.62 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0009.pt | 1000 | 4 | 8 | 1 | 1.0000 | 0.0500 | 0.0000 | 0.0000 |
| training_session_008_20260410_110113 | 22 | 1.0000 | 0 | 5.5312 | 352.50 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0013.pt | 1000 | 6 | 8 | 1 | 1.0000 | 0.0800 | 0.0000 | 0.0000 |
| training_session_009_20260410_130216 | 30 | 0.8750 | 1 | 5.3750 | 323.38 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0023.pt | 800 | 6 | 8 | 1 | 1.0000 | 0.1000 | 0.0000 | 0.0000 |
| training_session_010_20260410_135342 | 30 | 0.8750 | 1 | 5.3750 | 304.62 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0029.pt | 650 | 6 | 8 | 1 | 1.0000 | 0.1000 | 0.0000 | 0.0000 |
| training_session_011_20260410_142828 | 31 | 1.0000 | 0 | 4.6562 | 256.38 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0029.pt | 800 | 6 | 8 | 1 | 1.0000 | 0.1000 | 0.0000 | 0.0000 |
| training_session_012_20260410_152524 | 34 | 0.7500 | 4 | 5.1406 | 463.06 | training_runs/training_session_011_20260410_142828/checkpoints/checkpoint_0031.pt | 800 | 6 | 16 | 1 | 1.0000 | 0.1000 | 0.0000 | 0.0000 |
| training_session_013_20260410_153747 | 34 | 0.6875 | 5 | 5.1250 | 519.88 | training_runs/training_session_011_20260410_142828/checkpoints/checkpoint_0031.pt | 800 | 6 | 16 | 1 | 1.0000 | 0.0800 | 0.0000 | 0.0000 |

## Update Summary

| Session | Update | Eval 1st Place | Eval Truncations | Eval Final VP | Eval Turns | Train Final VP | Train Turns | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| training_session_001_20260409_141409 | 2 | 0.0000 | 2 | 2.0000 | 1000.00 | 5.6250 | 208.50 | training_runs/training_session_001_20260409_141409/checkpoints/checkpoint_0002.pt |
| training_session_001_20260409_141409 | 3 | 0.0000 | 2 | 2.0000 | 1000.00 | 5.8750 | 258.00 | training_runs/training_session_001_20260409_141409/checkpoints/checkpoint_0003.pt |
| training_session_001_20260409_141409 | 4 | 0.0000 | 2 | 3.6250 | 1000.00 | 5.5000 | 305.50 | training_runs/training_session_001_20260409_141409/checkpoints/checkpoint_0004.pt |
| training_session_002_20260409_142708 | 1 | 0.0000 | 1 | 2.0000 | 10.00 | 2.0000 | 10.00 | training_runs/training_session_002_20260409_142708/checkpoints/checkpoint_0001.pt |
| training_session_003_20260409_143032 | 2 | 1.0000 | 0 | 4.6250 | 258.50 | 6.3750 | 351.50 | training_runs/training_session_003_20260409_143032/checkpoints/checkpoint_0002.pt |
| training_session_003_20260409_143032 | 3 | 0.0000 | 2 | 2.8750 | 1000.00 | 6.1250 | 281.50 | training_runs/training_session_003_20260409_143032/checkpoints/checkpoint_0003.pt |
| training_session_003_20260409_143032 | 4 | 1.0000 | 0 | 6.8750 | 505.50 | 7.0000 | 342.00 | training_runs/training_session_003_20260409_143032/checkpoints/checkpoint_0004.pt |
| training_session_005_20260409_152759 | 5 | 0.2500 | 3 | 5.1875 | 828.25 | 5.9688 | 290.75 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0005.pt |
| training_session_005_20260409_152759 | 6 | 0.2500 | 3 | 4.3750 | 833.00 | 6.2188 | 332.25 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0006.pt |
| training_session_005_20260409_152759 | 7 | 0.2500 | 3 | 4.2500 | 987.00 | 6.0312 | 303.62 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0007.pt |
| training_session_005_20260409_152759 | 8 | 0.7500 | 1 | 4.3125 | 668.75 | 6.2812 | 338.38 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0008.pt |
| training_session_005_20260409_152759 | 9 | 0.5000 | 2 | 5.4375 | 716.25 | 5.4062 | 205.12 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0009.pt |
| training_session_005_20260409_152759 | 10 | 0.5000 | 2 | 5.8125 | 727.25 | 5.7812 | 216.25 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0010.pt |
| training_session_005_20260409_152759 | 11 | 0.2500 | 3 | 4.9375 | 933.50 | 5.9375 | 215.25 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0011.pt |
| training_session_005_20260409_152759 | 12 | 0.0000 | 4 | 4.0625 | 1000.00 | 6.3125 | 287.12 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0012.pt |
| training_session_005_20260409_152759 | 13 | 0.5000 | 2 | 4.3750 | 763.25 | 5.9375 | 232.25 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0013.pt |
| training_session_005_20260409_152759 | 14 | 0.7500 | 1 | 4.1875 | 520.75 | 5.3125 | 212.62 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0014.pt |
| training_session_005_20260409_152759 | 15 | 0.7500 | 1 | 6.1875 | 674.75 | 6.3125 | 205.50 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0015.pt |
| training_session_005_20260409_152759 | 16 | 0.5000 | 2 | 4.8125 | 816.50 | 5.8125 | 283.12 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0016.pt |
| training_session_005_20260409_152759 | 17 | 0.7500 | 1 | 4.6250 | 469.75 | 5.8438 | 216.50 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0017.pt |
| training_session_005_20260409_152759 | 18 | 0.2500 | 3 | 3.9375 | 878.75 | 6.2188 | 271.62 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0018.pt |
| training_session_005_20260409_152759 | 19 | 0.5000 | 2 | 5.0625 | 674.00 | 6.0625 | 218.12 | training_runs/training_session_005_20260409_152759/checkpoints/checkpoint_0019.pt |
| training_session_006_20260409_162036 | 5 | 0.5000 | 4 | 5.4375 | 677.12 | 5.7500 | 356.25 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0005.pt |
| training_session_006_20260409_162036 | 6 | 0.2500 | 6 | 5.8438 | 863.00 | 5.8125 | 323.75 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0006.pt |
| training_session_006_20260409_162036 | 7 | 0.7500 | 2 | 5.4375 | 599.88 | 6.1250 | 364.25 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0007.pt |
| training_session_006_20260409_162036 | 8 | 0.6250 | 3 | 4.3125 | 724.62 | 6.4375 | 355.75 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0008.pt |
| training_session_006_20260409_162036 | 9 | 0.7500 | 2 | 4.7500 | 550.88 | 6.1875 | 260.75 | training_runs/training_session_006_20260409_162036/checkpoints/checkpoint_0009.pt |
| training_session_007_20260410_103344 | 10 | 0.7500 | 2 | 4.1562 | 453.50 | 7.0625 | 370.50 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0010.pt |
| training_session_007_20260410_103344 | 11 | 1.0000 | 0 | 4.9062 | 275.62 | 5.6875 | 301.75 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0011.pt |
| training_session_007_20260410_103344 | 12 | 0.5000 | 4 | 4.8750 | 665.38 | 6.2500 | 306.75 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0012.pt |
| training_session_007_20260410_103344 | 13 | 0.8750 | 1 | 5.0625 | 339.38 | 5.9375 | 326.00 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0013.pt |
| training_session_007_20260410_103344 | 14 | 0.6250 | 3 | 4.5625 | 527.62 | 5.8125 | 303.50 | training_runs/training_session_007_20260410_103344/checkpoints/checkpoint_0014.pt |
| training_session_008_20260410_110113 | 14 | 0.6250 | 3 | 4.9062 | 634.88 | 6.1250 | 244.17 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0014.pt |
| training_session_008_20260410_110113 | 15 | 0.6250 | 3 | 5.0625 | 641.50 | 5.8750 | 260.00 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0015.pt |
| training_session_008_20260410_110113 | 16 | 0.7500 | 2 | 4.5312 | 552.88 | 6.5833 | 276.17 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0016.pt |
| training_session_008_20260410_110113 | 17 | 0.1250 | 7 | 3.6875 | 911.88 | 5.3750 | 208.17 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0017.pt |
| training_session_008_20260410_110113 | 18 | 0.3750 | 5 | 4.1250 | 770.38 | 6.0417 | 288.50 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0018.pt |
| training_session_008_20260410_110113 | 19 | 0.6250 | 3 | 4.8125 | 581.50 | 6.2500 | 263.17 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0019.pt |
| training_session_008_20260410_110113 | 20 | 0.2500 | 6 | 3.9375 | 927.25 | 5.6667 | 204.50 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0020.pt |
| training_session_008_20260410_110113 | 21 | 0.3750 | 5 | 3.8125 | 873.62 | 6.0833 | 254.67 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0021.pt |
| training_session_008_20260410_110113 | 22 | 1.0000 | 0 | 5.5312 | 352.50 | 5.4583 | 176.83 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0022.pt |
| training_session_008_20260410_110113 | 23 | 1.0000 | 0 | 5.2500 | 339.25 | 5.5417 | 193.33 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0023.pt |
| training_session_008_20260410_110113 | 24 | 0.5000 | 4 | 4.9688 | 761.12 | 5.7500 | 246.33 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0024.pt |
| training_session_008_20260410_110113 | 25 | 0.8750 | 1 | 5.0938 | 446.88 | 5.8333 | 280.67 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0025.pt |
| training_session_008_20260410_110113 | 26 | 0.7500 | 2 | 5.0625 | 569.12 | 5.6250 | 191.83 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0026.pt |
| training_session_008_20260410_110113 | 27 | 0.5000 | 4 | 4.7188 | 699.50 | 5.7500 | 212.67 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0027.pt |
| training_session_008_20260410_110113 | 28 | 0.8750 | 1 | 5.0625 | 434.50 | 5.7500 | 270.83 | training_runs/training_session_008_20260410_110113/checkpoints/checkpoint_0028.pt |
| training_session_009_20260410_130216 | 24 | 0.1250 | 7 | 4.8125 | 728.62 | 5.3333 | 178.50 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0024.pt |
| training_session_009_20260410_130216 | 25 | 0.8750 | 1 | 4.9375 | 419.12 | 6.7500 | 301.17 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0025.pt |
| training_session_009_20260410_130216 | 26 | 0.6250 | 3 | 4.7188 | 530.25 | 5.0833 | 197.00 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0026.pt |
| training_session_009_20260410_130216 | 27 | 0.7500 | 2 | 4.2812 | 324.38 | 5.8750 | 282.83 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0027.pt |
| training_session_009_20260410_130216 | 28 | 0.7500 | 2 | 4.9688 | 460.00 | 6.0833 | 265.17 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0028.pt |
| training_session_009_20260410_130216 | 29 | 0.8750 | 1 | 5.0938 | 299.25 | 5.1667 | 164.17 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0029.pt |
| training_session_009_20260410_130216 | 30 | 0.8750 | 1 | 5.3750 | 323.38 | 6.3750 | 275.67 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0030.pt |
| training_session_009_20260410_130216 | 31 | 0.5000 | 4 | 4.0000 | 513.25 | 5.5000 | 154.83 | training_runs/training_session_009_20260410_130216/checkpoints/checkpoint_0031.pt |
| training_session_010_20260410_135342 | 30 | 0.8750 | 1 | 5.3750 | 304.62 | 5.1667 | 179.67 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0030.pt |
| training_session_010_20260410_135342 | 31 | 0.7500 | 2 | 4.4062 | 362.75 | 5.9583 | 214.00 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0031.pt |
| training_session_010_20260410_135342 | 32 | 0.3750 | 5 | 4.5938 | 511.50 | 5.9167 | 268.17 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0032.pt |
| training_session_010_20260410_135342 | 33 | 0.6250 | 3 | 4.0000 | 431.50 | 6.1667 | 168.17 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0033.pt |
| training_session_010_20260410_135342 | 34 | 0.5000 | 4 | 4.9375 | 529.38 | 5.9583 | 258.83 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0034.pt |
| training_session_010_20260410_135342 | 35 | 0.5000 | 4 | 4.8750 | 534.88 | 5.0000 | 145.83 | training_runs/training_session_010_20260410_135342/checkpoints/checkpoint_0035.pt |
| training_session_011_20260410_142828 | 30 | 0.6250 | 3 | 4.8438 | 529.75 | 5.7083 | 181.00 | training_runs/training_session_011_20260410_142828/checkpoints/checkpoint_0030.pt |
| training_session_011_20260410_142828 | 31 | 1.0000 | 0 | 4.6562 | 256.38 | 5.0417 | 195.33 | training_runs/training_session_011_20260410_142828/checkpoints/checkpoint_0031.pt |
| training_session_011_20260410_142828 | 32 | 0.6250 | 3 | 5.7812 | 579.62 | 6.2500 | 207.00 | training_runs/training_session_011_20260410_142828/checkpoints/checkpoint_0032.pt |
| training_session_012_20260410_152524 | 32 | 0.6250 | 6 | 4.7188 | 568.25 | 6.9583 | 252.17 | training_runs/training_session_012_20260410_152524/checkpoints/checkpoint_0032.pt |
| training_session_012_20260410_152524 | 33 | 0.4375 | 9 | 4.2188 | 599.81 | 5.8750 | 207.50 | training_runs/training_session_012_20260410_152524/checkpoints/checkpoint_0033.pt |
| training_session_012_20260410_152524 | 34 | 0.7500 | 4 | 5.1406 | 463.06 | 5.8333 | 170.83 | training_runs/training_session_012_20260410_152524/checkpoints/checkpoint_0034.pt |
| training_session_013_20260410_153747 | 32 | 0.5625 | 7 | 4.9531 | 538.81 | 5.6667 | 196.33 | training_runs/training_session_013_20260410_153747/checkpoints/checkpoint_0032.pt |
| training_session_013_20260410_153747 | 33 | 0.4375 | 9 | 4.4844 | 585.94 | 5.7083 | 234.00 | training_runs/training_session_013_20260410_153747/checkpoints/checkpoint_0033.pt |
| training_session_013_20260410_153747 | 34 | 0.6875 | 5 | 5.1250 | 519.88 | 6.0000 | 203.17 | training_runs/training_session_013_20260410_153747/checkpoints/checkpoint_0034.pt |
| training_session_013_20260410_153747 | 35 | 0.5000 | 8 | 4.8281 | 562.81 | 5.7083 | 197.17 | training_runs/training_session_013_20260410_153747/checkpoints/checkpoint_0035.pt |
| training_session_013_20260410_153747 | 36 | 0.3125 | 11 | 4.3438 | 654.31 | 5.7083 | 182.33 | training_runs/training_session_013_20260410_153747/checkpoints/checkpoint_0036.pt |
