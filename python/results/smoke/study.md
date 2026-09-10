# Parameter study

126 configurations: 7 barrier heights × 6 receiver distances × 3 wedge angles.

## Marginal means at 500 Hz

A fully crossed grid, so the mean at each level of a factor is directly
comparable across factors — which a one-at-a-time sweep would not give.

**height**

| Level | Mean IL (dB) |
|---|---:|
| 2 | 8.40 |
| 2.5 | 11.04 |
| 3 | 13.13 |
| 3.5 | 14.75 |
| 4 | 16.03 |
| 5 | 17.94 |
| 6 | 19.37 |

**receiver distance**

| Level | Mean IL (dB) |
|---|---:|
| 5 | 15.36 |
| 10 | 14.74 |
| 20 | 14.30 |
| 40 | 14.04 |
| 60 | 13.95 |
| 80 | 13.90 |

**wedge angle**

| Level | Mean IL (dB) |
|---|---:|
| 270° wedge | 12.83 |
| 315° wedge | 14.65 |
| thin screen | 15.67 |

## Extremes, A-weighted

| Configuration | Height (m) | Distance (m) | Wedge | δ (m) | A-weighted IL (dB) |
|---|---:|---:|---|---:|---:|
| `h6-d5-w360` | 6 | 5 | 360° | 3.454 | 27.70 |
| `h6-d10-w360` | 6 | 10 | 360° | 2.693 | 26.36 |
| `h6-d5-w315` | 6 | 5 | 315° | 3.454 | 25.68 |
| `h5-d5-w360` | 5 | 5 | 360° | 2.207 | 25.61 |
| `h6-d20-w360` | 6 | 20 | 360° | 2.227 | 25.46 |
| `h2-d10-w270` | 2 | 10 | 270° | 0.037 | 11.10 |
| `h2-d20-w270` | 2 | 20 | 270° | 0.031 | 10.88 |
| `h2-d40-w270` | 2 | 40 | 270° | 0.028 | 10.75 |
| `h2-d60-w270` | 2 | 60 | 270° | 0.027 | 10.70 |
| `h2-d80-w270` | 2 | 80 | 270° | 0.027 | 10.68 |
