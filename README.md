# Dynamic Transport Route Designer

## Project Overview

The idea behind this project was simple: could bus routes be designed my machine learning every week to make them more efficient?
My team and then created a life-like dataset and a pipeline that would attempt to predict ridership and then create routes to satisfy those trips.

## Components
1. Synthetic Data Generator (SynthDataGen)
2. Ridership Predictor (RIPR)
3. Ridership Count Predictor (RCP)
4. Route Designer (RoDes)
    - Rodes-a: 3 algorithms & heuristics that determine routes.
    - Rodes-b: machine learning clustering + algorithm that determines routes.
  
## Results

In the end we were able to predict origin-destination ridership percentage odds within our target for MAE (1.83e-5) and reduce average weight time in our city by 13% while increasing vehicle-km negligably (<0.6%).
You can find our final presentation on the topic here: [OneDrive](https://1drv.ms/p/c/86328ab4ef0062fc/IQC6lVEH0ubbS6WgZ-hp35mRAWWEaIdc4csTWEj34JkgH9Y?e=j1RgCn).

Here are 2 of our generated routes using 2 methods (left) and the city's current routes (right).
<img width="1237" height="716" alt="image" src="https://github.com/user-attachments/assets/dbc8951c-d9c9-4251-9156-70b6dded3552" />

## Future Work

This project is in no way done, and our approach was very naive. If I were to revive the project I would focus on the following:
- Gaining access to real smart-card origin-destination data would improve the project tremendously. That or spending much more time on improving the quality of the data generation. Our system took into account flows of people through sectors of the city which was based on real London data, but we couldn't test if our system could pick up on hidden behaviours in the real world since, naturally, we couldn't have "hidden" factors in the generated dataset.
- Avoid MSE/AE for error targets. Because the distribution had ~2000x2000 pairs it was hard to reason with our error values. Although they converged, they are biased some 2.5e-7 down just from the size of the origin-destination space. Instead we should use KL-divergence or something else more suited for distributions.
- Train a shallow NN instead of the MLR grid. This was originally the intention of the second phase of RIPR but this was dropped due to time concerns. Especially on real data this could be really useful as there are likely strong interactions between nearby stops' ridership.
