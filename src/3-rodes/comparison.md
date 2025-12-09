# Comparison
These are the measured metrics of the following designed paths:
- London's static system
- Our dynamic routing system + Clarke-Write Heuristic
- Our dynamic routing system + Google Ortools

All models are compared against eachother for the week of 2021-11-18.  
More comparisons should be made but inference on these models is very slow so we're only
able to do it for a couple of days.

All models are compared based on the following features:
- Total travel time (h).
- Average wait time (h).
- Vehicle-kilometers traveled (km).
- Bus utilization (% of available hours).
Lower is better for all metrics.

Absolute:
|   Model      | Total travel time (h) | Average wait time (h) | Vehicle-kilometers traveled (km) | Bus utilization (% of current estimated hours) | Bus utilization (ratio to static) |
| ------------ | --------------------- | --------------------- | -------------------------------- | ---------------------------------------------- | --------------------------------- |
| Static       | 16641.49              | 2.28                  | 416037.27                        | 1733.5%                                        | 1x                                |
| Clarke-Write | 82663.58              | 0.17                  | 2066564.38                       | 8610.7%                                        | 4.967x.                           |
| Google Ort.  | 161207.62             | 0.04                  | 4030190.41                       | 16792.5%                                       | 9.687x                            |

Normalized to Bus Utilization:
|   Model      | Total travel time (h) | Average wait time (h) | Vehicle-kilometers traveled per vehicle hour (km) | Bus utilization (% of current estimated hours. should be the same) | Bus utilization (ratio to static) |
| ------------ | --------------------- | --------------------- | --------------------------------------------------| ------------------------------------------------------------------ | --------------------------------- |
| Static       | 16641.49              | 2.28                  | 416037.27                                         | 1733.5%                                                            | 1x                                |
| Clarke-Write | 16642.55              | 0.84                  | 416058.86                                         | 1733.6%                                                            | 4.967x.                           |
| Google Ort.  | 16641.65              | 0.39                  | 416041.13                                         | 1733.5%                                                            | 9.687x                            |


