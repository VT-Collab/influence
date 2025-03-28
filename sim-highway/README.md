This directory provides the implemetnation details for the highway environment as discussed in Section $6.1$ of the paper.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/1c08e434-4f20-4a05-9bd5-b22f63d56cdc" style="width: 400px; height: auto; margin: 0 10px;">
</div>

## Environment
In this environment the simulated humans drives alongside an autonomous car on a highway environment. The autonomous car's objective is to safely slow down the human driver.
The simulated human attempts to get around the autonomous car and maximize their lane progress by following.

### Instructions
Run the simulation using 

```
julia highway-unified-sim.jl
```

We ran our simulations using [Julia 1.11.3](https://julialang.org/downloads/).

## Interactions
The arguments `--int_multiplier` controls the number of interactions. We report results from $100$ interactions which is the default value for this argument.

### Collecting starting positions
To compare this script with other methods run 

```
random_positions.jl
``` 

to generate random starting positions as a `JSON` file.

### Analysis and Plotting
Results will be saved in the folder `unified-hw-res`. To analyze the the interaction data run 

```
julia plot_data.jl
```

The varibale `inte_to_read` needs to be adjusted based on the number of interactions. Its default value is $100$ based on the number of interactions in our paper.
This script will genrate EXCEL files for SPSS analysis and SVG plots for simulated human car Y position, robot reward, collisions, robot Car Y position, simulated human interaction score. It also plots the trajectories for each interaction in the `trajs` folder. Data structure of the `JSON` files is provided as a comment at the begining of this script.
