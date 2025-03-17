This directory provides the implemetnation details for the Circle environment as discussed in Section $6.2$ of the paper.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/da19c46d-0893-4d1d-ba8d-1e3a509ad744" style="width: 388px; height: auto; margin: 0 10px;">
</div>

## Environment
The Circle environment is an instance of [pursuit-evasion games](https://ieeexplore.ieee.org/abstract/document/1067989) with two-dimensional states and actions. The robot agent (i.e., the pursuer) tries to reach the simulated human agent (i.e., the evader). To avoid the robot the human moves along the circumference of the circle. Here $z$ encodes the human's location along the circle, and $\phi$ governs how the human moves along the circle to avoid the robot. For example, the human evader might move away from the robot's previous position. The robot's reward is its negative distance from the simulated human.

### Instructions
Run the simulation using 

```
julia run_experiment.jl
```

We ran our simulations using [Julia 1.11.3](https://julialang.org/downloads/).

## Interactions
The variable `num_experiments` controls the number of experiments. We report results from $50$ episodes which is the default value for this variable.

### Analysis and Plotting
Results from the experiments are saved in two folders. `data` has the CSV data for each interaction, `figures` has rewards plots and video of agents for each interaction . To see the overall results for all experiments run

```
julia plot_data.jl
```
Overall results will be added to `figures` folder.
