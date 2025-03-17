This directory provides the implemetnation details for the Driving environment as discussed in Section $6.2$ of the paper.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/a4b204f2-e240-4c9d-bd0a-c6f2b2043ba0" style="width: 600px; height: auto; margin: 0 10px;">
</div>

## Environment
In Driving an autonomous car is trying to pass a human vehicle. At every interaction the human starts out in front of the autonomous car, and changes lanes as the autonomous car attempts to pass. Here $z$ encodes the lane that the human merges into, and $\phi$ determines how the human selects that lane. For instance, the simulated human may merge into the lane where the autonomous car passed at the previous interaction.
The robot is rewarded for passing the human, and penalized for crashing with the human.
Similar to the Circle environment, the robot does not know which lane the human will select; accordingly, to safely pass the human the robot must anticipate and influence the driver's behavior.

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
