This directory provides the implemetnation details for the Robot environment as discussed in Section $6.2$ of the paper.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/da19c46d-0893-4d1d-ba8d-1e3a509ad744" style="width: 600px; height: auto; margin: 0 10px;">
</div>

## Environment
In the Robot environment the autonomous agent reaches for goals within a shared workspace. The robot is rewarded if it reaches for the same goal as the simulated human (i.e., if both agents move to the far left object). The robot’s action space is its $3\text{-DoF}$ end-effector velocity, the latent representation $z$ encodes the target the human wants to reach, and $\phi$ determines how the human updates their target object. For instance, at interaction $i+1$ the human might select the object to the left of its target at the previous interaction $i$.

### Instructions
Run the simulation using 

```
julia run_experiment.jl
```

We ran our simulations using `Julia 1.11.3`.

## Interactions
The variable `num_experiments` controls the number of experiments. We report results from $50$ episodes which is the default value for this variable.

### Analysis and Plotting
Results from the experiments are saved in two folders. `data` has the CSV data for each interaction, `figures` has rewards plots and video of agents for each interaction . To see the overall results for all experiments run

```
julia plot_data.jl
```
Overall results will be added to `figures` folder.
