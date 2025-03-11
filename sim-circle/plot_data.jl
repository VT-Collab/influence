using CSV, DataFrames, Plots

directoryPath = dirname(@__FILE__)

# Initialize variables
total_timesteps = 1000
all_rewards = zeros(total_timesteps)
num_experiments = 2

# Read rewards from each CSV file and accumulate them
for i in 1:num_experiments
    df = CSV.read("$directoryPath/data/circle_unified_$i.csv", DataFrame, header=false)
    rewards = df[:, 8]  # reward column is the 8th column
    all_rewards[1:length(rewards)] += rewards
end

# Calculate average rewards
average_rewards = all_rewards / num_experiments

# Plot average rewards per timestep
plt = plot(1:total_timesteps, average_rewards, xlabel="Timestep", ylabel="Average Reward", title="Average Robot Reward per Timestep", legend=false)
savefig(plt, "$directoryPath/figures/average_reward_per_timestep.svg")