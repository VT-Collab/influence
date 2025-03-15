"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Highway environment simulation

This code saves the data in Excel for SPSS paired t-test.

Data Structure:
state_data = [time_step, robot_car_heading, robot_car_x, robot_car_y, robot_car_vx, robot_car_vy,
robot_car_angular_velocity, robot_car_acceleration, 
human_car_heading, human_car_x, human_car_y, human_car_vx, human_car_vy, 
human_car_angular_velocity, human_car_acceleration, 
collisions, interaction_score, total_score, robot_reward]
action_data = [robot_car_heading_input, robot_car_acceleration_input, human_car_heading_input, human_car_acceleration_input]
combined_data = vcat(state_data, action_data)

Note that the first element of the state is the timestamp which starts from 0 in the dataset.
"""

# using Pkg
# Pkg.add(["POMDPs", "POMDPTools", "DiscreteValueIteration", "CSV", "DataFrames", 
#                         "ArgParse", "Plots", "POMCPOW", "BasicPOMCP", "Distributions", 
#                         "ParticleFilters", "ProgressMeter", "LinearAlgebra", "StatsBase", 
#                         "Gtk4", "Joysticks", "Observables", "JSON", "XLSX"])
# Pkg.update()

using JSON
using Dates
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Statistics
using XLSX
using ProgressMeter

# User IDs and scenes
user_ids = [0] # for sim we have id=0

scenes = ["highway"]

inte_to_read = 100

# Directory path for this code
directoryPath = dirname(@__FILE__)

# Directory path for plots
plots_directory = joinpath(directoryPath, "plots")
mkpath(plots_directory)

# Folders for each scene
scene_folders = Dict(
    "highway" => ["unified-hw-res"]
)

# Time horizon for each scene
time_horizons = Dict(
    "highway" => 120
)

# Function to save data to an Excel file
function save_data_to_excel(file_path, data_dict, user_ids, num_interactions)
    # Define the desired order of data types
    data_order = ["Human Car Y", "Robot Reward", "Collisions", "Robot Car Y", "Interaction Score"]
    
    XLSX.openxlsx(file_path, mode="w") do xf
        sheet = xf[1]
        row = 1
        col = 3
        # Add headers for interactions
        sheet[row, 1] = "User ID"
        sheet[row, 2] = "Data Type"
        for j in 1:num_interactions
            sheet[row, col] = "Interaction $j"
            col += 1
        end
        row += 1
        # Add data for each user and metric in the specified order
        for key in data_order
            data = data_dict[key]
            for (i, user_id) in enumerate(user_ids)
                sheet[row, 1] = "user id $user_id"
                sheet[row, 2] = key
                for j in 1:num_interactions
                    sheet[row, j + 2] = data[(i - 1) * num_interactions + j]
                end
                row += 1
            end
        end
    end
end

function robot_reward_highway(robot_x, robot_y, robot_heading, human_x, human_y)
    dist_to_human = max(0., 2.5 - sqrt((robot_x - human_x)^2 + (robot_y - human_y)^2))
    block_human = (robot_x - human_x) ^ 2
    heading = (pi/2. - robot_heading) ^ 2

    robot_reward = dist_to_human * 10. + block_human + heading * 2.5
    return -robot_reward
end

# Get all .json files for specific user_id and scene
function get_json_files(user_ids, scene, folder, directoryPath, time_horizon)
    json_files = Dict{Int, Vector{String}}()
    for user_id in user_ids
        res_path = joinpath(directoryPath, folder)
        json_files[user_id] = []
        for pattern in [Regex(string(scene, "_unified_u", user_id, ".*\\.json\$"))]
            println("Looking for files with pattern: $pattern in folder: $res_path")
            files = filter(file -> occursin(pattern, file), readdir(res_path))
            sorted_files = sort(files, by = x -> parse(Int, match(r"u(\d+)", x).captures[1]))
            for file in sorted_files
                println("Checking file: $file")
                if occursin(pattern, file)
                    println("Matched file: $file")
                    push!(json_files[user_id], joinpath(res_path, file))
                end
            end
        end
    end
    return json_files
end

# Read and collect the required data points from a .json file
function read_and_collect_data(file_path, time_horizon, current_timestep, robot_x_data, robot_car_y_data, robot_heading_data, robot_vx_data, 
                                human_x_data, human_car_y_data, human_heading_data, 
                                collision_data, interaction_score_data, computed_reward_data, interactions_read, max_interactions, user_id, scene)
    dataset = open(file_path, "r") do io
        JSON.parse(IOBuffer(read(io, String)))
    end

    println("Reading file: $file_path")

    # Initialize cumulative reward
    cumulative_computed_reward = 0.0

    # Collect the required data points
    for i in eachindex(dataset)
        if (current_timestep - 1) % time_horizon == 0
            interactions_read += 1
            if interactions_read > max_interactions
                return current_timestep, interactions_read
            end
            # Reset cumulative reward for new interaction
            cumulative_computed_reward = 0.0
        end
        # Skip the reset datapoint (last timestep of each interaction)
        if current_timestep % time_horizon == 0
            current_timestep += 1
            continue
        end

        # Collect data for each valid timestep before the reset datapoint
        robot_x = dataset[i][3]  # robot_x is the 3rd element in state_data
        robot_car_y = dataset[i][4] 
        robot_heading = dataset[i][2]
        robot_vx = dataset[i][5]
        human_x = dataset[i][10]
        human_car_y = dataset[i][11]
        human_heading = dataset[i][9]
        collisions = dataset[i][16]
        interaction_score = dataset[i][17]

        # Compute the reward using the appropriate reward function
        computed_reward = robot_reward_highway(robot_x, robot_car_y, robot_heading, human_x, human_car_y)

        # Accumulate the computed reward
        cumulative_computed_reward += computed_reward

        # Collect data for the last valid timestep before the reset datapoint
        if current_timestep % time_horizon == time_horizon - 1
            push!(robot_x_data, robot_x)
            push!(robot_car_y_data, robot_car_y)
            push!(robot_heading_data, robot_heading)
            push!(robot_vx_data, robot_vx)
            push!(human_x_data, human_x)
            push!(human_car_y_data, human_car_y)
            push!(human_heading_data, human_heading)
            push!(collision_data, collisions)
            push!(interaction_score_data, interaction_score)
            push!(computed_reward_data, cumulative_computed_reward)
        end

        current_timestep += 1
    end

    return current_timestep, interactions_read
end


# Calculate the moving average
function moving_average(data, window_size)
    return [mean(data[max(1, i-window_size+1):i]) for i in 1:length(data)]
end

# Function to read and collect X-Y position data from a .json file
function read_xy_data(file_path, time_horizon, human_x_data, human_y_data, robot_x_data, robot_y_data)
    dataset = open(file_path, "r") do io
        JSON.parse(IOBuffer(read(io, String)))
    end

    println("Reading file for X-Y data: $file_path")

    # Collect the required data points
    for i in eachindex(dataset)
        # Skip the last timestep of each interaction
        if i % time_horizon == 0
            continue
        end

        # Collect data for each valid timestep
        robot_x = dataset[i][3]
        robot_y = dataset[i][4]
        human_x = dataset[i][10]
        human_y = dataset[i][11]

        push!(robot_x_data, robot_x)
        push!(robot_y_data, robot_y)
        push!(human_x_data, human_x)
        push!(human_y_data, human_y)
    end
end

# Function to plot X-Y locations for human and robot cars
function plot_xy_locations(human_x_data, human_y_data, robot_x_data, robot_y_data, interaction_num, folder, trajs_directory)
    plot(human_x_data, human_y_data, label="Human Car", color="blue", xlabel="X Position", ylabel="Y Position", title="Interaction $interaction_num - $folder", legend=:topright, aspect_ratio=:equal)
    plot!(robot_x_data, robot_y_data, label="Robot Car", color="red")
    savefig(joinpath(trajs_directory, "interaction_$(interaction_num)_$(folder)_xy_plot.svg"))
end

# Initialize a dictionary to store the overall average robot rewards for each folder
overall_robot_rewards = Dict("unified-hw-res" => 0.0)

# Directory path for this code
directoryPath = dirname(@__FILE__)

# Directory path for trajectory plots
plots_directory = joinpath(directoryPath, "plots")
trajs_directory = joinpath(plots_directory, "trajs")
mkpath(trajs_directory)

# Iterate over each scene and folder to generate plots
for scene in scenes
    for folder in scene_folders[scene]
        # Get the time horizon for the current scene
        time_horizon = time_horizons[scene]

        # Get the list of .json files for the current scene and folder
        local json_files = get_json_files(user_ids, scene, folder, directoryPath, time_horizon)

        # Check if any files were found
        if isempty(json_files)
            println("No .json files found for the specified user IDs and scenes.")
            continue
        else
            println("Found .json files: ", json_files)
        end

        # Initialize the current timestep and data arrays
        current_timestep = 1
        robot_x_data = []
        robot_car_y_data = []
        robot_heading_data = []
        robot_vx_data = []
        human_x_data = []
        human_car_y_data = []
        human_heading_data = []
        collision_data = []
        interaction_score_data = []
        computed_reward_data = []

        # Read and collect data from each .json file
        for user_id in user_ids
            interactions_read = 0
            max_interactions = inte_to_read
            files = json_files[user_id]
            # Read 100 interactions from the single file
            current_timestep, interactions_read = read_and_collect_data(files[1], time_horizon, current_timestep, robot_x_data, robot_car_y_data, robot_heading_data, robot_vx_data, human_x_data, human_car_y_data, human_heading_data, collision_data, interaction_score_data, computed_reward_data, interactions_read, max_interactions, user_id, scene)
        end

        # Debugging: Print lengths of data arrays
        println("Length of robot_x_data: ", length(robot_x_data))
        println("Length of robot_car_y_data: ", length(robot_car_y_data))
        println("Length of human_x_data: ", length(human_x_data))
        println("Length of human_car_y_data: ", length(human_car_y_data))
        println("Length of collision_data: ", length(collision_data))
        println("Length of interaction_score_data: ", length(interaction_score_data))
        println("Length of computed_robot_reward_data: ", length(computed_reward_data))

        # Organize data by interaction
        num_interactions = inte_to_read
        total_interactions = num_interactions * length(user_ids)

        # Check if we have enough data before reshaping
        if length(robot_car_y_data) >= total_interactions && length(human_car_y_data) >= total_interactions && length(collision_data) >= total_interactions && length(interaction_score_data) >= total_interactions && length(computed_reward_data) >= total_interactions
            robot_car_y_data_by_interaction = reshape(robot_car_y_data[1:total_interactions], num_interactions, length(user_ids))
            human_car_y_data_by_interaction = reshape(human_car_y_data[1:total_interactions], num_interactions, length(user_ids))
            collision_data_by_interaction = reshape(collision_data[1:total_interactions], num_interactions, length(user_ids))
            interaction_score_data_by_interaction = reshape(interaction_score_data[1:total_interactions], num_interactions, length(user_ids))
            computed_reward_data_by_interaction = reshape(computed_reward_data[1:total_interactions], num_interactions, length(user_ids))
        else
            println("Not enough data collected for reshaping.")
            continue
        end

        # Save data to Excel file
        data_dict = Dict(
            "Human Car Y" => human_car_y_data,
            "Robot Reward" => computed_reward_data,
            "Collisions" => collision_data,
            "Robot Car Y" => robot_car_y_data,
            "Interaction Score" => interaction_score_data
        )
        excel_file_path = joinpath(plots_directory, "$(scene)_$(folder)_data.xlsx")
        save_data_to_excel(excel_file_path, data_dict, user_ids, num_interactions)

        # Calculate the mean and standard deviation for each interaction
        robot_car_y_mean = mean(robot_car_y_data_by_interaction, dims=2)
        robot_car_y_std = std(robot_car_y_data_by_interaction, dims=2)
        human_car_y_mean = mean(human_car_y_data_by_interaction, dims=2)
        human_car_y_std = std(human_car_y_data_by_interaction, dims=2)
        collision_mean = mean(collision_data_by_interaction, dims=2)
        collision_std = std(collision_data_by_interaction, dims=2)
        interaction_score_mean = mean(interaction_score_data_by_interaction, dims=2)
        interaction_score_std = std(interaction_score_data_by_interaction, dims=2)
        comp_robot_reward_mean = mean(computed_reward_data_by_interaction, dims=2)
        comp_robot_reward_std = std(computed_reward_data_by_interaction, dims=2)

        # Calculate the moving average for the mean values
        window_size = 1  # Define the window size for the moving average
        robot_car_y_mean_ma = moving_average(robot_car_y_mean, window_size)
        human_car_y_mean_ma = moving_average(human_car_y_mean, window_size)
        collision_mean_ma = moving_average(collision_mean, window_size)
        interaction_score_mean_ma = moving_average(interaction_score_mean, window_size)
        comp_robot_reward_mean_ma = moving_average(comp_robot_reward_mean, window_size)

        # Calculate the overall mean for horizontal mean lines
        overall_robot_car_y_mean = mean(robot_car_y_data)
        overall_human_car_y_mean = mean(human_car_y_data)
        overall_collision_mean = mean(collision_data)
        overall_interaction_score_mean = mean(interaction_score_data)
        overall_comp_robot_reward_mean = mean(computed_reward_data)

        # Store the overall average robot reward in the dictionary
        overall_robot_rewards[folder] = overall_comp_robot_reward_mean

        # Define colors based on the folder
        robot_color = "#FF9900"
        human_color = "#2A8FBD"
        collision_color = "#B3B3B3"
        interaction_score_color = "#FF5733"
        comp_reward_color = "#FF5733"

        # Plot the mean values with fill-in-between using the standard deviation
        plot(1:num_interactions, human_car_y_mean_ma, label="Human Car Y MA", color=human_color, legend=:outerright, grid=false)
        xlabel!("Interaction")
        ylabel!("Lane Progress")
        title!("$scene - $folder")

        # Save the plot
        savefig(joinpath(plots_directory, "$(scene)_$(folder)_robot_vs_human_car_y.svg"))

        # Add a small constant to avoid zero values in the collision data
        collision_mean_adjusted = collision_mean .+ 1e-3
        collision_std_adjusted = collision_std .+ 1e-3
        overall_collision_mean_adjusted = overall_collision_mean + 1e-3

        # Plot the number of collisions for each interaction with STD using a logarithmic scale
        plot(1:num_interactions, collision_mean_ma .+ 1e-3, label="Collisions MA", color=collision_color, yscale=:log10, ylim=(1e-4, 1e4), legend=:outerright, grid=false)
        xlabel!("Interaction")
        ylabel!("Number of Collisions (log scale)")
        title!("$scene - $folder - # Collisions")

        # Save the plot
        savefig(joinpath(plots_directory, "$(scene)_$(folder)_collisions_per_interaction.svg"))

        # Initialize arrays for X-Y position data
        robot_x_data_xy = []
        robot_y_data_xy = []
        human_x_data_xy = []
        human_y_data_xy = []

        # Read and collect X-Y position data from each .json file
        for user_id in user_ids
            files = json_files[user_id]
            read_xy_data(files[1], time_horizon, human_x_data_xy, human_y_data_xy, robot_x_data_xy, robot_y_data_xy)
        end

        # Debugging: Print lengths of X-Y data arrays
        println("Length of robot_x_data_xy: ", length(robot_x_data_xy))
        println("Length of robot_y_data_xy: ", length(robot_y_data_xy))
        println("Length of human_x_data_xy: ", length(human_x_data_xy))
        println("Length of human_y_data_xy: ", length(human_y_data_xy))

        # Generate X-Y location plots for each interaction with a progress bar
        @showprogress for interaction_num in 1:num_interactions
            start_idx = (interaction_num - 1) * (time_horizon - 1) + 1
            end_idx = interaction_num * (time_horizon - 1)
            human_x_interaction = human_x_data_xy[start_idx:end_idx]
            human_y_interaction = human_y_data_xy[start_idx:end_idx]
            robot_x_interaction = robot_x_data_xy[start_idx:end_idx]
            robot_y_interaction = robot_y_data_xy[start_idx:end_idx]
            plot_xy_locations(human_x_interaction, human_y_interaction, robot_x_interaction, robot_y_interaction, interaction_num, folder, trajs_directory)
        end
    end
end

# Create a bar plot for the average robot reward for "unified-hw-res"
bar_plot_data_pomdp = [overall_robot_rewards["unified-hw-res"]]
bar_labels_pomdp = ["UNIFIED"]

bar(bar_labels_pomdp, bar_plot_data_pomdp, legend=false, title="Average Robot Reward - UNIFIED", ylabel="Average Reward", color=["#FF9900"], grid=false)

# Save the bar plot
savefig(joinpath(plots_directory, "average_robot_reward_unified_bar_plot.svg"))