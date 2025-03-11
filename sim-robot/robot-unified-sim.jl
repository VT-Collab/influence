"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Robot environment simulation
"""
# Comment out the Pkg stuff after the first run
# using Pkg
# Pkg.add(["POMDPs", "POMDPTools", "DiscreteValueIteration", "CSV", "DataFrames", 
#                         "ArgParse", "Plots", "POMCPOW", "BasicPOMCP", "Distributions", 
#                         "ParticleFilters", "ProgressMeter", "LinearAlgebra", "StatsBase", 
#                         "PyCall"])
# Pkg.update()

# Importing the packages
using POMDPs, POMDPTools
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Plots.PlotMeasures, ProgressMeter
using ArgParse, CSV, DataFrames 
using POMCPOW, BasicPOMCP
using Distributions, ParticleFilters
using Random, LinearAlgebra, StatsBase

ProgressMeter.ijulia_behavior(:clear)

# Define the arguments
function parse_commandline()
    arg = ArgParseSettings()

    @add_arg_table arg begin
        "--name" 
            help="output file name's id" 
            arg_type=String
            default="1"
        "--discount_factor" 
            help="discount factor" 
            arg_type=Float64
            default=Float64(0.95)
        "--init_z" 
            help="initial z (0-3)"
            arg_type=Int
            default=0
    end
    return parse_args(arg)
end

# Parse the arguments
parsed_args = parse_commandline()

#   Constants
#   ≡≡≡≡≡≡≡≡≡

# Robot
reset_pos = [4.27292109e-01, -4.92869634e-12,  5.27017295e-01]
action_space_Rx = Float32.([-0.1, 0.1])
action_space_Ry = Float32.([-0.1, 0.1])
action_space_Rz = Float32.([-0.1, 0.1])

# Goal index
goal_idx = [1, 2, 3]

# Goal target positions
pos_goal1 = Float32.([0.6, +0.3, 0.0])
pos_goal2 = Float32.([0.8, 0.0, 0.0])
pos_goal3 = Float32.([0.6, -0.3, 0.0])
pos_goals = [pos_goal1, pos_goal2, pos_goal3]

init_choice = 1

# Probablity of human changing type
change_human = 0.99
reset_choice = 0.999

total_timesteps = 1000
time_horizon = 10; # length of interactions
totalR = 0.
d = 1.0

mutable struct GlobalVars
    time_step::Int32
end;

global_vars = GlobalVars(0)

# Define the state type
mutable struct pomdpState
    choice::Int32 # Human's chosen goal (1, 2, 3)
    sR::Vector{Float32} # robot position
    z::Int32 # human type
end;

# Define the observation type
struct pomdpObservation
    choice::Int # Human's chosen goal (1, 2, 3)
    sR_1::Float32 # robot x position
    sR_2::Float32 # robot y position
    sR_3::Float32 # robot z position
end;

#   Define the state, action, and observation spaces
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Define continuous state space with each element of pomdpState type
struct continuousStateSpace end; 
    Base.eltype(sspace::Type{continuousStateSpace}) = pomdpState

struct continuousActionSpace <: AbstractRNG end;
function POMDPs.rand(rng::AbstractRNG, aspace::continuousActionSpace)
    return Float32.([rand(Distributions.Uniform(action_space_Rx[1], action_space_Rx[2])), 
    rand(Distributions.Uniform(action_space_Ry[1], action_space_Ry[2])), rand(Distributions.Uniform(action_space_Rz[1], action_space_Rz[2]))])
end

# Define continuous observation space with each element of pomdpObservation type
struct continuousObservationSpace end;
Base.eltype(ospace::Type{continuousObservationSpace}) = pomdpObservation
    
#   Define the POMDP World
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Defining the pomdp model
mutable struct pomdpWorld <: POMDP{pomdpState, Vector{Float32}, pomdpObservation}
    discount_factor::Float64 
end;

# Constructor for convenience
function pomdpWorld(;discount_factor::Float64=0.95)
    return pomdpWorld(discount_factor)
end;

function POMDPs.states(pomdp::pomdpWorld)
    return continuousStateSpace()
end

function POMDPs.actions(pomdp::pomdpWorld)
    return continuousActionSpace()
end

function POMDPs.observations(pomdp::pomdpWorld) 
    return continuousObservationSpace()
end

#   Human model
#   ≡≡≡≡≡≡≡≡≡≡≡

function human_policy(state::pomdpState, z_new::Int32)
    sR, choice = state.sR, state.choice
    sR_ = copy(sR)
    choice_ = copy(choice)

    if z_new == 0
        if sR_[2] < pos_goals[choice_][2]
            choice_ = mod(choice_ - 1 + 1, length(pos_goals)) + 1
        else
            choice_ = mod(choice_ - 1 - 1, length(pos_goals)) + 1
        end
    elseif z_new == 1
        if sR_[2] < pos_goals[choice_][2]
            choice_ = mod(choice_ - 1 + 0, length(pos_goals)) + 1
        else
            choice_ = mod(choice_ - 1 + 1, length(pos_goals)) + 1
        end
    elseif z_new == 2 
        choice_ = mod(choice_ - 1 + 1, length(pos_goals)) + 1
    elseif z_new == 3
        choice_ = mod(choice_ - 1 - 1, length(pos_goals)) + 1
    end

    return choice_
end;


#   Define the transition, observation, and reward functions
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Check if it's terminal state
function terminal_state(t)
    if t % time_horizon == 0
        return true
    else
        return false
    end
end;

# Define the transition function
function POMDPs.transition(pomdp::pomdpWorld, state::pomdpState, actionR::Vector{Float32})
    # update timestep
    global_vars.time_step += 1

    # Robot action
    sR, z = state.sR, state.z
    aRx, aRy, aRz = actionR[1], actionR[2], actionR[3]
    sR_ = copy(sR)
    sR_ += [aRx, aRy, aRz]

    # Human action
    choice = state.choice

    s_ = pomdpState(choice, sR_, z)
    if terminal_state(global_vars.time_step)
        global_vars.time_step = 0

        if rand() > reset_choice
            choice = rand(1:3)
        end

        # Update the human type based on its dynamics
        if rand() > change_human
            z_new = Int32(rand(0:3))
        else
            z_new = Int32(z)
        end

        choice = human_policy(state, z_new)

        return Deterministic(pomdpState(choice, reset_pos, z_new))
    else
        return Deterministic(s_)
    end
end;

# Define the observation space
function POMDPs.observation(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32}, statep::pomdpState)
    choice = copy(statep.choice)
    sR = copy(statep.sR)
    sR_1, sR_2, sR_3 = sR[1], sR[2], sR[3]
    return Deterministic(pomdpObservation(Int(choice), Float32(sR_1), Float32(sR_2), Float32(sR_3)))
end;

# Define the reward function
function POMDPs.reward(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32})
    choice, sR, z = state.choice, state.sR, state.z

    r = - norm(pos_goals[choice] - sR) * 100

    if terminal_state(global_vars.time_step)
        if choice == 1 && z < 3
            r += 100
        elseif choice == 2 && z == 3
            r += 100
        end
    end

    return r
end;
    
# MISC functions
POMDPs.discount(pomdp::pomdpWorld) = pomdp.discount_factor


function POMDPs.initialstate(pomdp::pomdpWorld)
    return Deterministic(pomdpState(init_choice, reset_pos, parsed_args["init_z"]))
end;


# dDefinition is done, lets solve it!
pomdp = pomdpWorld();
solver = POMCPOWSolver()
policy = solve(solver, pomdp);

# Define the custom resampler
struct POMDPResampler
    n::Int
end

function ParticleFilters.resample(r::POMDPResampler,
                                  bp::WeightedParticleBelief{pomdpState},
                                  pm::POMDP,
                                  rm::POMDP,
                                  b,
                                  a,
                                  o,
                                  rng::AbstractRNG)
    
    if weight_sum(bp) == 0.0
        # No appropriate particles - resample from the initial distribution
        new_ps = [rand(rng, initialstate(pm)) for i in 1:r.n]
        return ParticleCollection(new_ps)
    else
        # Normal resample
        return resample(LowVarianceResampler(r.n), bp, rng)
    end
end

belief_updater = BootstrapFilter(pomdp, 5000);
resampler = POMDPResampler(5000)
belief_updater = BasicParticleFilter(pomdp, pomdp, resampler, 5000, MersenneTwister(42))

# Get a state and initialize belief
init = initialstate(pomdp)
state = rand(init)
# Initial Belief, 50/50 chance of starting with z = 0 - 3
b = WeightedParticleBelief([pomdpState(state.choice, state.sR, state.z) for i in 1:5000], [1/5000 for i in 1:5000])

state_new = init.val

human_choice = []
robot_pos = []
robot_a = []
human_z = []
reward = []
belief = []
timestep_list = []

# Progress bar
p = Progress(total_timesteps, showspeed=true)
generate_showvalues(iter, total) = () -> [(:timestep, iter), (:total, total)]

# Main loop
for i in 1:total_timesteps
    global state_new, totalR, d, b
    global human_choice, robot_pos, human_z 
    global reward, belief

    push!(human_choice, state_new.choice)
    push!(robot_pos, state_new.sR)
    push!(human_z, state_new.z)
    push!(timestep_list, i)
    
    # Get robot action from the policy
    aR = action(policy, b)

    push!(robot_a, aR)
    
    # Robot takes an action
    state_new, o, r = @gen(:sp, :o, :r)(pomdp, state_new, aR)
  
    push!(reward, r)
    
    totalR += r
    d *= discount(pomdp)

    # Update belief
    pm = particle_memory(pomdp)
    resize!(pm, length(b.particles))  # To make sure the particle memory is correctly sized
    ParticleFilters.predict!(pm, pomdp, b, aR, MersenneTwister(42))
    wm = Vector{Float64}(undef, length(pm))
    ParticleFilters.reweight!(wm, pomdp, b, aR, pm, o)

    # Normalize weights to avoid collapse
    total_weight = sum(wm)
    if total_weight > 0
        wm ./= total_weight
    else
        wm .= 1.0 / length(wm)
    end

    b = ParticleFilters.resample(resampler, WeightedParticleBelief(pm, wm), pomdp, pomdp, b, aR, o, MersenneTwister(42))

    next!(p, step=1, showvalues=generate_showvalues(i, total_timesteps))
end

# In the same folder as the code, create subfolders
directoryPath = dirname(@__FILE__)
mkpath(directoryPath * "/figures")
mkpath(directoryPath * "/data")

#   Start visualization
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Plotting an entire Interaction
# Identify the start of each interaction
filtered_robot_reached = [i for i in 1:time_horizon:total_timesteps+1]

n_interactions = length(filtered_robot_reached) - 1
last_ints_to_plot = n_interactions
sidx = max(1, n_interactions - last_ints_to_plot)
start_idx = filtered_robot_reached[sidx]

println("Found $n_interactions interactions")

interaction_rewards = []

anim = Plots.Animation()
p = Progress(length(filtered_robot_reached[sidx + 1:n_interactions + 1]), showspeed=true)
total_reward2 = 0
for (num, end_idx) in enumerate(filtered_robot_reached[sidx + 1:n_interactions + 1])
    global start_idx
    global total_reward2

    interaction_reward = sum(reward[start_idx:end_idx-1])
    push!(interaction_rewards, interaction_reward)

    plt = plot(aspect_ratio=:equal, legend=false, title="Interaction: $num, Human Type: $(human_z[start_idx])", bottom_margin=10mm)

    # Plot the goals
    for i in 1:3
        plot!([pos_goals[i][1]], [pos_goals[i][2]], seriestype=:scatter, markersize=7, label="Goal $i", color=:black)
    end

    # Plot the robot's path
    plot!(plt, [x[1] for x in robot_pos[start_idx:end_idx-1]], [x[2] for x in robot_pos[start_idx:end_idx-1]], xlimits=(0.2, 1.), ylimits=(-0.5, 0.5), linewidth=1.5, marker=:rect, markersize=5, color=:orange)
    # plot human choice ( if choice is 1, plot a circle around the goal 1 and so on)
    for i in start_idx:end_idx-2
        plot!([pos_goals[human_choice[i]][1]], [pos_goals[human_choice[i]][2]], seriestype=:scatter, markersize=7, label="Human Choice", color=:red)
    end

    annotate!(plt, 0.3, -0.6, text("Cumulative Reward: $(round(sum(reward[start_idx:end_idx-1]), digits=2))", 10, :left, "Computer Modern"))

    total_reward2 += sum(reward[start_idx:end_idx-1])
    annotate!(plt, 0.3, -0.65, text("Total Reward: $(round(total_reward2, digits=2))", 10, :left, "Computer Modern"))

    start_idx = end_idx
    
    Plots.frame(anim)
    
    next!(p)
end
mp4(anim, directoryPath * "/figures/robot_unified_" * parsed_args["name"] * "_htype" * "_$(parsed_args["init_z"]).mp4", fps=5)

# Plot robot reward per interaction
plt_rewards = plot(1:n_interactions, interaction_rewards, xlabel="Interaction", ylabel="Reward", title="Robot Reward per Interaction", legend=false)
savefig(plt_rewards, directoryPath * "/figures/robot_interaction_rewards_" * parsed_args["name"] * ".svg")

########### Save the data ############
println("Total Reward: ", totalR)
# println("Reward: ", reward)
cum_reward = cumsum(reward)
println("Cumulative Reward: ", cum_reward[end])

# Create a DataFrame
# Flatten the list elements into separate columns
df = DataFrame(
    time_step = timestep_list,
    robot_pos_x = [x[1] for x in robot_pos],
    robot_pos_y = [x[2] for x in robot_pos],
    robot_pos_z = [x[3] for x in robot_pos],
    human_choice = human_choice,
    robot_a_x = [x[1] for x in robot_a],
    robot_a_y = [x[2] for x in robot_a],
    robot_a_z = [x[3] for x in robot_a],
    human_z = human_z,
    reward = reward,
    cum_reward = cum_reward
)

# Save the data w/ no headers
CSV.write(directoryPath * "/data/robot_unified_" * parsed_args["name"] * ".csv", df, writeheader=false)