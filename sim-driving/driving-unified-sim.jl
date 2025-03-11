"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Driving environment simulation
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

# define the arguments
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
            help="initial z (0-4)" 
            arg_type=Int
            default=0
    end
    return parse_args(arg)
end

# Parse the arguments
parsed_args = parse_commandline()

#   Constants
#   ≡≡≡≡≡≡≡≡≡

# Environment
road_length = Float32(11.)
road_width = Float32(3.)
car_width = Float32(1.)

# Robot
reset_pos = Float32.([0., 0.])
steer_bound = Float32.([-0.2, 0.2]) # Robot steering angle bounds

# Human lanes
lane_1 = Float32.([-1.5, 10.])
lane_2 = Float32.([-0.5, 10.])
lane_3 = Float32.([+0.5, 10.])
lane_4 = Float32.([+1.5, 10.])
human_lanes = [lane_1, lane_2, lane_3, lane_4]

# Initial states
init_state_H = Float32.(0.) # human x

# Probablity of human changing type
change_human = 0.99

total_timesteps = 1000
time_horizon = 10; # length of interaction
totalR = 0.
d = 1.0 # discount factor

mutable struct GlobalVars
    time_step::Int32
    aH::Int32
end

global_vars = GlobalVars(0, 0)

# Define the state type
mutable struct pomdpState
    sH::Float32 # human position
    sR::Vector{Float32} # robot position
    z::Int # human type
    sH_at_time_step_7::Float32
    human_lane::Int
    prev_lane::Vector{Float32}
end;

# Define the observation type
struct pomdpObservation
    sH::Float32 # human position
    sR_1::Float32 # robot x position
    sR_2::Float32 # robot y position which is 1.0
end;

#   Define the state, action, and observation spaces
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Define continuous state space with each element of pomdpState type
struct continuousStateSpace end; 
    Base.eltype(sspace::Type{continuousStateSpace}) = pomdpState

struct continuousActionSpace <: AbstractRNG end;
function POMDPs.rand(rng::AbstractRNG, aspace::continuousActionSpace)
    return Float32(
        rand(Distributions.Uniform(steer_bound[1], steer_bound[2])))
end;

# Define continuous observation space with each element of pomdpObservation type
struct continuousObservationSpace end;
Base.eltype(ospace::Type{continuousObservationSpace}) = pomdpObservation
    
#   Define the POMDP World
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Defining the pomdp model
mutable struct pomdpWorld <: POMDP{pomdpState, Float32, pomdpObservation}
    discount_factor::Float64 
end;

# Constructor for convenience
function pomdpWorld(;discount_factor::Float64=0.95)
    return pomdpWorld(discount_factor)
end;

function POMDPs.states(pomdp::pomdpWorld)
    return continuousStateSpace()
end;

function POMDPs.actions(pomdp::pomdpWorld)
    return continuousActionSpace()
end;

function POMDPs.observations(pomdp::pomdpWorld) 
    return continuousObservationSpace()
end;

#   Human model
#   ≡≡≡≡≡≡≡≡≡≡≡

function human_policy(state::pomdpState, z, time_step)
    sH, sR, = state.sH, state.sR
    sH_ = copy(sH)
    sH_1 = copy(state.sH_at_time_step_7)
    human_lane_1 = copy(state.human_lane)
    prev_lane = state.prev_lane
    if time_step == 7 && ( z == 0 || z == 1 )
        if z == 0
            if sR[1] > 0.
                sH_1 = Float32(0.5)
            else
                sH_1 = Float32(sR[1])
            end
        elseif z == 1
            if sR[1] < 0.
                sH_1 = Float32(-0.5)
            else
                sH_1 = Float32(sR[1])
            end
        end
    end

    if terminal_state(time_step)
        z_new = z
        if z_new == 2
            human_lane_1 = mod(human_lane_1 - 1, length(human_lanes)) + 1
            sH_ = human_lanes[human_lane_1][1]
        elseif z_new == 3
            human_lane_1 = mod(human_lane_1 + 1, length(human_lanes)) + 1
            sH_ = human_lanes[human_lane_1][1]
        elseif z_new == 4
            sH_ = prev_lane[1]
        else
            sH_ = sH_1
        end
    end

    return sH_1, sH_, human_lane_1
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
function POMDPs.transition(pomdp::pomdpWorld, state::pomdpState, actionR::Float32)
    # Update timestep
    global_vars.time_step += 1
    # Human action
    actionH = global_vars.aH

    sH, sR, z, human_lane, prev_lane = state.sH, state.sR, state.z, state.human_lane, state.prev_lane
    
    sH_1 = copy(state.sH_at_time_step_7)
    sH_1, _, _ = human_policy(state, z, global_vars.time_step)

    # Update the robot's position
    aR_ = Float32(actionR)
    sR_ = copy(sR)
    sR_ += [aR_, 1.0]
    s_ = pomdpState(sH, sR_, z, sH_1, human_lane, prev_lane)
   
    if terminal_state(global_vars.time_step)
        global_vars.time_step = 0
        
        if rand() > change_human
            z_new = rand(0:4)
        else
            z_new = state.z
        end
        
        sH_1, sH_, human_lane_1 = human_policy(state, z_new, global_vars.time_step)

        prev_lane = copy(sR_)
    
        return Deterministic(pomdpState(sH_, reset_pos, z_new, sH_1, human_lane_1, prev_lane))
    else
        return Deterministic(s_)
    end
end;

# Define the observation space
function POMDPs.observation(pomdp::pomdpWorld, state::pomdpState, action::Float32, statep::pomdpState)
    sH = copy(statep.sH)
    sR = copy(statep.sR)
    sR_1, sR_2 = sR[1], sR[2]
    return Deterministic(pomdpObservation(Float32(sH), Float32(sR_1), Float32(sR_2)))
end;

# Define the reward function
function POMDPs.reward(pomdp::pomdpWorld, state::pomdpState, action::Float32)
    time_step = global_vars.time_step
    sH, sR, z = state.sH, state.sR, state.z
    r = -abs(action) * 10

    if terminal_state(time_step)
        if abs(sR[1] - sH) < car_width
            r -= 100
        end
    end
    
    return r
end;
    
# MISC functions
POMDPs.discount(pomdp::pomdpWorld) = pomdp.discount_factor

function POMDPs.initialstate(pomdp::pomdpWorld)
    return Deterministic(pomdpState(init_state_H, reset_pos, parsed_args["init_z"], Float32(0.), 1, lane_1))
end;

# Definition is done, lets solve it!
pomdp = pomdpWorld();
solver = POMCPOWSolver();
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

resampler = POMDPResampler(5000)
belief_updater = BasicParticleFilter(pomdp, pomdp, resampler, 5000, MersenneTwister(42))

# Get a state and initialize belief
init = initialstate(pomdp)
state = rand(init)
# Initial Belief, equal chance of starting with either human type
b = WeightedParticleBelief([pomdpState(state.sH, state.sR, state.z, state.sH_at_time_step_7, state.human_lane, state.prev_lane) for i in 1:5000], [1/5000 for i in 1:5000])

state_new = init.val

human_pos = []
human_a = []
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
    global human_a, human_pos, robot_pos, human_z 
    global reward, belief

    push!(human_pos, state_new.sH)
    push!(human_a, global_vars.aH)
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

# iIn the same folder as the code, create subfolders
directoryPath = dirname(@__FILE__)
mkpath(directoryPath * "/figures")
mkpath(directoryPath * "/data")


#   Start visualization
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# sH is the x position of the human
human_pos = [[x[1], 10.] for x in human_pos]

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

    plt = plot(aspect_ratio=:equal, legend=false, title="Interaction: $num, Human Type: $(human_z[start_idx])")

    # Plot the road
    plot!([0, 0], [0, road_length], linewidth=road_width*30, color=:black, fill=(0.8, :lightgrey))
    # Plot the lane markings
    plot!([-1.5, -1.5], [0, road_length], linewidth=1.5, color=:white, linestyle=:dash)
    plot!([-0.5, -0.5], [0, road_length], linewidth=1.5, color=:white, linestyle=:dash)
    plot!([0.5, 0.5], [0, road_length], linewidth=1.5, color=:white, linestyle=:dash)
    plot!([1.5, 1.5], [0, road_length], linewidth=1.5, color=:white, linestyle=:dash)

    plot!(plt, [x[1] for x in robot_pos[start_idx:end_idx-1]], [x[2] for x in robot_pos[start_idx:end_idx-1]], xlimits=(-2, 2), ylimits=(0., 11.), linewidth=1.5, marker=:rect, markersize=8, color=:orange)
    plot!(plt, [x[1] for x in human_pos[start_idx:end_idx-2]], [x[2] for x in human_pos[start_idx:end_idx-2]], xlimits=(-2, 2), ylimits=(0., 11.), linewidth=1.5, marker=:rect, markersize=8, color=:green)
    annotate!(plt, -10.0, 1.2, text("Interaction Reward: $(round(sum(reward[start_idx:end_idx-1]), digits=2))", 10, :left, "Computer Modern"))

    total_reward2 += sum(reward[start_idx:end_idx-1])
    annotate!(plt, -10.0, 1.8, text("Total Reward: $(round(total_reward2, digits=2))", 10, :left, "Computer Modern"))

    start_idx = end_idx
    
    Plots.frame(anim)
    
    next!(p)
end
mp4(anim, directoryPath * "/figures/driving_unified_" * parsed_args["name"] * "_htype" * "_$(parsed_args["init_z"]).mp4", fps=5)

# Plot robot reward per interaction
plt_rewards = plot(1:n_interactions, interaction_rewards, xlabel="Interaction", ylabel="Reward", title="Robot Reward per Interaction", legend=false)
savefig(plt_rewards, directoryPath * "/figures/driving_interaction_rewards_" * parsed_args["name"] * ".svg")

########### Save the data ############
# println("Total Reward: ", totalR)
# println("Reward: ", reward)
cum_reward = cumsum(reward)
println("Total Reward: ", cum_reward[end])

# Create a DataFrame 
# Flatten the list elements into separate columns
df = DataFrame(
    time_step = timestep_list,
    robot_pos_x=[pos[1] for pos in robot_pos],
    robot_pos_y=[pos[2] for pos in robot_pos],
    human_pos_x=[pos[1] for pos in human_pos],
    human_pos_y=[pos[2] for pos in human_pos],
    robot_a = robot_a,
    human_z = human_z,
    reward = reward,
    cum_reward = cum_reward
)
# Save the data w/ no headers
CSV.write(directoryPath * "/data/driving_unified_" * parsed_args["name"] * ".csv", df, writeheader=false)