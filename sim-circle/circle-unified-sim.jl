"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Circle environment simulation
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
        "--init_hum" 
            help="initial hum position (0-2*pi)"
            arg_type=Float32
            default=Float32(0.)
    end
    return parse_args(arg)
end

# Parse the arguments
parsed_args = parse_commandline()

#   Constants
#   ≡≡≡≡≡≡≡≡≡

# For continuous action space
action_space_Rx = Float32.([-0.2, 0.2])
action_space_Ry = Float32.([-0.2, 0.2])

init_theta = parsed_args["init_hum"] # for the humuman
reset_pos = Float32.([0., 0.5]) # for the robot
reset_theta = 0.999
change_partner = 0.99

total_timesteps = 1000
time_horizon = 10; # number of interactions
totalR = 0.
d = 1.0 # discount factor
radius = 1.0 # radius of the circle

mutable struct GlobalVars
    time_step::Int32
    aH::Float32
end
global_vars = GlobalVars(0, Float32(0.))

# Define the state type
mutable struct pomdpState
    sH::Float32 # human position
    sR::Vector{Float32} # robot position
    z::Int # hum going CW or CCW
end;

# Define the observation type
struct pomdpObservation
    sH::Float32
end;

#   Define the state, action, and observation spaces
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Define continuous state space with each element of pomdpState type
struct continuousStateSpace end; 
    Base.eltype(sspace::Type{continuousStateSpace}) = pomdpState

struct continuousActionSpace <: AbstractRNG end;
function POMDPs.rand(rng::AbstractRNG, aspace::continuousActionSpace)
    return Float32.([rand(Distributions.Uniform(action_space_Rx[1], action_space_Rx[2])), 
    rand(Distributions.Uniform(action_space_Ry[1], action_space_Ry[2]))])
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

# Return human action of going CW or CCW. This is z
function human_policy(state::pomdpState)
    sH, sR, z = state.sH, state.sR, state.z
    center = Float32.([0., 0.])
    distR = norm(sR - center)

    if z == 0
        if distR > radius
            return pi/10.
        else
            return -pi/10.
        end
    end

    if z == 1
        if distR > radius
            return -pi/8.
        else
            return 0.
        end
    end
    # No influence
    if z == 2
        return pi/4.
    end
    # No influence
    if z == 3
        return -pi/2.
    end
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
    # Update timestep
    global_vars.time_step += 1

    # Robot action
    sR, z = state.sR, state.z
    aRx, aRy = actionR[1], actionR[2]
    sR_ = copy(sR)
    sR_ += [aRx, aRy]

    # Human action
    sH = state.sH
    actionH = global_vars.aH
    s_ = pomdpState(sH, sR_, z)
    if terminal_state(global_vars.time_step)
        global_vars.time_step = 0
        actionH = human_policy(state)
        actionH = Float32(actionH)
        global_vars.aH = actionH
        sH_ = copy(state.sH)
        sH_ += actionH
        if rand() > change_partner
            z_new = rand(0:3)
        else
            z_new = state.z
        end

        return Deterministic(pomdpState(sH_, reset_pos, z_new))
    else
        return Deterministic(s_)
    end
end;

# Define the observation space
function POMDPs.observation(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32}, statep::pomdpState)
    sH = copy(statep.sH)
    return Deterministic(pomdpObservation(Float32(sH)))
end;

# Define the reward function
function POMDPs.reward(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32})
    theta, sR = state.sH, state.sR
    human_state = [cos(theta), sin(theta)] .* radius
    robot_state = copy(sR)
    r = - norm(human_state - robot_state) * 100
    return r
end;
    
# MISC functions
POMDPs.discount(pomdp::pomdpWorld) = pomdp.discount_factor


function POMDPs.initialstate(pomdp::pomdpWorld)
    return Deterministic(pomdpState(init_theta, reset_pos, parsed_args["init_z"]))
end;


# Definition is done, lets solve it!
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
# Get a state and initialize belief
init = initialstate(pomdp)
state = rand(init)
# Initial Belief, 50/50 chance 
b = ParticleCollection([pomdpState(state.sH, state.sR, state.z) for i in 1:5000])

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
    # Update the belief
    b = ParticleFilters.update(belief_updater, b, aR, o)
    next!(p, step=1, showvalues=generate_showvalues(i, total_timesteps))
end

circ_human_pos = []
for theta in human_pos
    x = cos(theta) * radius
    y = sin(theta) * radius
    push!(circ_human_pos, [x, y])
end;

# In the same folder as the code, create subfolders
directoryPath = dirname(@__FILE__)
mkpath(directoryPath * "/figures")
mkpath(directoryPath * "/data")


#   Start visualization
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

function plot_circle(plt)
    thet = 0:0.01:2*pi
    circ_x = cos.(thet) * radius
    circ_y = sin.(thet) * radius
    plot!(plt, circ_x, circ_y, linewidth=1.5, xlims=[-2., 2.], ylims=[-2., 2.], label="")
end;

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
    plot_circle(plt)

    plot!(plt, [x[1] for x in robot_pos[start_idx:end_idx-1]], [x[2] for x in robot_pos[start_idx:end_idx-1]], xlimits=(-2., 2.), ylimits=(-2., 2.), linewidth=1.5, marker=:circle, color=:orange)
    plot!(plt, [x[1] for x in circ_human_pos[start_idx:end_idx-2]], [x[2] for x in circ_human_pos[start_idx:end_idx-2]], xlimits=(-2., 2.), ylimits=(-2., 2.), linewidth=1.5, marker=:circle, color=:green)
    annotate!(plt, -1.2, -1.6, text("Interaction Reward: $(round(sum(reward[start_idx:end_idx-1]), digits=2))", 10, :left, "Computer Modern"))

    total_reward2 += sum(reward[start_idx:end_idx-1])
    annotate!(plt, -1.2, -1.8, text("Total Reward: $(round(total_reward2, digits=2))", 10, :left, "Computer Modern"))

    start_idx = end_idx
    
    Plots.frame(anim)
    
    next!(p)
end

mp4(anim, directoryPath * "/figures/circle_unified_int_orig_" * parsed_args["name"] * "_htype" * "_$(parsed_args["init_z"]).mp4", fps=5)

# plot robot reward per interaction
plt_rewards = plot(1:n_interactions, interaction_rewards, xlabel="Interaction", ylabel="Reward", title="Robot Reward per Interaction", legend=false)
savefig(plt_rewards, directoryPath * "/figures/interaction_rewards_" * parsed_args["name"] * ".svg")

########### Save the data ############
println("Total Reward: ", totalR)
# println("Reward: ", reward)
cum_reward = cumsum(reward)
println("Cumulative Reward: ", cum_reward[end])

# Create a DataFrame 
# Flatten the list elements into separate columns
df = DataFrame(
    time_step = timestep_list,
    robot_pos_x=[pos[1] for pos in robot_pos],
    robot_pos_y=[pos[2] for pos in robot_pos],
    human_pos=human_pos,
    robot_a_x=[a[1] for a in robot_a],
    robot_a_y=[a[2] for a in robot_a],
    human_z=human_z,
    reward=reward,
    cum_reward=cum_reward
)

# save the data w/ no headers
CSV.write(directoryPath * "/data/circle_unified_" * parsed_args["name"] * ".csv", df, writeheader=false)