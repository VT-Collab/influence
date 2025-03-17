"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Highway environment simulation

Run random_positions.jl to create the positions.json file before running this script
"""
### Comment out the Pkg stuff after the first run
Pkg.add(["POMDPs", "POMDPTools", "DiscreteValueIteration", "CSV", "DataFrames", 
                        "ArgParse", "Plots", "POMCPOW", "BasicPOMCP", "Distributions", 
                        "ParticleFilters", "ProgressMeter", "LinearAlgebra", "StatsBase", 
                        "Gtk4", "Joysticks", "Observables", "PyCall"])
# Pkg.update()

# importing the packages
using POMDPs, POMDPTools
using Plots; default(fontfamily="Computer Modern", framestyle=:box, wtitle="Test")
using Plots.PlotMeasures, ProgressMeter
using ArgParse, CSV, DataFrames 
using POMCPOW, BasicPOMCP
using Distributions, ParticleFilters
using Random, LinearAlgebra, StatsBase, Gtk4, Joysticks, Observables
using ParticleFilters: LowVarianceResampler
using Base: Timer, Threads
using Printf
using JSON
using Dates

ProgressMeter.ijulia_behavior(:clear)

directoryPath = dirname(@__FILE__)
mkpath(directoryPath * "/unified-hw-res")

# paths for the random positions
positions_file_path = joinpath(directoryPath, "positions.json")
positions = JSON.parsefile(positions_file_path)

# define the arguments
function parse_commandline()
    arg = ArgParseSettings()

    @add_arg_table arg begin
        "--user_id"
            help="User ID"
            arg_type=Int32
            default=00 # for the sim
        "--int_multiplier"
            help="Interaction multiplier"
            arg_type=Int32
            default=100
        "--discount_factor" 
            help="discount factor" 
            arg_type=Float64
            default=0.95
    end
    return parse_args(arg)
end

# parse the arguments
parsed_args = parse_commandline()

#   User study and data
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
USER_ID = parsed_args["user_id"]
dataset = []
global human_switched_to_1 = 0
global human_switches_per_interaction = zeros(Int, parsed_args["int_multiplier"])
global robot_ahead_counter = 0
global robot_ahead_results = zeros(Int, parsed_args["int_multiplier"])
#   Constants
#   ≡≡≡≡≡≡≡≡≡≡≡
# Dynamics constants
const dt = Float32(0.2) # adjust as needed
const friction = Float32(0.1) # friction coefficient, adjust as needed
const time_horizon = Int32(120); # length of interaction
const MAX_ITERATIONS = time_horizon * parsed_args["int_multiplier"]   # maximum number of timesteps
# Environment
const road_length = Float32(1000) # in pixels
const road_width = Float32(800)
const highway_width = Float32(6.) # in meters
const car_radius = 30 # in pixels
const safe_distance = Float32(2.5)  # in meters, for human reward calculation
# Robot
const init_pos_R = Float32.([2.0, 10.])
const reset_vel_R = Float32.([0.0, 1.5]) # robot velocity
const reset_ang_R = Float32(0.0) # robot angular velocity
const reset_accel_R = Float32(0.0) # robot acceleration
const reset_heading_R = Float32(pi/2.)
const heading_bound_R = Float32.([pi/4., 3*pi/4.]) # Robot heading angle bounds
const accel_bound_R = Float32.([-1.0, 1.0]) # Robot acceleration bounds
const max_speed_R = Float32(1.75)
const min_speed_R = Float32(0.8)
# Human
const init_pos_H = Float32.([-2.0, 6.0])
const reset_vel_H = Float32.([0.0, 2.0])
const reset_ang_H = Float32(0.0)
const reset_accel_H = Float32(0.0)
const reset_heading_H = Float32(pi/2.)
const accel_bound_H = Float32.([-2.0, 20.0])
const max_speed_H = Float32(15.0)
const min_speed_H = Float32(0.0)

#   Changing variables
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
totalR = Float32(0.0) # for the robot
d = Float32(1.0)
counter = Int32(0)
global inter_num = Int32(1)

mutable struct GlobalVars
    aH::Vector{Float32}  # [heading, acceleration]
    interaction_score::Float32
    total_score::Float32 # for the human
    collisions::Int32
end
global_vars = GlobalVars(Float32.([reset_heading_H, reset_accel_H]), 0.0, 0.0, 0)

# define the state type
mutable struct pomdpState
    time_step::Int32
    sR::Float32 # robot heading angle
    pR::Vector{Float32} # robot position
    vR::Vector{Float32} # robot linear velocity for x and y
    angR::Float32 # robot angular velocity
    accelR::Float32 # robot acceleration
    sH::Float32 # human heading angle
    pH::Vector{Float32} # human position
    vH::Vector{Float32} # human linear velocity for x and y
    angH::Float32 # human angular velocity
    accelH::Float32 # human acceleration
end;

# define the observation type
struct pomdpObservation
    sR::Float32 # robot heading angle
    pR_1::Float32 # robot position x
    pR_2::Float32 # robot position y
    vR_1::Float32 # robot velocity x
    vR_2::Float32 # robot velocity y
    angR::Float32 # robot angular velocity
    pH_1::Float32 # human position x
    pH_2::Float32 # human position y
end;

#   Define the state, action, and observation spaces
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# Define continuous state space with each element of pomdpState type
struct continuousStateSpace end; 
    Base.eltype(sspace::Type{continuousStateSpace}) = pomdpState

struct continuousActionSpace <: AbstractRNG end;
function POMDPs.rand(rng::AbstractRNG, aspace::continuousActionSpace)
    return Float32.([rand(Distributions.Uniform(heading_bound_R[1], heading_bound_R[2])), 
                        rand(Distributions.Uniform(accel_bound_R[1], accel_bound_R[2]))                 
                            ])
end;

# Define continuous observation space with each element of pomdpObservation type
struct continuousObservationSpace end;
Base.eltype(ospace::Type{continuousObservationSpace}) = pomdpObservation
    
#   Define the POMDP World
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# defining the pomdp model
mutable struct pomdpWorld <: POMDP{pomdpState, Vector{Float32}, pomdpObservation}
    discount_factor::Float64 
end;

# Constructor for convenience
function pomdpWorld(;discount_factor::Float64=parsed_args["discount_factor"])
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
    @inbounds begin
        time_step = state.time_step
        time_step += 1

        sR, pR, vR, angR, accelR = state.sR, state.pR, state.vR, state.angR, state.accelR
        sH, pH, vH, angH, accelH = state.sH, state.pH, state.vH, state.angH, state.accelH

        # Robot action
        sR_ = copy(sR)
        headR, accelR_ = actionR[1], actionR[2]
        speedR = Float32(norm(vR))

        new_angular_velocityR = Float32(speedR * headR)
        new_accelerationR = Float32(accelR_ - friction * speedR)

        new_headingR = Float32(headR + (state.angR + new_angular_velocityR) * dt / 2.0)
        new_speedR = Float32(clamp(speedR + (state.accelR + new_accelerationR) * dt / 2.0, min_speed_R, max_speed_R))

        new_velocityR = Float32.([(speedR + new_speedR) / 2.0 * cos((new_headingR + headR) / 2.0),
                                  (speedR + new_speedR) / 2.0 * sin((new_headingR + headR) / 2.0)])

        pR_ = Float32.([pR[1], pR[2]]) + (vR + new_velocityR) * dt / 2.0
        sR_ = new_headingR

        # Human action
        actionH = global_vars.aH
        sH_ = copy(sH)
        pH_ = copy(pH)
        vH_ = copy(vH)
        angH_ = copy(angH)
        accelH_ = copy(accelH)
        # Human input
        sH_ = Float32(actionH[1])
        accelH_ = Float32(actionH[2])
        speedH = Float32(norm(state.vH))
        new_angular_velocityH = Float32(speedH * sH_)
        new_accelerationH = Float32(accelH_ - friction * speedH)
        new_headingH = Float32(state.sH + (state.angH + new_angular_velocityH) * dt / 2.0)
        new_speedH = Float32(clamp(speedH + (state.accelH + new_accelerationH) * dt / 2.0, min_speed_H, max_speed_H))
        new_velocityH = Float32.([(speedH + new_speedH) / 2.0 * cos((new_headingH + state.sH) / 2.0),
                                  (speedH + new_speedH) / 2.0 * sin((new_headingH + state.sH) / 2.0)])
        pH_ = Float32.([pH[1], pH[2]]) + (vH_ + new_velocityH) * dt / 2.0
        sH_ = Float32(mod(new_headingH, 2*pi))
        new_accelerationH = Float32(clamp(new_accelerationH, accel_bound_H[1], accel_bound_H[2]))

        s_ = pomdpState(time_step, sR_, pR_, new_velocityR, new_angular_velocityR, new_accelerationR, sH_, pH_, new_velocityH, new_angular_velocityH, new_accelerationH)

        if terminal_state(time_step)
            time_step = 0
            return Deterministic(pomdpState(time_step, sR_, Float32.([positions["robot"][inter_num], init_pos_R[2]]), reset_vel_R, reset_ang_R, reset_accel_R, reset_heading_H, Float32.([positions["human"][inter_num], init_pos_H[2]]), reset_vel_H, reset_ang_H, reset_accel_H))
        else
            return Deterministic(s_)
        end
    end
end;

# Define the observation space
function POMDPs.observation(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32}, statep::pomdpState)
    sR = copy(statep.sR)
    pR = copy(statep.pR)
    pR_1 = pR[1]
    pR_2 = pR[2]
    vR = copy(statep.vR)
    vR_1 = vR[1]
    vR_2 = vR[2]
    angR = copy(statep.angR)
    pH = copy(statep.pH)
    pH_1 = pH[1]
    pH_2 = pH[2]
    return Deterministic(pomdpObservation(Float32(sR), Float32(pR_1), Float32(pR_2), Float32(vR_1), Float32(vR_2) ,Float32(angR), Float32(pH_1), Float32(pH_2)))
end;

# Define the reward function
@fastmath function POMDPs.reward(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32})
    sH, pR, pH = state.sH, state.pR, state.pH

    dist_to_human = max(0., safe_distance - sqrt((pR[1] - pH[1])^2 + (pR[2] - pH[2])^2))
    block_human = (pH[1] - pR[1]) ^ 2
    heading = (pi/2. - action[1]) ^ 2

    r = dist_to_human * 10. + block_human + heading * 2.5

    return -r
end;
    
# Human reward for GUI
function human_reward(pomdp::pomdpWorld, state::pomdpState, action::Vector{Float32})
    sH, pR, pH = state.sH, state.pR, state.pH
    interaction_score_ = copy(global_vars.interaction_score)
    if -(highway_width/2.) <= state.pH[1] <= (highway_width/2.)
        interaction_score_ += state.pH[2] / 10.
    else
        interaction_score_ -= 10
    end
    # If there is a collision
    if sqrt((pR[1] - pH[1])^2 + (pR[2] - pH[2])^2) < safe_distance
        interaction_score_ -= 50
        global_vars.collisions += 1
        # println("Collision!")
    end
    global_vars.interaction_score = interaction_score_
    return interaction_score_
end;
    
# MISC functions
POMDPs.discount(pomdp::pomdpWorld) = pomdp.discount_factor

function POMDPs.initialstate(pomdp::pomdpWorld)
    return Deterministic(pomdpState(0, reset_heading_R, init_pos_R, reset_vel_R, reset_ang_R, reset_accel_R, reset_heading_H, init_pos_H, reset_vel_H, reset_ang_H, reset_accel_H))
end;

# Definition is done, lets solve it!
pomdp = pomdpWorld();
solver = POMCPOWSolver(eps= 1e-3, max_depth=20, criterion=MaxUCB(10.0), tree_queries=200, k_observation= 10, alpha_observation = 0.1, rng=MersenneTwister(42));
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

# Create the BasicParticleFilter with the custom resampler
resampler = POMDPResampler(5000)
belief_updater = BasicParticleFilter(pomdp, pomdp, resampler, 5000, MersenneTwister(42))


# Get a state and initialize belief
init = initialstate(pomdp)
state = rand(init)
# Initial Belief, equal chance of being in any state
b = WeightedParticleBelief([pomdpState(0, state.sR, state.pR, state.vR, state.angR, state.accelR, state.sH, state.pH, state.vH, state.angH, state.accelH) for i in 1:5000], [1/5000 for i in 1:5000])
state_new = init.val

# Initialize state variables
human_pos = [state_new.pH]  # Initialize with a starting position
robot_pos = [state_new.pR]
reward = []
belief = []

p = Progress(MAX_ITERATIONS, showspeed=true)
generate_showvalues(iter, total) = () -> [(:timestep, iter), (:total, total)]

# Create a GtkApplication
app = GtkApplication("com.example.Unified")

function human_cost(human_action, robot_action, state::pomdpState)
    # Create a copy of the current state
    state_copy = deepcopy(state)

    # Apply the actions to the copy
    headR, accelR_ = robot_action[1], robot_action[2]
    speedR = norm(state_copy.vR)
    new_angular_velocityR = speedR * headR
    new_accelerationR = accelR_ - friction * speedR
    new_headingR = headR + (state_copy.angR + new_angular_velocityR) * dt / 2.0
    new_speedR = clamp(speedR + (state_copy.accelR + new_accelerationR) * dt / 2.0, min_speed_R, max_speed_R)
    new_velocityR = [
        (speedR + new_speedR) / 2.0 * cos((new_headingR + headR) / 2.0),
        (speedR + new_speedR) / 2.0 * sin((new_headingR + headR) / 2.0)
    ]
    state_copy.pR[1] += new_velocityR[1] * dt
    state_copy.pR[2] += new_velocityR[2] * dt
    state_copy.sR = new_headingR
    state_copy.accelR = new_accelerationR
    state_copy.vR = new_velocityR

    headH, accelH_ = human_action[1], human_action[2]
    speedH = norm(state_copy.vH)
    new_angular_velocityH = speedH * headH
    new_accelerationH = accelH_ - friction * speedH
    new_headingH = headH + (state_copy.angH + new_angular_velocityH) * dt / 2.0
    new_speedH = clamp(speedH + (state_copy.accelH + new_accelerationH) * dt / 2.0, min_speed_H, max_speed_H)
    new_velocityH = [
        (speedH + new_speedH) / 2.0 * cos((new_headingH + headH) / 2.0),
        (speedH + new_speedH) / 2.0 * sin((new_headingH + headH) / 2.0)
    ]
    state_copy.pH[1] += new_velocityH[1] * dt
    state_copy.pH[2] += new_velocityH[2] * dt
    state_copy.sH = new_headingH
    state_copy.accelH = new_accelerationH
    state_copy.vH = new_velocityH

    dist_to_robot = max(0.0, 4.0 - sqrt(
        (state_copy.pR[1] - state_copy.pH[1])^2 + (state_copy.pR[2] - state_copy.pH[2])^2
    ))^2
    
    dist_to_center = (0.4 * state_copy.pH[1])^4
    speed_of_human = sign(state_copy.vH[2]) * state_copy.vH[2]^2

    return dist_to_robot * 10. + dist_to_center * 6. - speed_of_human * 15.
end;

function boltzmann_human(beta, UR, state::pomdpState, n_samples)
    global human_switched_to_1, human_switches_per_interaction, inter_num, robot_ahead_counter, robot_ahead_results

    local_UR = copy(UR)

    # Check if the robot is ahead at one timestep before the last of the interaction
    if state.time_step % time_horizon == time_horizon - 1
        if state.pR[2] > state.pH[2]
            robot_ahead_results[inter_num] = 1
        else
            robot_ahead_results[inter_num] = 0
        end
    end

    # For interactions 11 and after, check if the robot was ahead in more than 50% of the last 6 interactions
    if inter_num >= 11
        recent_results = robot_ahead_results[inter_num-6:inter_num-1]
        if sum(recent_results) > 0.5 * 10 || inter_num % 10 == 0
            # Check for collision
            if sqrt((state.pR[1] - state.pH[1])^2 + (state.pR[2] - state.pH[2])^2) >= safe_distance
                local_UR = [reset_heading_R, reset_accel_R]
                human_switched_to_1 += 1
                # println("Human switched to 1st at interaction $inter_num")
                human_switches_per_interaction[inter_num] += 1
            end
        end
    end

    UH = rand(Distributions.Uniform(-0.15, 0.15), n_samples, 2)
    P = zeros(Float64, n_samples)

    for idx in 1:n_samples
        cost = human_cost(UH[idx, :], local_UR, state)
        if isfinite(cost)
            P[idx] = exp(-beta * cost)
        else
            P[idx] = 0.0
        end
    end

    total_P = sum(P)
    if total_P == 0.0
        P .= 1.0 / n_samples
    else
        P /= total_P
    end

    idx_star = rand(Categorical(P))
    return UH[idx_star, :]
end;

function human_action(beta, UR, state::pomdpState, n_samples)
    return boltzmann_human(beta, UR, state, n_samples)
end;

# Define the update function
function update_state(reward_label_1, reward_label_2, collision_label, canvas, win)
    global state_new, totalR, d, b, aH
    global human_a, human_pos, robot_pos
    global reward, belief, start_idx, counter, inter_num
    global dataset

    # Check for a stopping condition
    if counter >= MAX_ITERATIONS
        close(win) 
        return false
    end

    # Get robot action from the policy
    aR = action(policy, b)
    
    # Get human action
    heading_value, acceleration_value = human_action(1.0, aR, state_new, 100)  # Adjust beta and n_samples as needed
    global_vars.aH[1] = heading_value
    global_vars.aH[2] = clamp(acceleration_value, accel_bound_H[1], accel_bound_H[2])

    # Human reward for GUI
    rH = human_reward(pomdp, state_new, global_vars.aH)

    # average reward for all interactions for the GUI
    if inter_num == 1
        avg_score = global_vars.total_score / inter_num
    else
        avg_score = global_vars.total_score / (inter_num - 1)
    end

    # Update the reward label text
    Gtk4.text(reward_label_1, @sprintf("Interaction Reward: %.2f", rH))
    Gtk4.text(reward_label_2, @sprintf("Average Reward: %.2f", avg_score))

    # Check for collision and update the collision label
    if sqrt((state_new.pR[1] - state_new.pH[1])^2 + (state_new.pR[2] - state_new.pH[2])^2) < safe_distance
        Gtk4.text(collision_label, "Collision!")
    else
        Gtk4.text(collision_label, "")
    end

    # Robot takes an action
    state_new, o, r = @gen(:sp, :o, :r)(pomdp, state_new, aR)
  
    push!(reward, r)
    
    totalR += r
    d *= discount(pomdp)
    
    # Update belief
    pm = particle_memory(pomdp)  # Use the prediction model
    resize!(pm, length(b.particles))  # So the particle memory is correctly sized
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

    push!(human_pos, state_new.pH)
    push!(robot_pos, state_new.pR)

    # Append state and action data to the dataset
    state_data = [counter, state_new.sR, state_new.pR[1], state_new.pR[2], state_new.vR[1], state_new.vR[2],
                     state_new.angR, state_new.accelR, state_new.sH, state_new.pH[1], 
                        state_new.pH[2], state_new.vH[1], state_new.vH[2], state_new.angH, 
                            state_new.accelH, global_vars.collisions, 
                                global_vars.interaction_score, global_vars.total_score, r]
    action_data = [aR[1], aR[2], global_vars.aH[1], global_vars.aH[2]]  # Flatten action_data
    combined_data = vcat(state_data, action_data)  # Concatenate the arrays vertically
    push!(dataset, combined_data)

    counter += 1

    # Increment interaction number if time horizon is reached
    if counter % time_horizon == 0
        inter_num += 1
        global_vars.aH = Float32.([reset_heading_H, reset_accel_H])  # Reset the human action
        global_vars.total_score += rH
        global_vars.interaction_score = 0.0
        robot_ahead_counter = 0  # Reset the robot ahead counter
    end

    # Redraw the canvas
    Gtk4.draw((canvas) -> draw_environment(canvas, state_new, aR), canvas)
    Gtk4.reveal(canvas)

    if counter >= MAX_ITERATIONS
        close(win)
    
        # Save the dataset to a .json file
        current_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        file_name = "highway_unified_u$(USER_ID)_$current_time.json"
        file_path = joinpath(directoryPath, "unified-hw-res", file_name)
        open(file_path, "w") do f
            JSON.print(f, dataset)
        end
    
        return false
    end

    return true
end;

function draw_environment(canvas, state::pomdpState, actionR::Vector{Float32})
    global robot_pos, human_pos

    cr = Gtk4.getgc(canvas)

    # Clear the canvas
    set_source_rgb(cr, 1, 1, 1)  # White background
    paint(cr)
    # Draw the road
    set_source_rgb(cr, 0.8, 0.8, 0.8)  # Grey road
    rectangle(cr, 0, 0, road_width, road_length)
    fill(cr)
    # Draw the dark grey rectangle in the middle of the road
    set_source_rgb(cr, 0.2, 0.2, 0.2)  # Dark grey color
    rectangle(cr, 300, 0, 200, road_length)  # Rectangle spanning the middle of the road
    fill(cr)
    # Draw the solid yellow end lane markings
    set_source_rgb(cr, 0.969, 0.710, 0)
    set_line_width(cr, 10)
    for x in [295, 505]
        move_to(cr, x, 0)
        line_to(cr, x, road_length)
        Gtk4.stroke(cr)
    end
    # Draw the lane markings as dotted lines
    set_source_rgb(cr, 1, 1, 1)  # White lane markings
    set_line_width(cr, 4)
    set_dash(cr, Float64[30.0, 50.0], 0.0) # Set the dash pattern to be 10 pixels on, 10 pixels off
    for x in [400]
        move_to(cr, x, 0)
        line_to(cr, x, road_length)
        Gtk4.stroke(cr)
    end
    set_dash(cr, Float64[], 0.0)  # Reset the dash pattern to solid line

    # Draw the robot if positions are available
    if !isempty(robot_pos)
        set_source_rgb(cr, 1, 0.5, 0)  # Orange robot
        robot_x, robot_y = robot_pos[end]
        robot_x_canvas = robot_x * 25 + (road_width/2) 
        robot_y_canvas = road_length - robot_y * 25
        save(cr)  # Save the current state of the context
        Gtk4.translate(cr, robot_x_canvas, robot_y_canvas)  # Translate the context to the robot position
        Gtk4.rotate(cr, -actionR[1])  # Rotate the context to the robot angle
        arc(cr, 0, 0, car_radius, 0, 2 * pi)  # Draw a circle
        fill(cr)
        restore(cr)  # Restore the context to the saved state
    end

    # Draw the human if positions are available
    if !isempty(human_pos)
        set_source_rgb(cr, 0, 0, 1)  # Blue human
        human_x, human_y = human_pos[end]
        human_x_canvas = human_x * 25 + (road_width/2)
        human_y_canvas = road_length - human_y * 25
        save(cr) 
        Gtk4.translate(cr, human_x_canvas, human_y_canvas)
        Gtk4.rotate(cr, -state.sH)
        arc(cr, 0, 0, car_radius, 0, 2 * pi)
        fill(cr)
        restore(cr)
    end

    return true
end

# Run the Gtk application
function on_activate(app)
    # Create a window
    win = GtkApplicationWindow(app, "Highway Unified"; default_width=road_width, default_height=road_length)

    # Apply CSS styling
    css = """
    window {
        background-color: #FFFFFF;
    }
    .reward_label_1 {
        font-family: 'Georgia';
        font-size: 24px;
        color: #000000;  /* Black color */
    }
    .reward_label_2 {
        font-family: 'Georgia';
        font-size: 24px;
        color: #000000;  /* Black color */
    }
    .collision_label {
        font-family: 'Georgia';
        font-size: 30px;
        font-weight: bold;
        color: #FF0000;  /* Red color */
    }
    """

    cssProvider = GtkCssProvider(css)
    push!(Gtk4.display(win), cssProvider)

    # Create a vertical box to hold the labels and canvas
    vbox = GtkBox(:v)
    push!(win, vbox)

    # Create a label for displaying the human reward
    reward_label_1 = GtkLabel("Interaction Reward: 0.0")
    Gtk4.add_css_class(reward_label_1, "reward_label_1")
    push!(vbox, reward_label_1)

    reward_label_2 = GtkLabel("Average Reward: 0.0")
    Gtk4.add_css_class(reward_label_2, "reward_label_2")
    push!(vbox, reward_label_2)

    # Create a label for displaying the collision message
    collision_label = GtkLabel("")
    Gtk4.add_css_class(collision_label, "collision_label")
    push!(vbox, collision_label)

    # Create a canvas for drawing
    canvas = GtkCanvas()
    push!(vbox, canvas)

    # Set properties for the box and widgets
    vbox.spacing = 10
    canvas.hexpand = true
    canvas.vexpand = true
    reward_label_1.hexpand = true
    reward_label_2.hexpand = true
    collision_label.hexpand = true

    # Show the window
    show(win)

    # Connect the draw function to the canvas with a default actionR
    Gtk4.draw((canvas) -> draw_environment(canvas, state_new, Float32[pi/2., 0.0]), canvas)

    # Set up a timer to call update_state periodically
    timer = Timer(0.05) do t
        Gtk4.GLib.g_idle_add() do
            update_state(reward_label_1, reward_label_2, collision_label, canvas, win)
            false
        end
    end

    # Start the update thread
    Threads.@spawn while true
        sleep(0.05)
        Gtk4.GLib.g_idle_add() do
            update_state(reward_label_1, reward_label_2, collision_label, canvas, win)
            false
        end
    end
end;

signal_connect(on_activate, app, "activate")
Gtk4.run(app)