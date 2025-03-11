# Code to run the sims for 50 different times
directoryPath = dirname(@__FILE__)

num_experiments = 2

for i in 1:num_experiments
    println("Running Experiment $i/$num_experiments")
    run(`julia $directoryPath/robot-unified-sim.jl --name $i`)
end