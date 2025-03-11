# Code to run the sims for 50 different times
directoryPath = dirname(@__FILE__)

num_experiments = 50

for i in 1:num_experiments
    println("Running Experiment $i/$num_experiments")
    run(`julia $directoryPath/circle-unified-sim.jl --name $i`)
end