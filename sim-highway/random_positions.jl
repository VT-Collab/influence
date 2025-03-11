"""
A Unified Framework for Robots that Influence Humans over Long-Term Interaction

Highway environment simulation

This script generated random numbers so all the cars start from the same position.

"""

using Random
using JSON

# Number of interactions
const NUM_INTERACTIONS = 101

# Generate random X positions for each interaction
positions = Dict("robot" => [], "human" => [])
for i in 1:NUM_INTERACTIONS
    push!(positions["robot"], rand([-2.0, 2.0]))
    push!(positions["human"], rand([-2.0, 2.0]))
end

# Save positions to a JSON file in the same directory as this script
script_dir = @__DIR__
file_path = joinpath(script_dir, "positions.json")
open(file_path, "w") do f
    JSON.print(f, positions)
end

println("Random positions saved to $file_path")