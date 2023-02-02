include("./scan_hyperparameters_emi4D_functions.jl")

@load "./models/2023-01-24_16-09-32_w=100_d=5_activation=sigmoid_bs=1024_lr=0.001_dr=0.0_2.jld2" m

# alternative:
# m = JLD2.load_object("./models/2023-01-24_16-09-32_w=100_d=5_activation=sigmoid_bs=1024_lr=0.001_dr=0.0_2.jld2")

println(m)
