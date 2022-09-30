using ArgParse
using DataFrames
using Flux
import JSON
using LinearAlgebra
using MLUtils
using Plots
using StatsBase
using StatsPlots

include("helpers.jl")
include("stats.jl")


# CLI
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--data-directory"
            help = "Target directory storing data"
            arg_type = String
        "--n-folds"
            help = "Number of cross validation folds to use. Recommended 5 or 10"
            arg_type = Int16
            default = Int16(5)
        "--depth-range"
            help = "Range of depths to search over"
            nargs = '*'
            default = [Int16(10), Int16(60)]
        "--width-range"
            help = "Range of widths to search over"
            nargs = '*'
            default = [Int16(2), Int16(7)]
        "--depth-steps"
            help = "Number of depths to search over in specified depth-range"
            arg_type = Int16
            default = Int16(5)
        "--width-steps"
            help = "Number of widths to search over in specified width-range"
            arg_type = Int16
            default = Int16(5)
        "--n-epochs"
            help = "Number of epochs to train each model on"
            arg_type = Int32
            default = Int32(3)
        "--loss"
            help = "Loss function to use. Can be one of \"mse\" (recommended) or \"mae\""
            arg_type = String
            default = "mse"
        "--log-training-starts"
            help = "Print which model is being trained"
            arg_type = Bool
            default = true
        "--log-folds"
            help = "Print when each fold of CV beings"
            arg_type = Bool
            default = false
        "--log-training-loss"
            help = "Print training loss per epoch"
            arg_type = Bool
            default = false
        "--outfile"
            help = "Filename to record results"
            arg_type = String
            default = "results.json"
    end

    return parse_args(s)
end


# reading in data + sanity checking
function getrawdata(target_directory::String)
    x_raw_df = DataFrame(
        DVAR1=Float32[],
        DVAR2=Float32[],
        DVAR3=Float32[],
        DVAR4=Float32[],
        DVAR5=Float32[],
        DVAR6=Float32[],
        DVAR7=Float32[],
        DVAR8=Float32[],
        DVAR9=Float32[],
        DVAR10=Float32[],
        DVAR11=Float32[],
        DVAR12=Float32[],
        DVAR13=Float32[],
        DVAR14=Float32[]
    )
    y_raw_df = DataFrame(
        OBJ1=Float32[],
        OBJ2=Float32[],
        OBJ3=Float32[],
        OBJ4=Float32[],
        OBJ5=Float32[],
        OBJ6=Float32[]
    )

    try
        x_raw_df, y_raw_df = readjsonsfromdirectory(target_directory, x_raw_df, y_raw_df)
    catch e
        println("You've entered an invalid target data directory.")
    end
        
    return x_raw_df, y_raw_df
end


function neuralnetwork(x_dimension::Int16, y_dimension::Int16, width::Int16, depth::Int16)
    Chain(
        Dense(x_dimension, width, x->σ.(x)),
        (Dense(width, width, x->σ.(x)) for _ in 1:depth)...,
        Dense(width, y_dimension)
    )
end


function buildandtrain(
    x_train,
    y_train;
    width::Int16,
    depth::Int16,
    n_epochs::Int32=100,
    batchsize::Int=1024,
    optimizer=ADAM(),
    loss_function=Flux.mse,
    log_training::Bool=false
)
    # batch data
    data_loader = Flux.Data.DataLoader((x_train', y_train'), batchsize=batchsize, shuffle=true)
    
    # instantiating the model
    m = neuralnetwork(Int16(size(x_train)[2]), Int16(size(y_train)[2]), width, depth)

    # training
    loss(x, y) = loss_function(m(x), y)
    training_losses = Float32[]
    epochs = Int32[]

    for epoch in 1:n_epochs
        Flux.train!(loss, Flux.params(m), data_loader, optimizer)
        push!(epochs, epoch)

        l = 0.
        for d in data_loader
            l += loss(d...)
        end

        if log_training
            println("epoch $epoch, loss=$l")
        end

        push!(training_losses, l)
    end

    return m, training_losses
end


function crossvalidate(
    x_train,
    y_train;
    n_folds::Int16=5,
    width::Int16,
    depth::Int16,
    n_epochs::Int32=100,
    batchsize::Int=1024,
    optimizer=ADAM(),
    loss_function=Flux.mse,
    log_training::Bool=false
)
    scores_total = initscoresdict(n_folds; include_losses=true)
    scores_by_response = Dict("OBJ$i"=>initscoresdict(n_folds) for i in 1:6)

    folds = kfolds(size(x_train)[1]; k=n_folds)

    for (fold_id, (train_temp_idxs, val_temp_idxs)) in collect(enumerate(Iterators.zip(folds...)))
        if log_folds
            println("fold $fold_id of w,d=$width,$depth beginning")
        end

        # select training and validation sets
        x_train_temp, x_val_temp = x_train[train_temp_idxs, :], x_train[val_temp_idxs, :]
        y_train_temp, y_val_temp = y_train[train_temp_idxs, :], y_train[val_temp_idxs, :]

        # train model
        m, training_losses = buildandtrain(
            x_train_temp, y_train_temp;
            width=width, depth=depth, n_epochs=n_epochs, batchsize=batchsize, optimizer=optimizer,
            loss_function=loss_function, log_training=log_training
        )

        # gather predictions
        y_train_temp_preds = m(x_train_temp')'; y_val_temp_preds = m(x_val_temp')'

        # update aggregate scores
        n_features = size(x_train_temp, 2)
        updatescoresdict!(
            scores_total, fold_id, y_train_temp, y_train_temp_preds, y_val_temp, y_val_temp_preds,
            n_features, training_losses
        )

        # update scores by objective
        for j in 1:6
            updatescoresdict!(
                scores_by_response["OBJ$j"], fold_id, y_train_temp[:, j], y_train_temp_preds[:, j],
                y_val_temp[:, j], y_val_temp_preds[:, j], n_features)
        end
    end
    return Dict("total"=>scores_total, "by_response"=>scores_by_response)
end



function main()
    # gather arguments
    parsed_args = parse_commandline()
    target_directory = parsed_args["data-directory"]
    depth_range = [parse(Int16, s) for s in parsed_args["depth-range"]]
    width_range = [parse(Int16, s) for s in parsed_args["width-range"]]
    depth_steps = parsed_args["depth-steps"]
    width_steps = parsed_args["width-steps"]
    n_epochs = parsed_args["n-epochs"]
    loss_function_string = parsed_args["loss"]
    log_training_starts = parsed_args["log-training-starts"]
    log_folds = parsed_args["log-folds"]
    log_training_loss = parsed_args["log-training-loss"]
    n_folds = parsed_args["n-folds"]
    outfile = parsed_args["outfile"]

    if ~endswith(outfile, ".json")
        throw("Outfile must be a .json file")
    end

    x_df, y_df = getrawdata(target_directory)

    x_scaled_df = minmaxscaledf(x_df)
    y_scaled_df = minmaxscaledf(y_df)

    x_train_df, x_test_df, y_train_df, y_test_df = traintestsplit(x_scaled_df, y_scaled_df, 0.8)

    # format to arrays
    x_train = Matrix(x_train_df); x_test = Matrix(x_test_df);
    y_train = Matrix(y_train_df); y_test = Matrix(y_test_df);

    # training parameters
    depths = stratifyarchitecturedimension(depth_range..., depth_steps)
    widths = stratifyarchitecturedimension(width_range..., width_steps)
    batchsize = 1024
    optimizer=ADAM() # can't change this for now
    loss_function = loss_function_string == "mse" ? Flux.mse : Flux.mae

    # instantiating outdata container
    outdata = Vector{Dict}(undef, length(depths)*length(widths))

    # training
    Threads.@sync begin
        Threads.@threads for (idx, (width, depth)) in collect(enumerate(Iterators.product(widths, depths)))
            if log_training_starts
                println("training width=$width, depth=$depth on thread $(Threads.threadid())")
            end

            cv_scores = crossvalidate(
                x_train, y_train;
                n_folds=n_folds, width=width, depth=depth, n_epochs=n_epochs, 
                loss_function=loss_function, log_training=log_training_loss
            )

            # recording results
            outdata_dict = Dict(
                "configs"=>Dict(
                    "n_folds"=>n_folds,
                    "width"=>width,
                    "depth"=>depth,
                    "n_epochs"=>n_epochs,
                    "batchsize"=>batchsize,
                    "optimizer"=>"ADAM",
                    "loss_function"=>loss_function_string
                ),
                "results"=>cv_scores
            )

            outdata[idx] = outdata_dict
        end
    end

    open(outfile, "a") do f
        JSON.print(f, outdata, 4)
    end

end


main()