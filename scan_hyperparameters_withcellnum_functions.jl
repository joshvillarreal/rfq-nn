using ArgParse
# using CUDA
#using BSON: @save
using DataFrames
using Flux
import JSON
using LinearAlgebra
using MLUtils
using StatsBase
using Plots
using JLD2, FileIO

include("helpers_temp.jl")
include("stats_temp.jl")


# CLI
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--data-directory"
            help = "Target directory storing data"
            arg_type = String
        "--n-folds"
            help = "Number of cross validation folds to use. Recommended 5 or 10"
            arg_type = Int
            default = 5
        "--depth-range"
            help = "Range of depths to search over"
            nargs = '*'
            default = [10, 60]
        "--width-range"
            help = "Range of widths to search over"
            nargs = '*'
            default = [2, 7]
        "--depth-steps"
            help = "Number of depths to search over in specified depth-range"
            arg_type = Int
            default = 5
        "--width-steps"
            help = "Number of widths to search over in specified width-range"
            arg_type = Int
            default = 5
        "--activation-functions"
            help = "Activation function to use. Can be one of \"sigmoid\", \"relu\", \"tanh\""
            nargs = '*'
            default = ["sigmoid"]
        "--batch-size-range"
            help = "Range of batch sizes to search over (values should be integer powers ot 2)"
            nargs = '*'
            default = ["512", "2048"]
        "--batch-size-steps"
            help = "Number of batch sizes to search over in specified batch-size-range"
            arg_type = Int
            default = 2
        "--learning-rate-range"
            help = "Range of learning rates to search over"
            nargs = '*'
            default = [0.0001, 0.01]
        "--learning-rate-steps"
            help = "Number of learning rates to search over in specified learning-rate-range"
            arg_type = Int
            default = 2
        "--dropout-rate-range"
            help = "Range of dropout rates to search over"
            nargs = '*'
            default = [0., 0.]
        "--dropout-rate-steps"
            help = "Number of dropout rates to search over in specified dropout-rate-range"
            arg_type = Int
            default = 2
        "--n-epochs"
            help = "Number of epochs to train each model on"
            arg_type = Int
            default = 3
        "--loss"
            help = "Loss function to use. Can be one of \"mse\" (recommended) or \"mae\""
            arg_type = String
            default = "mse"
        "--outfile"
            help = "Filename to record results"
            arg_type = String
            default = "results.json"
        "--log-training-starts"
            help = "Print which model is being trained"
            arg_type = Bool
            default = true
        "--log-training-loss"
            help = "Print training loss per epoch"
            arg_type = Bool
            default = false
        "--log-folds"
            help = "Print status of each fold"
            arg_type = Bool
            default = false
        "--gpu"
            help = "Flag to use GPU for training"
            action = :store_true
    end

    return parse_args(s)
end


function readjsonsfromdirectory_withcellnum(target_directory::String, x_df, y_df, cellnumber_df)
    # args x_df and y_df should be empty
    for file in readdir(target_directory)
        if endswith(file, ".json")
            data_raw = JSON.parsefile("$target_directory/$file")["samples"];

            for sample in data_raw
                features = sample[2]["dvar"]
                responses = sample[2]["obj"]

                cellnumber = Dict("cellnumber" => sample[2]["numcells"])

                push!(x_df, features)
                push!(y_df, responses)
                push!(cellnumber_df, cellnumber)
            end
        end
    end
    x_df, y_df, cellnumber_df
end


# reading in data + sanity checking
function getrawdata_withcellnum(target_directory::String)
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
        DVAR14=Float32[],
    )
    y_raw_df = DataFrame(
        OBJ1=Float32[],
        OBJ2=Float32[],
        OBJ3=Float32[],
        OBJ4=Float32[],
        OBJ5=Float32[],
        OBJ6=Float32[]
    )
    cellnumber_raw_df = DataFrame(
        cellnumber=Int[],
    )

    try
        x_raw_df, y_raw_df, cellnumber_raw_df = readjsonsfromdirectory_withcellnum(target_directory, x_raw_df, y_raw_df, cellnumber_raw_df)
    catch e
        println("Something went wrong.")
    end

    return x_raw_df, y_raw_df, cellnumber_raw_df
end


function neuralnetwork(x_dimension::Int, y_dimension::Int, width::Int, depth::Int, activation_function)
    Chain(
        Dense(x_dimension, width, x->activation_function(x)),
        (Dense(width, width, x->activation_function(x)) for _ in 1:depth)...,
        Dense(width, y_dimension)
    )
end


function neuralnetworkwithdropout(x_dimension::Int, y_dimension::Int, width::Int, depth::Int, dropout_rate::Float32, activation_function)
    Chain(
        Dense(x_dimension, width, x->activation_function(x)),
        (Chain(
            Dense(width, width, x->activation_function(x)),
            Dropout(dropout_rate),
        ) for _ in 1:depth)...,
        Dense(width, y_dimension)
    )
end


# TODO
function neuralnetworkwithbatchnorm(x_dimension::Int, y_dimension::Int, width::Int, depth::Int, activation_function)
    Chain()
end


function buildandtrain(
    x_train,
    y_train;
    width::Int,
    depth::Int,
    activation_function,
    n_epochs::Int=100,
    batchsize::Int=1024,
    optimizer=ADAM(),
    dropout_rate::Float32=0.0,
    loss_function=Flux.mse,
    log_training::Bool=false,
    model_id::String="",
    use_gpu::Bool=false
)
    # batch data
    data_loader = Flux.Data.DataLoader((x_train', y_train'), batchsize=batchsize, shuffle=true)
    
    # init model
    if use_gpu
        # Select GPU
        gpus = [gpu_id for gpu_id in devices()]
        device!(gpus[Threads.threadid()])
        # Create NN and offload to GPU
        m = neuralnetworkwithdropout(size(x_train)[2], size(y_train)[2], width, depth, dropout_rate, activation_function) |> gpu
    else
        m = neuralnetworkwithdropout(size(x_train)[2], size(y_train)[2], width, depth, dropout_rate, activation_function)
    end

    # training
    training_losses = Float32[]
    epochs = Int64[]
    l::Float32=0.0

    if use_gpu
        start_time = time()
        for epoch in 1:n_epochs

	    l=0.0
    
            for (xs, ys) in CuIterator(data_loader)
                gs = gradient(Flux.params(m)) do
		    y_pred = m(xs)
                    l += sum(loss_function(y_pred, ys))
                end
                Flux.Optimise.update!(optimizer, Flux.params(m), gs)
            end

            push!(epochs, epoch)
            push!(training_losses, l)
	    
	    if log_training && mod(epoch, 25) == 0.0
	        epoch_time = time() - start_time
	        println("    epoch $epoch, loss=$l, time=$epoch_time")
	    end
      	end
        end_time = time()

	m = cpu(m)

        # save model
        #@save "models/$model_id.bson" m
        model_state = Flux.state(m);
        jldsave("models/$model_id.jld2"; model_state)

        return m, training_losses, end_time-start_time
    else
        loss(x, y) = loss_function(m(x), y)
        start_time = time()
        for epoch in 1:n_epochs
            Flux.train!(loss, Flux.params(m), data_loader, optimizer)

            l = 0.0
            for d in data_loader
                l += loss(d...)
            end

            push!(epochs, epoch)
            push!(training_losses, l)
            if log_training
                println("    epoch $epoch, loss=$l")
            end
        end
        end_time = time()

        # save model
        # @save "models/$model_id.jld2" m
        model_state = Flux.state(m);
        jldsave("models/$model_id.jld2"; model_state)

        return m, training_losses, end_time-start_time
    end
end


function crossvalidate(
    x_train,
    y_train;
    n_folds::Int=5,
    width::Int,
    depth::Int,
    activation_function,
    n_epochs::Int=100,
    batchsize::Int=1024,
    optimizer=ADAM(),
    dropout_rate::Float32=0.,
    loss_function=Flux.mse,
    log_training::Bool=false,
    log_folds::Bool=false,
    model_id::String="",
    y_scalers=nothing,
    use_gpu::Bool=false
)
    
    scores_total = initscoresdict(n_folds; by_response=false)
    scores_by_response = Dict("OBJ$i"=>initscoresdict(n_folds; by_response=true) for i in 1:size(y_train)[2])

    train_temp_idxs, val_temp_idxs = kfolds(size(x_train)[1]; k=n_folds)

    for i in 1:n_folds
        if log_folds
            println("  - Fold $i of $model_id")
        end

        # select training and validation sets
        x_train_temp, x_val_temp = x_train[train_temp_idxs[i], :], x_train[val_temp_idxs[i], :]
        y_train_temp, y_val_temp = y_train[train_temp_idxs[i], :], y_train[val_temp_idxs[i], :]

        # train model
        m, training_losses, dt = buildandtrain(
            x_train_temp, y_train_temp;
            width=width, depth=depth, activation_function=activation_function,
            n_epochs=n_epochs, batchsize=batchsize, optimizer=optimizer, dropout_rate=dropout_rate,
            loss_function=loss_function, log_training=log_training, model_id=(model_id * "_$i"),
	    use_gpu=use_gpu
        )

        # gather predictions
        y_train_temp_preds = m(x_train_temp')'; y_val_temp_preds = m(x_val_temp')'

	# Test loading m back in as m2
	# m2 = JLD2.load_object("models/$(model_id * "_$i").jld2")
        # y_train_temp_preds2 = m2(x_train_temp')'; y_val_temp_preds2 = m2(x_val_temp')'
	
        # Q&D Plotting of OBJ6
        # if i == 1
        #    p1 = histogram(y_val_temp[:, 6])
	#    p2 = histogram(y_val_temp_preds2[:, 6])
	#    plot(p1, p2, layout=(1,2), legend=false)
	#    gui()
	#    readline()
	#end

        # update aggregate scores
        updatescoresdict!(
            scores_total, i, y_train_temp, y_train_temp_preds, y_val_temp, y_val_temp_preds,
            size(x_train_temp, 2); training_losses=training_losses, dt=dt
        )

        # update scores by objective
        for j in 1:size(y_train_temp)[2]
            y_scaler = y_scalers["OBJ$j"]
            updatescoresdict!(
                scores_by_response["OBJ$j"], i, y_train_temp[:, j], y_train_temp_preds[:, j],
                y_val_temp[:, j], y_val_temp_preds[:, j], size(x_train_temp, 2); y_scaler=y_scaler
            )
        end
    end
    return Dict("total"=>scores_total, "by_response"=>scores_by_response)
end



function main()
    # Do sanity check that we have exaclty N Threads == N CUDA devices
    # if length(devices()) != Threads.nthreads()
    #     println("N Threads must be equal N GPUs! Aborting...")
	# exit()
    # end

    # Temp
    println("Passed N Threads == N GPU sanity check!")

    # gather arguments
    println("Gathering arguments...")
    parsed_args = parse_commandline()
    target_directory = parsed_args["data-directory"]
    depth_range = [parse(Int64, s) for s in parsed_args["depth-range"]]
    width_range = [parse(Int64, s) for s in parsed_args["width-range"]]
    depth_steps = parsed_args["depth-steps"]
    width_steps = parsed_args["width-steps"]
    activation_function_strings = parsed_args["activation-functions"]
    batch_size_range = [parse(Int64, s) for s in parsed_args["batch-size-range"]]
    batch_size_steps = parsed_args["batch-size-steps"]
    learning_rate_range = [parse(Float32, s) for s in parsed_args["learning-rate-range"]]
    learning_rate_steps = parsed_args["learning-rate-steps"]
    dropout_rate_range = [parse(Float32, s) for s in parsed_args["dropout-rate-range"]]
    dropout_rate_steps = parsed_args["dropout-rate-steps"]
    n_epochs = parsed_args["n-epochs"]
    loss_function_string = parsed_args["loss"]
    log_training_starts = parsed_args["log-training-starts"]
    log_training_loss = parsed_args["log-training-loss"]
    log_folds = parsed_args["log-folds"]
    n_folds = parsed_args["n-folds"]
    outfile = parsed_args["outfile"]
    use_gpu = parsed_args["gpu"]

    if ~endswith(outfile, ".json")
        throw("Outfile must be a .json file")
    end

    println("Formatting data...")
    x_raw_df, y_df, cellnumber_df = getrawdata_withcellnum(target_directory)

    # we are interested in whether cellnumber is odd or even
    cellnumber_df = cellnumber_df .% 2

    # how to process cellnumber, for now let's make it another dvar
    x_raw_df = hcat(x_raw_df, cellnumber_df)

    # cutting
    println("Cutting Transmission to 60-100 percent...")
    lower::Float32 = 60
    upper::Float32 = 120
    x_raw_df, y_df = applycut(x_raw_df, y_df, "OBJ1", lower, upper; with_numcells=true)

    # decorrelating
    println("Decorrelating...")
    x_df = decorrelatedvars(x_raw_df; with_numcells=true)

    # modify obj5 and obj6
    emits = hcat(y_df[:, "OBJ5"], y_df[:, "OBJ6"]);

    x_scaled_df, _ = minmaxscaledf(x_df)
    y_scaled_df, y_scalers = minmaxscaledf(y_df)

    # need to make sure that column names didn't switch
    println("- names of x_raw_df", names(x_raw_df))
    println("- names of x_scaled_df", names(x_scaled_df))
    @assert names(x_raw_df) == names(x_scaled_df)

    println()
    println("Y-Scalers:")
    println(y_scalers)

    x_train_df, x_test_df, y_train_df, y_test_df = traintestsplit(x_scaled_df, y_scaled_df; read_in=false)

    x_train = Float32.(Matrix(x_train_df))
    x_test = Float32.(Matrix(x_test_df))
    y_train = Float32.(Matrix(y_train_df))
    y_test = Float32.(Matrix(y_test_df))

    # training parameters
    println("Preparing for training...")
    println("- ", x_train[1, :])
    depths = stratifyarchitecturedimension(depth_range[1], depth_range[2], depth_steps; ints_only=true)
    widths = stratifyarchitecturedimension(width_range[1], width_range[2], width_steps; ints_only=true)
    batchsizes = [2^logbs for logbs in stratifyarchitecturedimension(Int(log2(batch_size_range[1])), Int(log2(batch_size_range[2])), batch_size_steps; ints_only=true)]
    learning_rates = [10^loglr for loglr in stratifyarchitecturedimension(Float32(log10(learning_rate_range[1])), Float32(log10(learning_rate_range[2])), learning_rate_steps; ints_only=false)]
    dropout_rates = stratifyarchitecturedimension(dropout_rate_range[1], dropout_rate_range[2], dropout_rate_steps; ints_only=false)
    loss_function = loss_function_string == "mse" ? Flux.mse : Flux.mae

    # instantiating outdata container
    outdata = Vector{Dict}(undef, length(depths)*length(widths)*length(activation_function_strings)*length(batchsizes)*length(learning_rates)*length(dropout_rates))

    # training
    # TODO threading -- looking into polyesther library? 
    println("Beginning training...")
    Threads.@threads for (idx, (width, depth, activation_function_string, batchsize, learning_rate, dropout_rate)) in collect(enumerate(Iterators.product(widths, depths, activation_function_strings, batchsizes, learning_rates, dropout_rates)))
        if log_training_starts
            println("- Training width=$width, depth=$depth, activation=$activation_function_string, batchsize=$batchsize, learning_rate=$learning_rate, dropout_rate=$dropout_rate on thread $(Threads.threadid())")
        end

        activation_function = parseactivationfunctions([activation_function_string])[1]
        optimizer = ADAM(learning_rate)

        model_id = generatemodelid(width, depth, activation_function_string, batchsize, learning_rate, dropout_rate)
        cv_scores = crossvalidate(
            x_train, y_train;
            n_folds=n_folds, width=width, depth=depth, activation_function=activation_function, n_epochs=n_epochs, 
            batchsize=batchsize, optimizer=optimizer, dropout_rate=dropout_rate, loss_function=loss_function, log_training=log_training_loss,
            log_folds=log_folds, model_id=model_id, y_scalers=y_scalers, use_gpu=use_gpu
        )

        println(cv_scores["by_response"]["OBJ5"]["mape_val"])
        println(cv_scores["by_response"]["OBJ6"]["mape_val"])

        # recording results
        outdata_dict = Dict(
            "model_id"=>model_id,
            "configs"=>Dict(
                "n_folds"=>n_folds,
                "width"=>width,
                "depth"=>depth,
                "activation_function"=>activation_function_string,
                "n_epochs"=>n_epochs,
                "batchsize"=>batchsize,
                "learning_rate"=>learning_rate,
                "dropout_rate"=>dropout_rate,
                "optimizer"=>"ADAM",
                "loss_function"=>loss_function_string
            ),
            "results"=>cv_scores
        )

        outdata[idx] = outdata_dict
    end

    open("results/$(stringnow())_" * outfile, "a") do f
        JSON.print(f, outdata, 4)
    end

end
