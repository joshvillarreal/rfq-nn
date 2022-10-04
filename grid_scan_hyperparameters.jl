using ArgParse
using DataFrames
using Flux
import JSON
using Parameters: @with_kw

include("helpers.jl")
include("stats.jl")
include("scalers.jl")
include("ml.jl")


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
            default = ["10", "60"]
        "--width-range"
            help = "Range of widths to search over"
            nargs = '*'
            default = ["2", "7"]
        "--depth-steps"
            help = "Number of depths to search over in specified depth-range"
            arg_type = Int16
            default = Int16(5)
        "--width-steps"
            help = "Number of widths to search over in specified width-range"
            arg_type = Int16
            default = Int16(5)
        "--learning-rate-range"
            help="Range of learning rates to use for optimizer"
            nargs= '*'
            default = ["0.000001", "0.0001"]
        "--learning-rate-steps"
            help = "Number of learning rates to search over in specified learning-rate-range"
            arg_type = Int16
            default = Int16(2)
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
            action = :store_true
        "--log-folds"
            help = "Print when each fold of CV beings"
            action = :store_true
        "--log-training-loss"
            help = "Print training loss per epoch"
            action = :store_true
        "--outfile"
            help = "Filename to record results"
            arg_type = String
            default = "results.json"
        "--toy" # TODO -- implement toy example to it's easy to test things out
            help = "Use small toy example to try things out"
            action = :store_true
    end

    return parse_args(s)
end

function checkargs(parsed_args)
    if ~endswith(parsed_args.outfile, ".json")
        throw("Outfile must be a .json file")
    end
end

function main()
    # gather arguments
    parsed_args = parse_commandline()
    target_directory = parsed_args["data-directory"]
    depth_range = [parse(Int16, s) for s in parsed_args["depth-range"]]
    width_range = [parse(Int16, s) for s in parsed_args["width-range"]]
    depth_steps = parsed_args["depth-steps"]
    width_steps = parsed_args["width-steps"]
    learning_rate_range = [parse(Float64, s) for s in parse_args["learning-rate-range"]]
    learning_rate_steps = parsed_args["learning-rate-steps"]
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
    learning_rates = 
    batchsize = 1024
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