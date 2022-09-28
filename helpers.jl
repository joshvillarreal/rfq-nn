using DataFrames;
using JSON;

include("stats.jl");


tofloat((k,v)) = k => parse(Float64, v)


function readjsonsfromdirectory(target_directory::String, x_df, y_df)
    #= 
    Updates two empty DataFrames, x_df and y_df, populated with data from some target directory
    =#
    for jsonfile in readdir(target_directory)
        if endswith(jsonfile, ".json")
            data_raw = JSON.parsefile("$target_directory/$jsonfile")["samples"];

            indexes = Vector{String}()

            # build x
            for sample in data_raw
                features = Dict{String, Any}(sample[2]["dvar"])
                features_formatted = Dict{String, Float64}(Iterators.map(tofloat, pairs(features)))
                push!(x_df, features_formatted)
                push!(indexes, sample[1])
            end

            # build y
            for sample in enumerate(data_raw)
                responses = Dict{String, Any}(sample[2]["obj"])
                responses_formatted = Dict{String, Float64}(Iterators.map(tofloat, pairs(responses)))
                push!(y_df, responses_formatted)
            end
        end
    end
    
    x_df, y_df
end


# scaler
mutable struct MinMaxScaler
    data_min::Float64
    data_max::Float64
end

function fit!(scaler, data)
    scaler.data_min = minimum(data); scaler.data_max = maximum(data)
end

function transform(scaler, data)
    [2*(d - scaler.data_min)/(scaler.data_max - scaler.data_min) - 1 for d in data]
end

function inverse_transform(scaler, data_scaled)
    [0.5*(scaler.data_max - scaler.data_min)*(d_s + 1) + scaler.data_min for d_s in data_scaled]
end

function fit_transform(data)
    scaler = MinMaxScaler(0., 0.)
    fit!(scaler, data)
    transform(scaler, data)
end


# minmax scale data
function minmaxscaledf(df)
    return hcat(DataFrame.(colname=>fit_transform(df[!, colname]) for colname in names(df))...)
end


# train test split
function traintestsplit(x_df, y_df, train_frac)
    data_size = nrow(x_df)
    train_size = trunc(Int, train_frac * data_size)

    train_indexes = sample(1:data_size, train_size, replace=false)
    test_indexes = (1:data_size)[(1:data_size) .∉ Ref(train_indexes)]

    x_train_df = x_df[train_indexes, :]; x_test_df = x_df[test_indexes, :];
    y_train_df = y_df[train_indexes, :]; y_test_df = y_df[test_indexes, :];

    return x_train_df, x_test_df, y_train_df, y_test_df
end


# documenting scores
function initscoresdict(n_folds)
    scores_dict = Dict(
        "r2score_train"=>Vector{Float64}(undef, n_folds),
        "r2score_val"=>Vector{Float64}(undef, n_folds),
        "adj_r2score_train"=>Vector{Float64}(undef, n_folds),
        "adj_r2score_val"=>Vector{Float64}(undef, n_folds),
        "mse_train"=>Vector{Float64}(undef, n_folds),
        "mse_val"=>Vector{Float64}(undef, n_folds),
        "mae_train"=>Vector{Float64}(undef, n_folds),
        "mae_val"=>Vector{Float64}(undef, n_folds),
    )
end


function updatescoresdict!(scores_dict, fold_id, y_train, y_train_preds, y_val, y_val_preds, n_features::Int, training_losses=nothing)
    scores_dict["r2score_train"][fold_id] = r2score(y_train, y_train_preds)
    scores_dict["r2score_val"][fold_id] = r2score(y_val, y_val_preds)

    scores_dict["adj_r2score_train"][fold_id] = adjustedr2score(y_train, y_train_preds, n_features)
    scores_dict["adj_r2score_val"][fold_id] = adjustedr2score(y_val, y_val_preds, n_features)

    scores_dict["mse_train"][fold_id] = Flux.mse(y_train_preds, y_train)
    scores_dict["mse_val"][fold_id] = Flux.mse(y_val_preds, y_val)

    scores_dict["mae_train"][fold_id] = Flux.mae(y_train_preds, y_train)
    scores_dict["mae_val"][fold_id] = Flux.mae(y_val_preds, y_val)

    if training_losses === Nothing()
        return scores_dict
    else
        scores_dict["training_losses"] = training_losses
        return scores_dict
    end
end


#= if storage becomes an issue
function percentchangeinlastn(histories; n::Int=0)
    history_length = size(histories)[1]
    initial_index = history_length - n
    initial = histories[initial_index]; final = histories[history_length]
    return (final - initial) / initial
end
=#


function stratifyarchitecturedimension(specified_min::Int, specified_max::Int, n::Int=5)
    if specified_min >= specified_max
        return [specified_min]
    else
        stepsize = (specified_max - specified_min) / (n - 1)
        result = []
        for i in 1:n-1
            push!(result, Int(floor(specified_min + stepsize * (i-1))))
        end
        push!(result, specified_max)
        return unique(result)
    end
end