using DataFrames;
using JSON;


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
            for (i, sample) in enumerate(data_raw)
                responses = Dict{String, Any}(sample[2]["obj"])
                responses_formatted = Dict{String, Float64}(Iterators.map(tofloat, pairs(responses)))
                push!(y_df, responses_formatted)
                @assert(sample[1] == indexes[i])
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


#=== statistics ===#
# rsquared
function r2score(yvec, ŷvec)
    ymean = mean(yvec)
    numerator = sum((y - ŷ)^2 for (y, ŷ) in zip(yvec, ŷvec))
    denominator = sum((y - ymean)^2 for y in yvec)
    1 - numerator / denominator
end
    
function r2score_multidim(ys, ŷs, multioutput::String="uniformaverage")
    d = size(ys, 2)
    r2score_rawvalues = [r2score(ys[:, i], ŷs[:, i]) for i in 1:d]
    if multioutput == "rawvalues"
        return r2score_rawvalues
    elseif multioutput == "uniformaverage"
        return mean(r2score_rawvalues)
    else
        error("multioutput must be one of \"rawvalues\" or \"uniformaverage\"")
    end
end


# adjusted rsquared
function adjustedr2score(yvec, ŷvec, p::Int)
    M = size(yvec, 1)
    return 1 - (1 - r2score(yvec, ŷvec)) * (M - 1) / (M - p - 1)
end

function adjustedr2score_multidim(ys, ŷs, p::Int, multioutput::String="uniformaverage")
    d = size(ys, 2)
    adjustedr2score_rawvalues = [adjustedr2score(ys[:, i], ŷs[:, i], p) for i in 1:d]
    if multioutput == "rawvalues"
        return adjustedr2score_rawvalues
    elseif multioutput == "uniformaverage"
        return mean(adjustedr2score_rawvalues)
    else
        error("multioutput must be one of \"rawvalues\" or \"uniformaverage\"")
    end
end


# documenting scores
function initscoresdict()
    return Dict(
        "r2score_train"=>[], "r2score_val"=>[],
        "adj_r2score_train"=>[], "adj_r2score_val"=>[],
        "mse_train"=>[], "mse_val"=>[],
        "mae_train"=>[], "mae_val"=>[],
        "training_losses"=>[]
    )
end


function updatescoresdict!(scores_dict, y_train, y_train_preds, y_val, y_val_preds, n_features::Int)
    push!(scores_dict["r2score_train"], r2score(y_train, y_train_preds))
    push!(scores_dict["r2score_val"], r2score(y_val, y_val_preds))
    
    push!(scores_dict["adj_r2score_train"], adjustedr2score(y_train, y_train_preds, n_features))
    push!(scores_dict["adj_r2score_val"], adjustedr2score(y_val, y_val_preds, n_features))

    push!(scores_dict["mse_train"], Flux.mse(y_train_preds, y_train))
    push!(scores_dict["mse_val"], Flux.mse(y_val_preds, y_val))

    push!(scores_dict["mae_train"], Flux.mae(y_train_preds, y_train))
    push!(scores_dict["mae_val"], Flux.mae(y_val_preds, y_val))
end


function updateaggregatescoresdict!(
    scores_dict, y_train, y_train_preds, y_val, y_val_preds, n_features::Int, training_losses
)
    push!(scores_dict["r2score_train"], r2score_multidim(y_train, y_train_preds))
    push!(scores_dict["r2score_val"], r2score_multidim(y_val, y_val_preds))
    
    push!(scores_dict["adj_r2score_train"], adjustedr2score_multidim(y_train, y_train_preds, n_features))
    push!(scores_dict["adj_r2score_val"], adjustedr2score_multidim(y_val, y_val_preds, n_features))

    push!(scores_dict["mse_train"], Flux.mse(y_train_preds, y_train))
    push!(scores_dict["mse_val"], Flux.mse(y_val_preds, y_val))

    push!(scores_dict["mae_train"], Flux.mae(y_train_preds, y_train))
    push!(scores_dict["mae_val"], Flux.mae(y_val_preds, y_val))

    push!(scores_dict["training_losses"], training_losses)
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