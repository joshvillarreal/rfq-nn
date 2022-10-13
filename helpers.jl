using DataFrames;
using JSON;


tofloat((k,v)) = k => parse(Float64, v)


function readdataentry(sample, key)
    unformatted_data = Dict{String, Any}(sample[2][key])
    return Dict{String, Float64}(Iterators.map(tofloat, pairs(unformatted_data)))
end


function readjsonsfromdirectory(target_directory::String, x_df, y_df)
    # x_df and y_df should be empty
    for file in readdir(target_directory)
        if endswith(file, ".json")
            data_raw = JSON.parsefile("$target_directory/$file")["samples"];

            for sample in data_raw
                features = readdataentry(sample, "dvar")
                responses = readdataentry(sample, "obj")

                push!(x_df, features)
                push!(y_df, responses)
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

function minmaxscaledf(df)
    # return hcat(DataFrame.(colname=>fit_transform(df[!, colname]) for colname in names(df))...)
    scaled_data_dict = Dict(colname=>[] for colname in names(df))
    scalers = Dict(colname=>MinMaxScaler(0., 0.) for colname in names(df))

    for colname in names(df)
        data = df[!, colname]
        scaler = MinMaxScaler(0., 0.)
        fit!(scaler, data)

        scaled_data_dict[colname] = transform(scaler, data)
        scalers[colname] = scaler
    end
    return DataFrame(scaled_data_dict), scalers
end

#=
function minmaxunscaledf(df, scalers_dict)
    unscaled_data_dict = Dict(colname=>[] for colname in names(df))

    for colname in names(df)
        println("Unscaling column $colname")
        scaler = scalers_dict[colname]
        data = df[!, colname]
        unscaled_data = inverse_transform(scaler, data)

        unscaled_data_dict[colname] = unscaled_data
    end

    return hcat(DataFrame.(unscaled_data_dict...))
end =#


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
