using DataFrames;
using JSON;
using Statistics;


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


tofloat((k,v)) = k => parse(Float32, v)


function readdataentry(sample, key)
    unformatted_data = Dict{String, Any}(sample[2][key])
    return Dict{String, Float32}(Iterators.map(tofloat, pairs(unformatted_data)))
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


# min max scaler
mutable struct MinMaxScaler
    data_min::Float32
    data_max::Float32
end

function minmax_fit!(scaler, data)
    scaler.data_min = minimum(data); scaler.data_max = maximum(data)
end

function minmax_transform(scaler, data)
    [2*(d - scaler.data_min)/(scaler.data_max - scaler.data_min) - 1 for d in data]
end

function minmax_inverse_transform(scaler, data_scaled)
    [0.5*(scaler.data_max - scaler.data_min)*(d_s + 1) + scaler.data_min for d_s in data_scaled]
end

function minmax_fit_transform(data)
    scaler = MinMaxScaler(0., 0.)
    minmax_fit!(scaler, data)
    minmax_transform(scaler, data)
end


# minmax scale data
function minmaxscaledf(df)
    return hcat(DataFrame.(colname=>minmax_fit_transform(df[!, colname]) for colname in names(df))...)
end


# standard scaler
mutable struct StandardScaler
    mean::Float32
    std::Float32
end

function standard_fit!(scaler, data)
    scaler.mean = Statistics.mean(data); scaler.std = Statistics.std(data)
end

function standard_transform(scaler, data)
    [(d - scaler.mean) / scaler.std for d in data]
end

function standard_inverse_transform(scaler, data_scaled)
    [(scaler.std * d_s + scaler.mean) for d_s in data_scaled]
end

function standard_fit_transform(data)
    scaler = StandardScaler(0., 1.)
    fit!(scaler, data)
    transform(scaler, data)
end


# standard scale data
function standardscaledf(df)
    return hcat(DataFrame.(colname=>standard_fit_transform(df[!, colname]) for colname in names(df))...)
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


#= if storage becomes an issue
function percentchangeinlastn(histories; n::Int=0)
    history_length = size(histories)[1]
    initial_index = history_length - n
    initial = histories[initial_index]; final = histories[history_length]
    return (final - initial) / initial
end
=#


function stratifyarchitecturedimension(specified_min::Int16, specified_max::Int16, n::Int16=5)
    if specified_min >= specified_max
        return [specified_min]
    else
        stepsize = Int16((specified_max - specified_min) / (n - 1))
        result = []
        for i in 1:n-1
            push!(result, Int16(floor(specified_min + stepsize * (i-1))))
        end
        push!(result, specified_max)
        return unique(result)
    end
end
