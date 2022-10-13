using CSV;
using DataFrames;
using Dates;
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

# decorrelating design variables
function getdvarprime(data_to_transform, dynamic_lower_bound, strict_upper_bound)
    (data_to_transform .- dynamic_lower_bound) ./ (strict_upper_bound .- dynamic_lower_bound)
end

function decorrelatedvars(df_raw)
    maxes = Dict(
        "DVAR3"=>160.,
        "DVAR5"=>1.85,
        "DVAR9"=>-25.,
        "DVAR12"=>2.,
        "DVAR13"=>-20.
    )
    etas = Dict(
        "DVAR3"=>10.,
        "DVAR5"=>0.05,
        "DVAR9"=>2.5,
        "DVAR12"=>0.05,
        "DVAR13"=>2.5
    )

    DataFrame(
        "DVAR1"=>df_raw[:, "DVAR1"],
        "DVAR2"=>df_raw[:, "DVAR2"],
        "DVAR3"=>getdvarprime(df_raw[:, "DVAR3"], etas["DVAR3"] .+ df_raw[:, "DVAR2"], maxes["DVAR3"]),
        "DVAR4"=>df_raw[:, "DVAR4"],
        "DVAR5"=>getdvarprime(df_raw[:, "DVAR5"], etas["DVAR5"] .+ df_raw[:, "DVAR4"], maxes["DVAR5"]),
        "DVAR6"=>df_raw[:, "DVAR6"],
        "DVAR7"=>df_raw[:, "DVAR7"],
        "DVAR8"=>df_raw[:, "DVAR8"],
        "DVAR9"=>getdvarprime(df_raw[:, "DVAR9"], etas["DVAR9"] .+ df_raw[:, "DVAR8"], maxes["DVAR9"]),
        "DVAR10"=>df_raw[:, "DVAR10"],
        "DVAR11"=>df_raw[:, "DVAR11"],
        "DVAR12"=>getdvarprime(df_raw[:, "DVAR12"], etas["DVAR12"] .+ df_raw[:, "DVAR5"], maxes["DVAR12"]),
        "DVAR13"=>getdvarprime(df_raw[:, "DVAR13"], etas["DVAR13"] .+ df_raw[:, "DVAR9"], maxes["DVAR13"]),
        "DVAR14"=>df_raw[:, "DVAR14"]
    )
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


# train test split
function traintestsplit(x_df, y_df; train_frac=0.8, read_in=false)
    if !read_in
        data_size = nrow(x_df)
        train_size = trunc(Int, train_frac * data_size)

        train_indexes = sample(1:data_size, train_size, replace=false)
        test_indexes = (1:data_size)[(1:data_size) .∉ Ref(train_indexes)]
    else
        train_index_df = CSV.File("indexes/train_indexes.csv"; header=0) |> DataFrame
        test_index_df = CSV.File("indexes/test_indexes.csv"; header=0) |> DataFrame

        train_indexes = train_index_df[:, "Column1"]
        test_indexes = test_index_df[:, "Column1"]
    end

    x_train_df = x_df[train_indexes, :]; x_test_df = x_df[test_indexes, :];
    y_train_df = y_df[train_indexes, :]; y_test_df = y_df[test_indexes, :];

    return x_train_df, x_test_df, y_train_df, y_test_df
end


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

function stringnow()
    Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
end

function generatemodelid(width::Int, depth::Int)
    stringnow() * "_w=$width" * "_d=$depth"
end

