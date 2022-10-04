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

function traintestsplit(x_df, y_df, train_frac)
    data_size = nrow(x_df)
    train_size = trunc(Int, train_frac * data_size)

    train_indexes = sample(1:data_size, train_size, replace=false)
    test_indexes = (1:data_size)[(1:data_size) .∉ Ref(train_indexes)]

    x_train_df = x_df[train_indexes, :]; x_test_df = x_df[test_indexes, :];
    y_train_df = y_df[train_indexes, :]; y_test_df = y_df[test_indexes, :];

    return x_train_df, x_test_df, y_train_df, y_test_df
end

function stratifyarchitecturedimension(specified_min::Int16, specified_max::Int16, n::Int16=5)
    if specified_min >= specified_max
        return [specified_min]
    else
        stepsize = Int16((specified_max - specified_min) / (n - 1))
        result = [Int16(floor(specified_min + stepsize * (i-1))) for i in 1:n-1]
        return unique(result)
    end
end

function stratifylearningrate(specified_min::Float64, specified_max::Float64, n::Int16=5)
    if specified_min >= specified_max
        return [specified_min]
    else
        logstepsize = Float64((log10(specified_max) - log10(specified_min)) / (n - 1))
        result = [specified_min + 10^(stepsize + (i-1)) for i in 1:n-1]
        return unique(result)
