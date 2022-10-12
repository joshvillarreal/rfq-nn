using DataFrames;

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
endM


# dynamic minmax scaler (used for decorrelating)
mutable struct DynamicMinMaxScaler
    dynamic_min
    dynamic_max
end

function dynamicminmax_transform(scaler, data)
    2 .* (data .- scaler.dynamic_min) ./ (scaler.dynamic_max .- scaler.dynamic_min) .- 1
end

function dynamicminmax_inverse_transform(scaler, data_scaled)
    0.5 .* (scaler.dynamic_max .- scaler.dynamic_min) .* (data_scaled .+ 1) .+ scaler.dynamic_min
end

function dynamicminmax_fit_transform(data, dynamic_min, dynamic_max)
    scaler = DynamicMinMaxScaler(dynamic_min, dynamic_max)
    dynamicminmax_transform(scaler, data)
end


# dynamicminmax scale data
function dynamicminmaxscaledf(df)
    mins_and_maxes = Dict(
        "DVAR1"=>(8.5, 12.0),
        "DVAR2"=>(5., 140.),
        "DVAR3"=>(df[!, "DVAR2"] .+ 10., 160.),
        "DVAR4"=>(1.005, 1.7),
        "DVAR5"=>(df[!, "DVAR4"] .+ 0.05, 1.85),
        "DVAR6"=>(1., 500.),
        "DVAR7"=>(1., 500.),
        "DVAR8"=>(-89.95, -30.),
        "DVAR9"=>(df[!, "DVAR8"] .+ 2.5, -25.),
        "DVAR10"=>(1., 500.),
        "DVAR11"=>(1., 500.),
        "DVAR12"=>(df[!, "DVAR5"] .+ 0.05, 2.0),
        "DVAR13"=>(df[!, "DVAR9"] .+ 2.5, -20.),
        "DVAR14"=>(0.055, 0.075),
    )

    return hcat(
        DataFrame.(
            colname=>dynamicminmax_fit_transform(
                df[!, colname],
                mins_and_maxes[colname]...
            )
            for colname in names(df)
        )...
    )
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

function standardscaledf(df)
    return hcat(DataFrame.(colname=>standard_fit_transform(df[!, colname]) for colname in names(df))...)
end