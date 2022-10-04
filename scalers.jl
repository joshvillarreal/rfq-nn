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