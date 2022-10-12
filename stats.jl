using Flux
using Statistics


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
function adjustedr2score(yvec, ŷvec, p::Int16)
    M = size(yvec, 1)
    return 1 - (1 - r2score(yvec, ŷvec)) * (M - 1) / (M - p - 1)
end

function adjustedr2score_multidim(ys, ŷs, p::Int16, multioutput::String="uniformaverage")
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

# mape
function mape(yvec, ŷvec)
    return Statistics.mean((broadcast(abs, ŷvec-yvec) ./ broadcast(abs, yvec)))
end


# documenting scores
function initscoresdict(n_folds; include_losses=false)
    scores_dict = Dict{String, Any}(
        "r2score_train"=>Vector{Float32}(undef, n_folds),
        "r2score_val"=>Vector{Float32}(undef, n_folds),
        "adj_r2score_train"=>Vector{Float32}(undef, n_folds),
        "adj_r2score_val"=>Vector{Float32}(undef, n_folds),
        "mse_train"=>Vector{Float32}(undef, n_folds),
        "mse_val"=>Vector{Float32}(undef, n_folds),
        "mae_train"=>Vector{Float32}(undef, n_folds),
        "mae_val"=>Vector{Float32}(undef, n_folds),
    )

    if include_losses
        scores_dict["training_losses"] = Vector{Vector{Float32}}(undef, n_folds)
    end

    return scores_dict
end


function updatescoresdict!(scores_dict, fold_id, y_train, y_train_preds, y_val, y_val_preds, n_features::Int16, training_losses=nothing)
    scores_dict["r2score_train"][fold_id] = r2score(y_train, y_train_preds)
    scores_dict["r2score_val"][fold_id] = r2score(y_val, y_val_preds)

    scores_dict["adj_r2score_train"][fold_id] = adjustedr2score(y_train, y_train_preds, n_features)
    scores_dict["adj_r2score_val"][fold_id] = adjustedr2score(y_val, y_val_preds, n_features)

    scores_dict["mse_train"][fold_id] = Flux.mse(y_train_preds, y_train)
    scores_dict["mse_val"][fold_id] = Flux.mse(y_val_preds, y_val)

    scores_dict["mae_train"][fold_id] = Flux.mae(y_train_preds, y_train)
    scores_dict["mae_val"][fold_id] = Flux.mae(y_val_preds, y_val)

    if training_losses != nothing
        scores_dict["training_losses"][fold_id] = training_losses
    end

    return scores_dict
end