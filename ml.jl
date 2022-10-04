using ArgParse
using DataFrames
using Flux
import JSON
using Parameters: @with_kw

include("helpers.jl")
include("stats.jl")

# TODO -- cleaner hyperparam object
@with_kw struct HyperParams
    width::Int16
    depth::Int16
    n_epochs::Int32
    batchsize::Int
    learning_rate::Float64
    loss_function::Function
    activation_function::Function
end

function neuralnetwork(x_dimension::Int16, y_dimension::Int16, width::Int16, depth::Int16)
    Chain(
        Dense(x_dimension, width, x->σ.(x)),
        (Dense(width, width, x->σ.(x)) for _ in 1:depth)...,
        Dense(width, y_dimension)
    )
end

function buildandtrain(
    x_train,
    y_train;
    width::Int16,
    depth::Int16,
    n_epochs::Int32=100,
    batchsize::Int=1024,
    learning_rate::Float64=0.0001
    loss_function=Flux.mse,
    log_training::Bool=false
)
    # batch data
    data_loader = Flux.Data.DataLoader((x_train', y_train'), batchsize=batchsize, shuffle=true)
    
    # instantiating the model
    optimizer = Adam(learning_rate)
    m = neuralnetwork(Int16(size(x_train)[2]), Int16(size(y_train)[2]), width, depth)

    # training
    loss(x, y) = loss_function(m(x), y)
    training_losses = Float32[]
    epochs = Int32[]

    for epoch in 1:n_epochs
        Flux.train!(loss, Flux.params(m), data_loader, optimizer)
        push!(epochs, epoch)

        l = 0.
        for d in data_loader
            l += loss(d...)
        end

        if log_training
            println("epoch $epoch, loss=$l")
        end

        push!(training_losses, l)
    end

    return m, training_losses
end


function crossvalidate(
    x_train,
    y_train;
    n_folds::Int16=5,
    width::Int16,
    depth::Int16,
    n_epochs::Int32=100,
    batchsize::Int=1024,
    learning_rate::Float64=0.001,
    loss_function=Flux.mse,
    log_training::Bool=false
)
    scores_total = initscoresdict(n_folds; include_losses=true)
    scores_by_response = Dict("OBJ$i"=>initscoresdict(n_folds) for i in 1:6)

    folds = kfolds(size(x_train)[1]; k=n_folds)

    for (fold_id, (train_temp_idxs, val_temp_idxs)) in collect(enumerate(Iterators.zip(folds...)))
        if log_folds
            println("fold $fold_id of w,d=$width,$depth beginning")
        end

        # select training and validation sets
        x_train_temp, x_val_temp = x_train[train_temp_idxs, :], x_train[val_temp_idxs, :]
        y_train_temp, y_val_temp = y_train[train_temp_idxs, :], y_train[val_temp_idxs, :]

        # train model
        m, training_losses = buildandtrain(
            x_train_temp, y_train_temp;
            width=width, depth=depth, n_epochs=n_epochs, batchsize=batchsize, learning_rate=learning_rate,
            loss_function=loss_function, log_training=log_training
        )

        # gather predictions
        y_train_temp_preds = m(x_train_temp')'; y_val_temp_preds = m(x_val_temp')'

        # update aggregate scores
        n_features = size(x_train_temp, 2)
        updatescoresdict!(
            scores_total, fold_id, y_train_temp, y_train_temp_preds, y_val_temp, y_val_temp_preds,
            n_features, training_losses
        )

        # update scores by objective
        for j in 1:6
            updatescoresdict!(
                scores_by_response["OBJ$j"], fold_id, y_train_temp[:, j], y_train_temp_preds[:, j],
                y_val_temp[:, j], y_val_temp_preds[:, j], n_features)
        end
    end
    return Dict("total"=>scores_total, "by_response"=>scores_by_response)
end