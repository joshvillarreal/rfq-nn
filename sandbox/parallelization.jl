using DataFrames, Distributions, Flux, JSON, Random;


function generatedata(mu::Float64=0., sig::Float64=1., n::Int=1000)
    d = Normal(mu, sig)
    x1 = rand(d, n)
    x2 = rand(d, n)
    y = [cos(x1[i] - x2[i]) + x2[i]^3 - 4*x1[i] for i in 1:n]
    return hcat(x1, x2), y
end


function traintestsplit(x, y, train_frac::Float64=0.8)
    data_size = size(x)[1]
    train_size = trunc(Int, train_frac * data_size)

    train_indexes = sample(1:data_size, train_size, replace=false)
    test_indexes = (1:data_size)[(1:data_size) .∉ Ref(train_indexes)]

    x_train = x[train_indexes, :]; x_test = x[test_indexes, :];
    y_train = y[train_indexes, :]; y_test = y[test_indexes, :];

    return x_train, x_test, y_train, y_test
end


function neuralnetwork()
    Chain(
        Dense(2, 4, x->σ.(x)),
        Dense(4, 4, x->σ.(x)),
        Dense(4, 4, x->σ.(x)),
        Dense(4, 4, x->σ.(x)),
        Dense(4, 1)
    )
end


function buildandtrain(
    x_train,
    y_train;
    n_epochs::Int=10000,
    batchsize::Int=64,
    optimizer=ADAM(),
    log_training::Bool=false
)
    # batch data
    data_loader = Flux.Data.DataLoader((x_train', y_train'), batchsize=batchsize, shuffle=true)
    
    # instantiating the model
    m = neuralnetwork()

    # training
    loss(x, y) = Flux.mse(m(x), y)
    training_losses = Float64[]
    epochs = Int64[]

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


function main()
    # data handling
    x, y = generatedata(0., 1., 1000)
    x_train, x_test, y_train, y_test = traintestsplit(x, y)

    # training 4 times
    outdata = Dict("losses"=>[])
    Threads.@threads for i in 1:4
        m, losses = buildandtrain(x_train, y_train; log_training=true)
        push!(outdata["losses"], losses)
    end

    # write histories to file
    open("parallelization_results.json", "w") do f
        JSON.print(f, outdata)
    end
end


main()