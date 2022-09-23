using DataFrames, Distributions, Flux, Random;

# generate toy data
function generatedata(mu::Float64=0., sig::Float64=1., n::Int=1000)
    d = Normal(mu, sig)
    x1 = rand(d, n)
    x2 = rand(d, n)
    y = [cos(x1[i] - x2[i]) + x2[i]^3 - 4*x1[i] for i in 1:n]
    return hcat(x1, x2), y
end


function main()
    x, y = generatedata(0., 1., 10)
    println(x)
    println(y)
end


main()