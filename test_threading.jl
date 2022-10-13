function first_loop()
    Threads.@sync begin
        Threads.@threads for i in 1:10
            println("i=$i, thread $(Threads.threadid())")
        end
    end
end

function second_loop()
    Threads.@sync begin
        Threads.@threads for i in 1:10
            println("i=$i, thread $(Threads.threadid())")
        end
    end
end

function main()
    println("first loop")
    @time first_loop()
    println("second loop")
    @time second_loop()
end

main()