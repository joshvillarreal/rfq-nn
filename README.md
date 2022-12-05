# rfq-nn
Using deep learning to simulate RFQ-throughgoing beam dynamics

## Julia Installation Instructions
You can download Julia <a href="https://julialang.org/downloads/">here</a>. You will also need to be able to run Julia from the command line. This can be done by adding the following line to your `~/.bashrc`:

```export PATH="$PATH:/path/to/julia-1.8.1/bin"```

Or alternatively, from the command line:

```$ sudo ln -s /Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia```

### Julia versions
In case Julia comes preinstalled on your local machine, a tool that I found helpful for managing Julia versions is <a href="https://github.com/johnnychen94/jill.py">`jill.py`</a>.

## Package requirements
Scripts can be installed by leveraging the Julia `Pkg` module, which is prewritten in `requirements.jl`. Once you've installed Julia, run the following from the root of the directory:

```$ julia requirements.jl```
