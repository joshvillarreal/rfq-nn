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
The `Project.toml` and `Manifest.toml` contain all necessary dependencies, and effectively create an environment that can be used to run all scripts in this project. See <a href="https://pkgdocs.julialang.org/v1/toml-files/">here</a>.

To set up this environment, run the following command from the base directory:

```$ julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'```

### Using Jupyter Notebooks

To ensure your environment is properly configured when beginning a Jupyter session, first launch Julia from the base directory with ``julia --project=.``, and then run

```julia
using IJulia
notebook(dir=".")
```

You will likely then have to activate and instantiate the environment from your notebook once the kernel is started:


```julia
using Pkg
Pkg.activate("/path/to/rfq-nn/.")
Pkg.instantiate()
```
