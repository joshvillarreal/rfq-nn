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
## Results
In the `/results/` directory, the following files are relevant to the final write-up of this analysis:
* `2023-06-17_05-13-31_results.json`, `2023-06-17_05-52-33_results.json`, `2023-06-17_10-22-31_results.json`: Collectively the full hyperparameter scan on all available training data
* `2023-06-16_22-09-00_results.json`: Full hyperparameter scan on data having transmission restricted to $\geq 60\%$
* `2023-06-18_18-09-23_results.json`, `2023-06-18_18-24-20_results.json`, `2023-06-19_00-25-36_results.json`, `2023-06-19_15-04-48_results.json`: Collectively the scan on learning rates and dropout rates for the width 6, depth 100 neural network trained on data having transmission restricted to $\geq 60\%$
* `2023-06-19_02-49-35_results.json`, `2023-06-19_08-31-59_results.json`, `2023-06-19_11-07-34_results.json`, `2023-06-19_19-38-21_results.json`: Collectively the scan on learning rates and dropout rates for the width 7, depth 100 neural network trained on data having transmission restricted to $\geq 60\%$.
