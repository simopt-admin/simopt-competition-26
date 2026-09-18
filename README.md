# simopt-competition-26

## Run the experiment and plot the results

The commands below are for macOS or Linux. Start with a local checkout of this repository and Git installed; the project fetches SimOpt from GitHub during setup.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Restart your terminal so it picks up uv, then check the installation:

   ```sh
   uv --version
   ```

2. Enter the repository directory, replacing the path below with your checkout location. Install Python 3.13, the version in `.python-version`, and sync the project dependencies:

   ```sh
   cd /path/to/simopt-competition-26
   uv python install 3.13
   uv sync
   ```

   uv creates a `.venv` for the project. You do not need to activate it before using `uv run`. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for details.

3. Run the experiment from the repository directory:

   ```sh
   uv run simopt-competition-26
   ```

   This runs `RandomSearch`, `DemoSolver` from [demo.py](src/simopt_competition_26/demo.py), and `DASTRODF` from [dastrodf.py](src/simopt_competition_26/dastrodf.py) on the tele-operator scheduling problem, performs the post-replications, and plots all three mean, unnormalized progress curves together in `plot.png` in the current directory at 300 dpi. `DemoSolver` searches feasible neighbors by adding or removing one driver at one shift start, using 10 simulation replications per candidate and accepting the first improvement in randomized order. It stops when the budget runs out or a full neighborhood gives no improvement. All three solvers use the same experiment settings and replication budget. Plotting happens automatically at the end of the run. Running the command again replaces `plot.png`.

   The experiment settings are in [the entry point](src/simopt_competition_26/__init__.py): 10 macroreplications, 20 post-replications, and 20 post-replications for the initial and reference solutions. The default optimization budget is 1000 simulation replications, each covering nine simulated days, so the full experiment can take time. A plot file has been saved at `plot.png`.
