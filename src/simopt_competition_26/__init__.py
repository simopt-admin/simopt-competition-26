from simopt.analysis.progress_curve import ProgressCurve
from simopt.experiment.api import SimulationConfig, run_experiment
from simopt.experiment.single import ProblemSolver
from simopt.solvers.randomsearch import RandomSearch

from .teleops import TeleopsProblem


def main() -> None:
    problem = TeleopsProblem()
    n_mreps = 10
    n_preps = 20
    n_preps_x0_xstar = 20

    experiments = [ProblemSolver(solver=RandomSearch(), problem=problem, create_pickle=False)]
    simulation_config = SimulationConfig(
        n_mreps=n_mreps, n_preps=n_preps, n_preps_x0_xstar=n_preps_x0_xstar
    )

    analysis_inputs = run_experiment(experiments, simulation_config)
    fig, _ = ProgressCurve(agg="mean", normalize=False).plot(analysis_inputs, experiments)
    fig.savefig("plot.png", dpi=300)
