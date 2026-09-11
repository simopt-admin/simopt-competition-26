"""Create your solver here."""

from typing import Annotated, ClassVar

from pydantic import Field
from simopt.base import (
    ConstraintType,
    Context,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)


class DemoSolverConfig(SolverConfig):
    sample_size: Annotated[int, Field(default=10, gt=0, description="replications per point")]


class DemoSolver(Solver):
    name: str = "DemoSolver"
    config_class: ClassVar[type[SolverConfig]] = DemoSolverConfig
    class_name_abbr: ClassVar[str] = "DemoSolver"
    class_name: ClassVar[str] = "Demo Local Search"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: Problem, ctx: Context) -> None:
        sample_size = self.factors["sample_size"]
        incumbent = ctx.evaluate(tuple(problem.factors["initial_solution"]), 0)
        ctx.log(incumbent)
        if not ctx.budget.remaining:
            return
        incumbent = ctx.evaluate(incumbent, min(sample_size, ctx.budget.remaining))

        while ctx.budget.remaining:
            moves = [(i, step) for i in range(problem.dim) for step in (-1, 1)]
            self.rng_list[1].shuffle(moves)
            for coordinate, step in moves:
                neighbor = list(incumbent.x)
                neighbor[coordinate] += step
                x = tuple(neighbor)
                if not problem.check_deterministic_constraints(x):
                    continue
                if not ctx.budget.remaining:
                    return
                candidate = ctx.evaluate(x, min(sample_size, ctx.budget.remaining))
                # Context converts objectives to minimization, including TELEOPS.
                if candidate.objectives_mean[0] < incumbent.objectives_mean[0]:
                    incumbent = candidate
                    ctx.log(incumbent)
                    break
            else:
                return
