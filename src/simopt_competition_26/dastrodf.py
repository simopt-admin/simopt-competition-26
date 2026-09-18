"""Discrete trust-region AstroDF-style solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import Field, model_validator
from simopt.base import (
    ConstraintType,
    Context,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)


@dataclass
class _LocalModel:
    center: np.ndarray
    radius: float
    intercept: float
    linear: np.ndarray
    quadratic: np.ndarray

    def predict(self, x: tuple[int, ...]) -> float:
        delta = (np.array(x, dtype=float) - self.center) / self.radius
        return float(self.intercept + delta @ self.linear + (delta * delta) @ self.quadratic)

    def coordinate_contribution(self, idx: int, step: int) -> float:
        scaled = step / self.radius
        return float(self.linear[idx] * scaled + self.quadratic[idx] * scaled * scaled)


class DASTRODFConfig(SolverConfig):
    """Configuration for the discrete AstroDF-style solver."""

    eta_1: Annotated[
        float, Field(default=0.1, ge=0.0, description="threshold for a successful iteration")
    ]
    eta_2: Annotated[
        float, Field(default=0.75, le=1.0, description="threshold for a very successful iteration")
    ]
    gamma_1: Annotated[
        float, Field(default=2.0, gt=1.0, description="trust-region radius increase factor")
    ]
    gamma_2: Annotated[
        float, Field(default=0.5, gt=0.0, lt=1.0, description="trust-region radius decrease factor")
    ]
    pilot_run: Annotated[
        int, Field(default=2, gt=0, description="initial sample size for a new solution")
    ]
    search_batch_size: Annotated[
        int, Field(default=2, gt=0, description="number of trial solutions evaluated per iteration")
    ]
    candidate_pool_size: Annotated[
        int, Field(default=24, gt=0, description="number of model-scored trial solutions generated")
    ]
    model_point_limit: Annotated[
        int,
        Field(
            default=64, gt=1, description="maximum number of cached points used in the local model"
        ),
    ]
    ridge: Annotated[
        float,
        Field(default=1e-6, gt=0.0, description="ridge penalty for the local regression model"),
    ]
    weighted_local_model: Annotated[
        bool,
        Field(
            default=True,
            description="use distance- and replication-weighted regression in local-model fitting",
        ),
    ]
    shake_after: Annotated[
        int,
        Field(
            default=4, gt=0, description="failed iterations before enabling a larger search step"
        ),
    ]
    max_compare_reps: Annotated[
        int,
        Field(default=6, gt=1, description="maximum replications used in ambiguous comparisons"),
    ]
    unbounded_search_radius: Annotated[
        int,
        Field(
            default=5,
            gt=0,
            description="finite coordinate search radius used for unbounded variables",
        ),
    ]

    @model_validator(mode="after")
    def _validate_eta_order(self) -> Self:
        if self.eta_2 <= self.eta_1:
            raise ValueError("eta_2 must be greater than eta_1.")
        return self


class DASTRODF(Solver):
    """Discrete trust-region solver adapted from the AstroDF framework."""

    name: str = "DASTRODF"
    config_class: ClassVar[type[SolverConfig]] = DASTRODFConfig
    class_name_abbr: ClassVar[str] = "DASTRODF"
    class_name: ClassVar[str] = "Discrete AstroDF"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_needed: ClassVar[bool] = False

    def _initialize(self, problem: Problem, ctx: Context) -> None:
        self.problem = problem
        self.ctx = ctx
        initial_x = np.array(problem.factors["initial_solution"], dtype=int)
        search_radius = self.factors["unbounded_search_radius"]
        lower_bounds = np.array(problem.lower_bounds, dtype=float)
        upper_bounds = np.array(problem.upper_bounds, dtype=float)
        self.lower_bounds = np.where(
            np.isneginf(lower_bounds), initial_x - search_radius, lower_bounds
        ).astype(int)
        self.upper_bounds = np.where(
            np.isposinf(upper_bounds), initial_x + search_radius, upper_bounds
        ).astype(int)
        ranges = self.upper_bounds - self.lower_bounds
        self.max_radius = max(1, int(np.max(ranges)))
        self.radius = self.max_radius
        self.failed_iterations = 0
        self.iteration_count = 0
        self.cache: dict[tuple[int, ...], Solution] = {}
        self.find_next_soln_rng = self.rng_list[1]
        self.search_rng = self.rng_list[2]

    def _transformed_objective(self, solution: Solution) -> float:
        if solution.n_reps == 0:
            return float("inf")
        # Context.evaluate normalizes every objective to minimization.
        value = solution.objectives_mean
        return float(np.asarray(value).reshape(-1)[0])

    def _objective_variance(self, solution: Solution) -> float:
        variance = np.asarray(solution.objectives_var).reshape(-1)
        if variance.size == 0:
            return 0.0
        return float(variance[0])

    def _ensure_replications(self, solution: Solution, target: int) -> None:
        if solution.n_reps >= target or self.ctx.budget.remaining <= 0:
            return
        need = min(target - solution.n_reps, self.ctx.budget.remaining)
        if need <= 0:
            return
        self.ctx.evaluate(solution, need)

    def _get_solution(self, x: tuple[int, ...]) -> Solution:
        if x not in self.cache:
            self.cache[x] = self.ctx.evaluate(x, 0)
        return self.cache[x]

    def _distance(self, x: tuple[int, ...], y: tuple[int, ...]) -> int:
        return int(np.max(np.abs(np.array(x, dtype=int) - np.array(y, dtype=int))))

    def _fit_local_model(self) -> _LocalModel:
        center = np.array(self.incumbent_solution.x, dtype=float)
        radius = float(max(self.radius, 1))
        cached_points = sorted(
            self.cache.values(),
            key=lambda solution: (
                self._distance(solution.x, self.incumbent_solution.x),
                -solution.n_reps,
            ),
        )
        cached_points = cached_points[: self.factors["model_point_limit"]]

        if not cached_points:
            dim = self.problem.dim
            return _LocalModel(center, radius, 0.0, np.zeros(dim), np.zeros(dim))

        x_rows = []
        y_vals = []
        for solution in cached_points:
            delta = (np.array(solution.x, dtype=float) - center) / radius
            x_rows.append(np.concatenate(([1.0], delta, delta * delta)))
            y_vals.append(self._transformed_objective(solution))

        x_mat = np.array(x_rows)
        y_vec = np.array(y_vals)
        if self.factors["weighted_local_model"]:
            weights = []
            for solution in cached_points:
                dist = self._distance(solution.x, self.incumbent_solution.x)
                weights.append(np.sqrt(max(solution.n_reps, 1)) / (1.0 + dist / radius))
            w_vec = np.array(weights)
        else:
            w_vec = np.ones(len(cached_points))
        wx = x_mat * w_vec[:, None]
        wy = y_vec * w_vec

        gram = wx.T @ wx
        ridge = self.factors["ridge"]
        gram[1:, 1:] += ridge * np.identity(gram.shape[0] - 1)
        rhs = wx.T @ wy

        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs

        dim = self.problem.dim
        intercept = float(beta[0])
        linear = beta[1 : dim + 1]
        quadratic = beta[dim + 1 :]
        return _LocalModel(center, radius, intercept, linear, quadratic)

    def _allowed_steps(self, idx: int, radius: int, base_x: tuple[int, ...]) -> tuple[int, int]:
        lo = max(int(self.lower_bounds[idx] - base_x[idx]), -radius)
        hi = min(int(self.upper_bounds[idx] - base_x[idx]), radius)
        return lo, hi

    def _best_coordinate_steps(self, model: _LocalModel) -> tuple[np.ndarray, np.ndarray]:
        best_steps = np.zeros(self.problem.dim, dtype=int)
        scores = np.zeros(self.problem.dim)
        center = self.incumbent_solution.x

        for idx in range(self.problem.dim):
            lo, hi = self._allowed_steps(idx, self.radius, center)
            best_contribution = 0.0
            best_step = 0
            for step in range(lo, hi + 1):
                if step == 0:
                    continue
                contribution = model.coordinate_contribution(idx, step)
                if contribution < best_contribution:
                    best_contribution = contribution
                    best_step = step
            best_steps[idx] = best_step
            scores[idx] = max(0.0, -best_contribution)

        return best_steps, scores

    def _perturb_probability(self, radius: int) -> float:
        base = min(20.0 / max(self.problem.dim, 1), 1.0)
        scaled = base * (radius / self.max_radius)
        return max(1.0 / max(self.problem.dim, 1), min(1.0, scaled))

    def _sample_indices(self, radius: int) -> list[int]:
        probability = self._perturb_probability(radius)
        indices = [idx for idx in range(self.problem.dim) if self.search_rng.random() < probability]
        if indices:
            return indices
        return [self.search_rng.randint(0, self.problem.dim - 1)]

    def _sample_step(
        self, idx: int, radius: int, base_x: tuple[int, ...], preferred_step: int
    ) -> int:
        lo, hi = self._allowed_steps(idx, radius, base_x)
        if lo == 0 and hi == 0:
            return 0
        if preferred_step != 0 and lo <= preferred_step <= hi and self.search_rng.random() < 0.65:
            return preferred_step

        if preferred_step > 0 and hi > 0 and self.search_rng.random() < 0.5:
            return self.search_rng.randint(1, hi)
        if preferred_step < 0 and lo < 0 and self.search_rng.random() < 0.5:
            return self.search_rng.randint(lo, -1)

        while True:
            step = self.search_rng.randint(lo, hi)
            if step != 0:
                return step

    def _repair_candidate(self, x: list[int]) -> tuple[int, ...]:
        candidate = tuple(
            int(min(max(value, int(self.lower_bounds[idx])), int(self.upper_bounds[idx])))
            for idx, value in enumerate(x)
        )
        if self.problem.check_deterministic_constraints(candidate):
            return candidate

        incumbent = list(self.incumbent_solution.x)
        order = list(range(self.problem.dim))
        for idx in range(len(order) - 1, 0, -1):
            swap_idx = self.search_rng.randint(0, idx)
            order[idx], order[swap_idx] = order[swap_idx], order[idx]

        repaired = list(candidate)
        for idx in order:
            repaired[idx] = incumbent[idx]
            trial = tuple(repaired)
            if self.problem.check_deterministic_constraints(trial):
                return trial

        return self.problem.get_random_solution(self.find_next_soln_rng)

    def _build_greedy_candidate(
        self, best_steps: np.ndarray, scores: np.ndarray
    ) -> tuple[int, ...] | None:
        order = np.argsort(-scores)
        max_changes = max(1, round(self._perturb_probability(self.radius) * self.problem.dim))
        candidate = list(self.incumbent_solution.x)
        n_changes = 0

        for idx in order:
            if scores[idx] <= 0 and n_changes > 0:
                break
            step = int(best_steps[idx])
            if step == 0:
                continue
            candidate[idx] += step
            n_changes += 1
            if n_changes >= max_changes:
                break

        if n_changes == 0:
            return None
        return self._repair_candidate(candidate)

    def _build_random_candidate(
        self, best_steps: np.ndarray, radius: int, force_global: bool = False
    ) -> tuple[int, ...]:
        if force_global:
            return self.problem.get_random_solution(self.find_next_soln_rng)

        candidate = list(self.incumbent_solution.x)
        changed = False
        for idx in self._sample_indices(radius):
            step = self._sample_step(idx, radius, self.incumbent_solution.x, int(best_steps[idx]))
            if step == 0:
                continue
            candidate[idx] += step
            changed = True

        if not changed:
            idx = self.search_rng.randint(0, self.problem.dim - 1)
            step = self._sample_step(idx, radius, self.incumbent_solution.x, int(best_steps[idx]))
            candidate[idx] += step

        return self._repair_candidate(candidate)

    def _sample_distinct_global_candidate(
        self, seen: set[tuple[int, ...]], max_attempts: int = 32
    ) -> tuple[int, ...] | None:
        for _ in range(max_attempts):
            candidate = self.problem.get_random_solution(self.find_next_soln_rng)
            if candidate not in seen and self.problem.check_deterministic_constraints(candidate):
                return candidate
        return None

    def _generate_candidates(self, model: _LocalModel) -> list[tuple[int, ...]]:
        best_steps, scores = self._best_coordinate_steps(model)
        candidates = []
        seen = {self.incumbent_solution.x}

        def add_candidate(candidate: tuple[int, ...] | None) -> None:
            if candidate is None or candidate in seen:
                return
            if not self.problem.check_deterministic_constraints(candidate):
                return
            seen.add(candidate)
            candidates.append(candidate)

        if self.iteration_count <= 2 or self.failed_iterations >= self.factors["shake_after"]:
            add_candidate(
                self._build_random_candidate(best_steps, self.max_radius, force_global=True)
            )

        add_candidate(self._build_greedy_candidate(best_steps, scores))

        for idx in np.argsort(-scores)[: max(4, self.factors["search_batch_size"] * 2)]:
            if scores[idx] <= 0:
                break
            candidate = list(self.incumbent_solution.x)
            candidate[idx] += int(best_steps[idx])
            add_candidate(self._repair_candidate(candidate))

        if self.failed_iterations >= self.factors["shake_after"]:
            shake_radius = min(self.max_radius, max(self.radius + 1, ceil(1.5 * self.radius)))
            add_candidate(self._build_random_candidate(best_steps, shake_radius))

        attempts = 0
        max_attempts = max(self.factors["candidate_pool_size"] * 8, 32)
        while len(candidates) < self.factors["candidate_pool_size"] and attempts < max_attempts:
            attempts += 1
            add_candidate(self._build_random_candidate(best_steps, self.radius))

        if not candidates:
            add_candidate(self._sample_distinct_global_candidate(seen))

        return candidates

    def _select_trial_solutions(self, model: _LocalModel) -> list[Solution]:
        candidates = self._generate_candidates(model)
        scored_candidates = sorted(candidates, key=model.predict)
        search_batch_size = self.factors["search_batch_size"]
        pilot_run = self.factors["pilot_run"]
        selected = scored_candidates[:search_batch_size]

        if selected and not any(
            candidate not in self.cache or self.cache[candidate].n_reps < pilot_run
            for candidate in selected
        ):
            for candidate in scored_candidates[search_batch_size:]:
                solution = self.cache.get(candidate)
                if solution is None or solution.n_reps < pilot_run:
                    selected[-1] = candidate
                    break

        trial_solutions = []
        for candidate in selected:
            solution = self._get_solution(candidate)
            self._ensure_replications(solution, pilot_run)
            if solution.n_reps > 0:
                trial_solutions.append(solution)
        return trial_solutions

    def _comparison_error(self, candidate_solution: Solution) -> float:
        incumbent_var = self._objective_variance(self.incumbent_solution)
        candidate_var = self._objective_variance(candidate_solution)
        incumbent_err = incumbent_var / max(self.incumbent_solution.n_reps, 1)
        candidate_err = candidate_var / max(candidate_solution.n_reps, 1)
        return float(np.sqrt(max(incumbent_err + candidate_err, 1e-12)))

    def _refine_comparison(self, candidate_solution: Solution) -> float:
        actual_reduction = self._transformed_objective(
            self.incumbent_solution
        ) - self._transformed_objective(candidate_solution)
        while (
            abs(actual_reduction) <= self._comparison_error(candidate_solution)
            and max(candidate_solution.n_reps, self.incumbent_solution.n_reps)
            < self.factors["max_compare_reps"]
            and self.ctx.budget.remaining > 0
        ):
            self._ensure_replications(candidate_solution, candidate_solution.n_reps + 1)
            self._ensure_replications(self.incumbent_solution, self.incumbent_solution.n_reps + 1)
            actual_reduction = self._transformed_objective(
                self.incumbent_solution
            ) - self._transformed_objective(candidate_solution)
        return actual_reduction

    def _update_radius_after_success(self, rho: float, predicted_reduction: float) -> None:
        if predicted_reduction > 0 and rho < self.factors["eta_1"]:
            shrunk = floor(self.factors["gamma_2"] * self.radius)
            self.radius = max(1, shrunk)
            return

        if predicted_reduction > 0 and rho >= self.factors["eta_2"]:
            expanded = ceil(self.factors["gamma_1"] * self.radius)
            self.radius = min(self.max_radius, max(self.radius, expanded))

    def _iterate(self) -> None:
        self.iteration_count += 1
        model = self._fit_local_model()
        trial_solutions = self._select_trial_solutions(model)
        if not trial_solutions:
            self.failed_iterations += 1
            self.radius = max(1, floor(self.factors["gamma_2"] * self.radius))
            return

        candidate_solution = min(trial_solutions, key=self._transformed_objective)
        actual_reduction = self._refine_comparison(candidate_solution)
        predicted_reduction = model.predict(self.incumbent_solution.x) - model.predict(
            candidate_solution.x
        )
        rho = actual_reduction / predicted_reduction if predicted_reduction > 0 else 1.0

        if actual_reduction > 0:
            self.incumbent_solution = candidate_solution
            self.failed_iterations = 0
            self.ctx.log(candidate_solution)
            self._update_radius_after_success(rho, predicted_reduction)
            return

        self.failed_iterations += 1
        shrunk = floor(self.factors["gamma_2"] * self.radius)
        self.radius = max(1, shrunk)

    def solve(self, problem: Problem, ctx: Context) -> None:
        self._initialize(problem, ctx)
        initial_x = tuple(int(value) for value in problem.factors["initial_solution"])
        self.incumbent_solution = self._get_solution(initial_x)
        self._ensure_replications(self.incumbent_solution, self.factors["pilot_run"])
        self.ctx.log(self.incumbent_solution)

        stalled_iterations = 0
        while self.ctx.budget.remaining > 0:
            budget_before = self.ctx.budget.used
            self._iterate()
            if self.ctx.budget.used == budget_before:
                stalled_iterations += 1
                if stalled_iterations >= self.factors["shake_after"] + 1:
                    break
            else:
                stalled_iterations = 0
