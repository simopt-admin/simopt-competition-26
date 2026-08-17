"""SimOpt model and problem wrappers for the teleops simulation."""

from __future__ import annotations

import math
from typing import Annotated, ClassVar

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a
from pydantic import BaseModel, Field
from simopt import dsl
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import override

from .distributions import BimodalLognormal
from .simulation import Demand, Payment, Service, StaffingPlan, run_simulation

DEFAULT_CAR_ONLY_DRIVERS = (6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0)
DEFAULT_CAR_TRUCK_DRIVERS = (1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
DEFAULT_TRUCK_ONLY_DRIVERS = (
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)
DEFAULT_INITIAL_SOLUTION = (
    DEFAULT_CAR_ONLY_DRIVERS + DEFAULT_CAR_TRUCK_DRIVERS + DEFAULT_TRUCK_ONLY_DRIVERS
)
DEFAULT_HOURLY_RATE_MULTIPLIERS = (
    0.25,
    0.2,
    0.15,
    0.1,
    0.15,
    0.3,
    0.85,
    1.8,
    2.2,
    1.8,
    1.2,
    1.05,
    1.0,
    1.0,
    1.05,
    1.2,
    1.65,
    2.1,
    1.95,
    1.45,
    1.0,
    0.7,
    0.5,
    0.35,
)
DEFAULT_CAR_DRIVE_TIME_PARAMS = (2.0, 5.0, 0.15, 0.12, 0.5)
DEFAULT_TRUCK_DRIVE_TIME_PARAMS = (2.0, 10.0, 0.15, 0.12, 0.5)

NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
DriveTimeParams = tuple[PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat, Probability]


class TeleopsModelConfig(BaseModel):
    """Configuration for one teleops simulation replication."""

    car_only_drivers: Annotated[
        tuple[NonnegativeInt, ...],
        Field(
            default=DEFAULT_CAR_ONLY_DRIVERS,
            min_length=24,
            max_length=24,
            description="car-only shift starts by clock hour",
        ),
    ]
    car_truck_drivers: Annotated[
        tuple[NonnegativeInt, ...],
        Field(
            default=DEFAULT_CAR_TRUCK_DRIVERS,
            min_length=24,
            max_length=24,
            description="flexible car+truck shift starts by clock hour",
        ),
    ]
    truck_only_drivers: Annotated[
        tuple[NonnegativeInt, ...],
        Field(
            default=DEFAULT_TRUCK_ONLY_DRIVERS,
            min_length=24,
            max_length=24,
            description="truck-only shift starts by clock hour",
        ),
    ]
    car_calls_per_hour: Annotated[
        PositiveFloat, Field(default=12.0, description="daily-average car calls per hour")
    ]
    truck_calls_per_hour: Annotated[
        PositiveFloat, Field(default=7.0, description="daily-average truck calls per hour")
    ]
    hourly_rate_multipliers: Annotated[
        tuple[PositiveFloat, ...],
        Field(
            default=DEFAULT_HOURLY_RATE_MULTIPLIERS,
            min_length=24,
            max_length=24,
            description="arrival-rate multiplier by clock hour",
        ),
    ]
    car_drive_time_params: Annotated[
        DriveTimeParams,
        Field(
            default=DEFAULT_CAR_DRIVE_TIME_PARAMS,
            description="car service (mode1, mode2, sigma1, sigma2, weight)",
        ),
    ]
    truck_drive_time_params: Annotated[
        DriveTimeParams,
        Field(
            default=DEFAULT_TRUCK_DRIVE_TIME_PARAMS,
            description="truck service (mode1, mode2, sigma1, sigma2, weight)",
        ),
    ]
    objective_weight: Annotated[
        float, Field(default=0.00005, ge=0, description="staff-cost penalty weight")
    ]
    car_rate: Annotated[float, Field(default=20.0, ge=0)]
    truck_rate: Annotated[float, Field(default=30.0, ge=0)]
    late_night_start_hour: Annotated[int, Field(default=22, ge=0, le=23)]
    late_night_end_hour: Annotated[int, Field(default=6, ge=0, le=23)]
    late_night_rate_multiplier: Annotated[float, Field(default=1.2, ge=0)]
    overtime_rate_multiplier: Annotated[float, Field(default=1.5, ge=0)]
    shift_hours: Annotated[int, Field(default=8, gt=0, le=24)]
    lunch_start: Annotated[
        float, Field(default=4.0, ge=0, description="hours after shift start before lunch")
    ]


class TeleopsProblemConfig(BaseModel):
    """Configuration for optimizing teleops shift starts."""

    initial_solution: Annotated[
        tuple[NonnegativeInt, ...],
        Field(
            default=DEFAULT_INITIAL_SOLUTION,
            min_length=72,
            max_length=72,
            description="24 shift starts for each of the three driver roles",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            gt=0,
            description="maximum number of simulation replications",
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class TeleopsModel(Model):
    """SimOpt simulation model for a 24-hour teleops center."""

    class_name_abbr: ClassVar[str] = "TELEOPS"
    class_name: ClassVar[str] = "Teleops Center"
    config_class: ClassVar[type[BaseModel]] = TeleopsModelConfig
    n_rngs: ClassVar[int] = 4
    n_responses: ClassVar[int] = 8

    def replicate(self, factors: dict, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Run one replication using factors and RNGs supplied by SimOpt."""
        staffing_plan = StaffingPlan(
            car_only_drivers=tuple(int(value) for value in factors["car_only_drivers"]),
            car_truck_drivers=tuple(int(value) for value in factors["car_truck_drivers"]),
            truck_only_drivers=tuple(int(value) for value in factors["truck_only_drivers"]),
        )
        demand = Demand(
            car_calls_per_hour=factors["car_calls_per_hour"],
            truck_calls_per_hour=factors["truck_calls_per_hour"],
            hourly_rate_multipliers=tuple(factors["hourly_rate_multipliers"]),
        )
        car_mode1, car_mode2, car_sigma1, car_sigma2, car_weight = factors["car_drive_time_params"]
        truck_mode1, truck_mode2, truck_sigma1, truck_sigma2, truck_weight = factors[
            "truck_drive_time_params"
        ]
        service = Service(
            car_distribution=BimodalLognormal(
                mode1=car_mode1,
                mode2=car_mode2,
                sigma1=car_sigma1,
                sigma2=car_sigma2,
                weight=car_weight,
            ),
            truck_distribution=BimodalLognormal(
                mode1=truck_mode1,
                mode2=truck_mode2,
                sigma1=truck_sigma1,
                sigma2=truck_sigma2,
                weight=truck_weight,
            ),
        )

        (
            immediate_service_fraction,
            staff_cost,
            daily_sum_car_wait_times,
            daily_sum_truck_wait_times,
            daily_car_driver_overtime,
            daily_truck_driver_overtime,
            daily_car_truck_driver_overtime,
        ) = run_simulation(
            staffing_plan,
            demand,
            service,
            rngs,
            payment=Payment(
                car_rate=factors["car_rate"],
                truck_rate=factors["truck_rate"],
                late_night_start_hour=factors["late_night_start_hour"],
                late_night_end_hour=factors["late_night_end_hour"],
                late_night_rate_multiplier=factors["late_night_rate_multiplier"],
                overtime_rate_multiplier=factors["overtime_rate_multiplier"],
            ),
            shift_hours=factors["shift_hours"],
            lunch_start_hours=factors["lunch_start"],
        )
        responses = {
            "immediate_service_fraction": immediate_service_fraction,
            "staff_cost": staff_cost,
            "daily_sum_car_wait_times": daily_sum_car_wait_times,
            "daily_sum_truck_wait_times": daily_sum_truck_wait_times,
            "daily_car_driver_overtime": daily_car_driver_overtime,
            "daily_truck_driver_overtime": daily_truck_driver_overtime,
            "daily_car_truck_driver_overtime": daily_car_truck_driver_overtime,
            "objective": immediate_service_fraction - factors["objective_weight"] * staff_cost,
        }
        return responses, {}


class TeleopsProblem(Problem):
    """Maximize immediate service minus weighted simulated staffing cost."""

    class_name_abbr: ClassVar[str] = "TELEOPS-1"
    class_name: ClassVar[str] = "Max Teleops Service Minus Staffing Cost"
    config_class: ClassVar[type[BaseModel]] = TeleopsProblemConfig
    model_class: ClassVar[type[Model]] = TeleopsModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: ClassVar[tuple | None] = None
    model_default_factors: ClassVar[dict] = {
        "car_only_drivers": DEFAULT_CAR_ONLY_DRIVERS,
        "car_truck_drivers": DEFAULT_CAR_TRUCK_DRIVERS,
        "truck_only_drivers": DEFAULT_TRUCK_ONLY_DRIVERS,
    }
    model_decision_factors: ClassVar[set[str]] = {
        "car_only_drivers",
        "car_truck_drivers",
        "truck_only_drivers",
    }

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        staffing = problem.add_integer_vector(
            lb=0, ub=np.inf, shape=(self.dim,), initial=self.factors["initial_solution"]
        )
        simulation = self.add_simulation(problem, {"staffing": staffing})
        problem.maximize(dsl.mean(simulation.metric("objective")))
        return problem

    @property
    @override
    def dim(self) -> int:
        return len(self.factors["initial_solution"])

    @property
    @override
    def lower_bounds(self) -> tuple[float, ...]:
        return (0.0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple[float, ...]:
        return (np.inf,) * self.dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict[str, tuple]:
        return {
            "car_only_drivers": vector[:24],
            "car_truck_drivers": vector[24:48],
            "truck_only_drivers": vector[48:],
        }

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (
            tuple(factor_dict["car_only_drivers"])
            + tuple(factor_dict["car_truck_drivers"])
            + tuple(factor_dict["truck_only_drivers"])
        )

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return len(x) == 72 and super().check_deterministic_constraints(x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Default random solution generator. You may change this."""
        mean_shifts = max(1, sum(self.factors["initial_solution"]))
        total = int(rand_sol_rng.expovariate(math.log1p(1 / mean_shifts)))
        counts = [0] * self.dim
        for _ in range(total):
            counts[rand_sol_rng.randint(0, self.dim - 1)] += 1
        return tuple(counts)
