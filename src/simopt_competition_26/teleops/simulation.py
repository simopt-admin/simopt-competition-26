"""SimPy model of a 24-hour teleops center.

Calls and idle drivers are matched by one central dispatcher. Dedicated
car-only and truck-only drivers receive compatible calls first; flexible
car+truck drivers receive the oldest call left after those assignments.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import simpy

from .distributions import BimodalLognormal, Nhpp

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60.0
SHIFT_HOURS = 8
SHIFT_MINUTES = SHIFT_HOURS * MINUTES_PER_HOUR
LUNCH_START_HOURS = 4.0
LUNCH_BREAK_MINUTES = 30.0
DAY_MINUTES = HOURS_PER_DAY * MINUTES_PER_HOUR
SIMULATION_DAYS = 9
WARMUP_DAYS = 1
END_BUFFER_DAYS = 1
MEASUREMENT_DAYS = SIMULATION_DAYS - WARMUP_DAYS - END_BUFFER_DAYS
SIMULATION_MINUTES = SIMULATION_DAYS * DAY_MINUTES
MEASUREMENT_START = WARMUP_DAYS * DAY_MINUTES
MEASUREMENT_END = (SIMULATION_DAYS - END_BUFFER_DAYS) * DAY_MINUTES


class CallKind(StrEnum):
    """Kinds of calls handled by the center."""

    CAR = "car"
    TRUCK = "truck"


class DriverKind(StrEnum):
    """Fixed roles a driver can hold for one shift."""

    CAR_ONLY = "car_only"
    CAR_TRUCK = "car_truck"
    TRUCK_ONLY = "truck_only"


@dataclass(frozen=True)
class StaffingPlan:
    """Shift-start staffing choices supplied by an optimizer or caller."""

    # Each tuple is indexed by clock hour and gives the number of drivers of
    # that fixed role who begin an eight-hour shift in that hour.
    car_only_drivers: tuple[int, ...]
    car_truck_drivers: tuple[int, ...]
    truck_only_drivers: tuple[int, ...]


@dataclass(frozen=True)
class Demand:
    """Daily-average call rates and their shared hourly demand profile."""

    car_calls_per_hour: float
    truck_calls_per_hour: float
    hourly_rate_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class Service:
    """Service-time distributions for car and truck calls."""

    car_distribution: BimodalLognormal
    truck_distribution: BimodalLognormal


@dataclass(frozen=True)
class Payment:
    """Driver pay rates used to price completed shifts."""

    car_rate: float
    truck_rate: float
    late_night_start_hour: int
    late_night_end_hour: int
    late_night_rate_multiplier: float
    overtime_rate_multiplier: float


@dataclass(frozen=True)
class Call:
    """One car or truck service request."""

    kind: CallKind
    service_minutes: float


@dataclass
class DriverOffer:
    """One idle driver's cancellable offer to accept a dispatched call."""

    accept_until: float
    assignment: simpy.Event
    active: bool = True


@dataclass(frozen=True)
class _QueuedCall:
    sequence: int
    call: Call
    submitted_at: float


@dataclass
class Collector:
    """Accumulate measured shift costs and call waiting times."""

    measurement_start: float = 0.0
    measurement_end: float | None = None
    staff_cost: float = 0.0
    car_wait_minutes: float = 0.0
    truck_wait_minutes: float = 0.0
    car_driver_overtime_minutes: float = 0.0
    truck_driver_overtime_minutes: float = 0.0
    car_truck_driver_overtime_minutes: float = 0.0

    def submit_shift(
        self, shift_start: float, kind: DriverKind, cost: float, overtime_minutes: float
    ) -> None:
        """Add the cost and overtime of one completed measured shift."""
        if self._in_measurement_window(shift_start):
            self.staff_cost += cost
            if kind is DriverKind.CAR_ONLY:
                self.car_driver_overtime_minutes += overtime_minutes
            elif kind is DriverKind.TRUCK_ONLY:
                self.truck_driver_overtime_minutes += overtime_minutes
            else:
                self.car_truck_driver_overtime_minutes += overtime_minutes

    def submit_wait(self, call_arrival: float, kind: CallKind, wait_minutes: float) -> None:
        """Add one measured call's waiting time to its vehicle total."""
        if not self._in_measurement_window(call_arrival):
            return
        if kind is CallKind.CAR:
            self.car_wait_minutes += wait_minutes
        else:
            self.truck_wait_minutes += wait_minutes

    def _in_measurement_window(self, time: float) -> bool:
        return time >= self.measurement_start and (
            self.measurement_end is None or time < self.measurement_end
        )


class Dispatcher:
    """Match waiting calls to idle drivers using dedicated-first routing."""

    def __init__(
        self,
        env: simpy.Environment,
        measurement_start: float = 0.0,
        measurement_end: float | None = None,
        collector: Collector | None = None,
    ) -> None:
        self.env = env
        self.measurement_start = measurement_start
        self.measurement_end = measurement_end
        self.collector = collector
        self._waiting: dict[CallKind, deque[_QueuedCall]] = {
            CallKind.CAR: deque(),
            CallKind.TRUCK: deque(),
        }
        self._idle: dict[DriverKind, deque[DriverOffer]] = {
            DriverKind.CAR_ONLY: deque(),
            DriverKind.CAR_TRUCK: deque(),
            DriverKind.TRUCK_ONLY: deque(),
        }
        self._next_sequence = 0
        self._calls_submitted = 0
        self._calls_immediately_serviced = 0

    @property
    def waiting_calls(self) -> tuple[Call, ...]:
        """Return currently waiting calls in global FIFO order."""
        queued = (*self._waiting[CallKind.CAR], *self._waiting[CallKind.TRUCK])
        return tuple(item.call for item in sorted(queued, key=lambda item: item.sequence))

    @property
    def immediate_service_fraction(self) -> float:
        """Return the fraction of submitted calls assigned without waiting."""
        return self._calls_immediately_serviced / self._calls_submitted

    def submit(self, call: Call) -> None:
        """Add a call and immediately make every currently possible match."""
        self._waiting[call.kind].append(_QueuedCall(self._next_sequence, call, self.env.now))
        self._next_sequence += 1
        if self._in_measurement_window(self.env.now):
            self._calls_submitted += 1
        self._dispatch()

    def record_unanswered_waits(self) -> None:
        """Record remaining waits once, after the simulation has stopped."""
        if self.collector is None:
            return
        for calls in self._waiting.values():
            for queued_call in calls:
                self.collector.submit_wait(
                    queued_call.submitted_at,
                    queued_call.call.kind,
                    self.env.now - queued_call.submitted_at,
                )

    def _in_measurement_window(self, time: float) -> bool:
        """Return whether a call arriving at ``time`` belongs in the response."""
        return time >= self.measurement_start and (
            self.measurement_end is None or time < self.measurement_end
        )

    def offer(self, kind: DriverKind, accept_until: float) -> DriverOffer:
        """Register one idle driver and return its assignment event."""
        offer = DriverOffer(accept_until, self.env.event())
        self._idle[kind].append(offer)
        self._dispatch()
        return offer

    def withdraw(self, offer: DriverOffer) -> None:
        """Make an unassigned idle offer ineligible for future dispatch."""
        offer.active = False

    def _dispatch(self) -> None:
        # Exhaust dedicated-driver matches before flexible drivers can take
        # work from either call queue.
        self._assign_dedicated(CallKind.CAR, DriverKind.CAR_ONLY)
        self._assign_dedicated(CallKind.TRUCK, DriverKind.TRUCK_ONLY)
        self._assign_flexible()

    def _assign_dedicated(self, call_kind: CallKind, driver_kind: DriverKind) -> None:
        calls = self._waiting[call_kind]
        while calls:
            offer = self._take_idle(driver_kind)
            if offer is None:
                return
            self._assign(offer, calls.popleft())

    def _assign_flexible(self) -> None:
        while self._waiting[CallKind.CAR] or self._waiting[CallKind.TRUCK]:
            offer = self._take_idle(DriverKind.CAR_TRUCK)
            if offer is None:
                return

            car_calls = self._waiting[CallKind.CAR]
            truck_calls = self._waiting[CallKind.TRUCK]
            if not truck_calls or (car_calls and car_calls[0].sequence < truck_calls[0].sequence):
                queued_call = car_calls.popleft()
            else:
                queued_call = truck_calls.popleft()
            self._assign(offer, queued_call)

    def _take_idle(self, kind: DriverKind) -> DriverOffer | None:
        offers = self._idle[kind]
        while offers:
            offer = offers.popleft()
            if offer.active and self.env.now < offer.accept_until:
                return offer
            offer.active = False
        return None

    def _assign(self, offer: DriverOffer, queued_call: _QueuedCall) -> None:
        offer.active = False
        if self.collector is not None:
            self.collector.submit_wait(
                queued_call.submitted_at,
                queued_call.call.kind,
                self.env.now - queued_call.submitted_at,
            )
        if queued_call.submitted_at == self.env.now and self._in_measurement_window(
            queued_call.submitted_at
        ):
            self._calls_immediately_serviced += 1
        offer.assignment.succeed(queued_call.call)


def call_generator(
    env: simpy.Environment,
    dispatcher: Dispatcher,
    kind: CallKind,
    arrival_distribution: Nhpp,
    service_distribution: BimodalLognormal,
    arrival_rng: random.Random,
    service_rng: random.Random,
) -> Generator[simpy.Event, Any]:
    """Generate one type of call for the full simulation horizon."""
    while True:
        yield env.timeout(arrival_distribution.sample(env.now, arrival_rng))
        call = Call(kind=kind, service_minutes=service_distribution.sample(service_rng))
        dispatcher.submit(call)


def _serve_until(
    env: simpy.Environment,
    dispatcher: Dispatcher,
    kind: DriverKind,
    accept_until: float,
    shift_end: float,
) -> Generator[simpy.Event, Any, float]:
    """Serve through a cutoff and return exact service overtime in minutes."""
    overtime_minutes = 0.0
    while env.now < accept_until:
        offer = dispatcher.offer(kind, accept_until)
        cutoff = env.timeout(accept_until - env.now)
        outcome = yield offer.assignment | cutoff

        if offer.assignment not in outcome:
            dispatcher.withdraw(offer)
            return overtime_minutes

        call: Call = outcome[offer.assignment]
        yield env.timeout(call.service_minutes)
        overtime_minutes = max(overtime_minutes, env.now - shift_end)
    return overtime_minutes


def _interval_cost(
    start: float, end: float, hourly_rate: float, payment: Payment, overtime: bool = False
) -> float:
    """Price an exact interval, splitting only where the hourly rate changes."""
    cost = 0.0
    while start < end:
        segment_end = min(end, (int(start // MINUTES_PER_HOUR) + 1) * MINUTES_PER_HOUR)
        clock_hour = int(start // MINUTES_PER_HOUR) % HOURS_PER_DAY
        night_start = payment.late_night_start_hour
        night_end = payment.late_night_end_hour
        if night_start < night_end:
            late_night = night_start <= clock_hour < night_end
        else:
            late_night = (
                night_start == night_end or clock_hour >= night_start or clock_hour < night_end
            )
        rate_multiplier = payment.overtime_rate_multiplier if overtime else 1.0
        if late_night:
            rate_multiplier *= payment.late_night_rate_multiplier
        cost += hourly_rate * rate_multiplier * (segment_end - start) / MINUTES_PER_HOUR
        start = segment_end

    return cost


def driver(
    env: simpy.Environment,
    dispatcher: Dispatcher,
    kind: DriverKind,
    shift_end: float,
    payment: Payment,
    collector: Collector,
    lunch_start_hours: float = LUNCH_START_HOURS,
) -> Generator[simpy.Event, Any]:
    """Serve one shift and submit its exact cost when the driver leaves."""
    shift_start = env.now
    lunch_time = min(shift_start + lunch_start_hours * MINUTES_PER_HOUR, shift_end)
    hourly_rate = payment.car_rate if kind is DriverKind.CAR_ONLY else payment.truck_rate
    cost = _interval_cost(shift_start, shift_end, hourly_rate, payment)

    overtime_minutes = yield from _serve_until(env, dispatcher, kind, lunch_time, shift_end)
    # Scheduled pay already covers lunch-delay service within the shift.
    # Upgrade those minutes to overtime; service beyond shift end is priced below.
    lunch_delay_end = min(env.now, shift_end)
    lunch_overtime_minutes = max(0.0, lunch_delay_end - lunch_time)
    cost += _interval_cost(
        lunch_time, lunch_delay_end, hourly_rate, payment, overtime=True
    ) - _interval_cost(lunch_time, lunch_delay_end, hourly_rate, payment)
    if env.now < shift_end:
        # Lunch is unpaid at the rates in effect when the break actually occurs.
        cost -= _interval_cost(
            env.now, min(env.now + LUNCH_BREAK_MINUTES, shift_end), hourly_rate, payment
        )
        yield env.timeout(LUNCH_BREAK_MINUTES)
        overtime_minutes = max(
            overtime_minutes, (yield from _serve_until(env, dispatcher, kind, shift_end, shift_end))
        )

    cost += _interval_cost(
        shift_end, shift_end + overtime_minutes, hourly_rate, payment, overtime=True
    )
    collector.submit_shift(shift_start, kind, cost, overtime_minutes + lunch_overtime_minutes)


def start_shift(
    env: simpy.Environment,
    dispatcher: Dispatcher,
    staffing_plan: StaffingPlan,
    shift_start_hour: int,
    payment: Payment,
    collector: Collector,
    shift_hours: float = SHIFT_HOURS,
    lunch_start_hours: float = LUNCH_START_HOURS,
) -> Generator[simpy.Event, Any]:
    """Start the driver cohort scheduled for one absolute clock hour."""
    shift_start = shift_start_hour * MINUTES_PER_HOUR
    if shift_start > env.now:
        yield env.timeout(shift_start - env.now)
    shift_end = shift_start + shift_hours * MINUTES_PER_HOUR

    clock_start_hour = shift_start_hour % HOURS_PER_DAY
    cohorts = (
        (DriverKind.CAR_ONLY, staffing_plan.car_only_drivers),
        (DriverKind.TRUCK_ONLY, staffing_plan.truck_only_drivers),
        (DriverKind.CAR_TRUCK, staffing_plan.car_truck_drivers),
    )
    for kind, hourly_starts in cohorts:
        for _ in range(hourly_starts[clock_start_hour]):
            env.process(
                driver(
                    env,
                    dispatcher,
                    kind,
                    shift_end,
                    payment=payment,
                    collector=collector,
                    lunch_start_hours=lunch_start_hours,
                )
            )


def run_simulation(
    staffing_plan: StaffingPlan,
    demand: Demand,
    service: Service,
    rngs: Sequence[random.Random],
    payment: Payment,
    shift_hours: float = SHIFT_HOURS,
    lunch_start_hours: float = LUNCH_START_HOURS,
) -> tuple[float, float, float, float, float, float, float]:
    """Run nine days and average responses over the middle seven days."""
    car_arrival_rng, truck_arrival_rng, car_service_rng, truck_service_rng = rngs

    env = simpy.Environment()
    collector = Collector(measurement_start=MEASUREMENT_START, measurement_end=MEASUREMENT_END)
    dispatcher = Dispatcher(
        env,
        measurement_start=MEASUREMENT_START,
        measurement_end=MEASUREMENT_END,
        collector=collector,
    )

    env.process(
        call_generator(
            env,
            dispatcher,
            CallKind.CAR,
            Nhpp(demand.car_calls_per_hour, demand.hourly_rate_multipliers),
            service.car_distribution,
            car_arrival_rng,
            car_service_rng,
        )
    )
    env.process(
        call_generator(
            env,
            dispatcher,
            CallKind.TRUCK,
            Nhpp(demand.truck_calls_per_hour, demand.hourly_rate_multipliers),
            service.truck_distribution,
            truck_arrival_rng,
            truck_service_rng,
        )
    )

    # Start empty at midnight and repeat the staffing plan for all nine days.
    # Day one warms up the system and day nine supplies an end buffer; calls
    # arriving on those days evolve the state but do not enter the response.
    for shift_start_hour in range(SIMULATION_DAYS * HOURS_PER_DAY):
        env.process(
            start_shift(
                env,
                dispatcher,
                staffing_plan,
                shift_start_hour,
                payment=payment,
                collector=collector,
                shift_hours=shift_hours,
                lunch_start_hours=lunch_start_hours,
            )
        )

    env.run(until=SIMULATION_MINUTES)
    dispatcher.record_unanswered_waits()
    return (
        dispatcher.immediate_service_fraction,
        collector.staff_cost / MEASUREMENT_DAYS,
        collector.car_wait_minutes / MINUTES_PER_HOUR / MEASUREMENT_DAYS,
        collector.truck_wait_minutes / MINUTES_PER_HOUR / MEASUREMENT_DAYS,
        collector.car_driver_overtime_minutes / MINUTES_PER_HOUR / MEASUREMENT_DAYS,
        collector.truck_driver_overtime_minutes / MINUTES_PER_HOUR / MEASUREMENT_DAYS,
        collector.car_truck_driver_overtime_minutes / MINUTES_PER_HOUR / MEASUREMENT_DAYS,
    )
