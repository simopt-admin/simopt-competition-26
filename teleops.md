# Tele-Operator Scheduling Model

## Model: Tele-Operators (TELEOPS)

### Description

When autonomous vehicles are unable to determine the right course of action, they can connect with a tele-operator center staffed by human drivers. The drivers can then remotely operate the vehicle until its usual autonomous function can resume. If all drivers are busy when a vehicle connects with the center then the vehicle has to wait for the next available driver.

We consider a shift-scheduling problem for a 24 hour tele-operator center. Tele-operators work 8 hour shifts that start on the hour, every hour, with an unpaid 30 minute "lunch" break that begins after the driver has worked for 4 hours. If a driver is driving a vehicle when their lunch break is scheduled to start, or when their shift is scheduled to end, then they finish driving the vehicle they are currently driving and then go to lunch or end their shift. The drivers are paid overtime at 1.5 times their usual hourly rate for this additional work and are also paid at a higher hourly rate between 10pm and 6am. For example, a car driver shift starting at 6pm that runs for 8 hours with an unpaid 30 minute lunch break at 10pm at a cost of \$20 per hour costs \$20 per hour from 6pm-10pm, then at a premium of 1.2 for late hours, \$24 per hour from 10:30pm till 2am, giving a total of \$20\*4 + \$24 \* 3.5 = \$164. Overtime costs are in addition to this value for the shift cost.

The center handles calls from both autonomous trucks and autonomous cars. Drivers have one role for their entire shift: car-only drivers accept only car calls, flexible car+truck drivers accept either kind of call, and truck-only drivers accept only truck calls. Compatible dedicated drivers receive waiting calls before flexible drivers; flexible drivers then receive the oldest call. Flexible car+truck drivers are paid at the truck-driver rate.

The times required to handle car requests are i.i.d. random variables denoted by $`V^{(C)}`$, measured in minutes. The corresponding i.i.d. times for truck requests are denoted $`V^{(T)}`$, and are independent of car request times. These values do not count the queueing time (waiting time of a request, prior to being assigned to a driver). These successive waiting times are denoted $`W^{(C)}`$ and $`W^{(T)}`$ respectively for cars and trucks, are measured internally in minutes, and equal 0 when vehicles reach a driver immediately. Reported daily waiting-time responses convert the accumulated minutes to hours.

Each replication consists of nine continuous days, starting from an empty system at midnight. Day 1 is a warm-up period, calls arriving during days 2 through 8 enter the reported responses, and day 9 is an end buffer. Calls arriving during the warm-up and end-buffer days still evolve the system state but are excluded from the reported responses.

For calls arriving during days 2 through 8, waiting time runs from arrival until assignment to a driver, including assignments during day 9. For measured calls still unanswered at the end of day 9, report the elapsed time from arrival to the end of day 9. Sum these waits and divide by seven days (and 60 to convert minutes to hours) for the daily waiting-time responses. This cutoff does not count unanswered calls as served.

Service between scheduled lunch start and actual lunch start earns the overtime rate. Within the scheduled shift, this replaces regular pay for those minutes; the actual lunch break remains unpaid. If the same call continues past shift end, those later minutes earn overtime once. Reported overtime includes both lunch-delay service and shift-end overruns.

### Sources of Randomness

1.  **Car and truck requests** arise according to independent Poisson processes with cyclic daily rates given by $`(\lambda^{(C)}(t): 0 \le t < 24)`$ and $`(\lambda^{(T)}(t): 0 \le t < 24)`$ for cars and trucks respectively. These rate functions are piecewise constant with one rate per hour.
2.  **Driving time durations** $`(V^{(C)})`$ are a mixture of two log-normal distributions. More specifically, with probability $`p^{(C)}_i`$ we set $`m^{(C)} = m^{(C)}_i, \sigma^{(C)} = \sigma^{(C)}_i`$, for $`i=1, 2`$, where $`p^{(C)}_1+p^{(C)}_2=1`$. The values on the right-hand sides of these expressions with subscripts 1 and 2 are model factors. We then set $`\mu = \ln(m^{(C)}) + (\sigma^{(C)})^2`$ and use the method `rng.lognormvariate`$`(\mu, \sigma)`$ to obtain the driving duration in minutes. Driving time durations for trucks are similar, but with truck-specific parameters.
3.  Independent random streams are used for car arrivals, truck arrivals, car driving durations and truck driving durations.

### Model Factors

- `car_only_drivers`: Number of car-only drivers starting a shift at each clock hour.
  - This is a vector of 24 non-negative integers.
  - Default: `(6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0)`

- `car_truck_drivers`: Number of flexible car+truck drivers starting a shift at each clock hour.
  - This is a vector of 24 non-negative integers.
  - Default: `(1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)`

- `truck_only_drivers`: Number of truck-only drivers starting a shift at each clock hour.
  - This is a vector of 24 non-negative integers.
  - Default: `(2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)`

- `car_calls_per_hour`: Daily-average number of car requests per hour, scaled by the hourly rate multipliers.
  - Default: 12.0

- `truck_calls_per_hour`: Daily-average number of truck requests per hour, scaled by the hourly rate multipliers.
  - Default: 7.0

- `hourly_rate_multipliers`: Multiplicative factors that scale the daily-average arrival rates by clock hour.
  - Default: (0.25, 0.2, 0.15, 0.1, 0.15, 0.3, 0.85, 1.8, 2.2, 1.8, 1.2, 1.05, 1.0, 1.0, 1.05, 1.2, 1.65, 2.1, 1.95, 1.45, 1.0, 0.7, 0.5, 0.35)

- `car_drive_time_params`: Car-service parameters `(mode1, mode2, sigma1, sigma2, weight)`, where the modes are in minutes, the sigmas are log-scale standard deviations, and the weight is the probability of selecting the first lognormal component.
  - Default: `(2.0, 5.0, 0.15, 0.12, 0.5)`

- `truck_drive_time_params`: Truck-service parameters `(mode1, mode2, sigma1, sigma2, weight)`, where the modes are in minutes, the sigmas are log-scale standard deviations, and the weight is the probability of selecting the first lognormal component.
  - Default: `(2.0, 10.0, 0.15, 0.12, 0.5)`

- `car_rate`: Dollars per hour paid to car-only drivers.
  - Default: 20

- `truck_rate`: Dollars per hour paid to truck-only and flexible car+truck drivers.
  - Default: 30

- `late_night_start_hour`: Time at which per-hour rates get higher  
  - Default: 22

- `late_night_end_hour`: Time at which per-hour rates return to normal  
  - Default: 6

- `late_night_rate_multiplier`: Multiplier on the hourly cost for late-night hours.  
  - Default: 1.2

- `overtime_rate_multiplier`: Multiplier on the hourly cost for service delaying lunch or continuing past shift end.
  - Default: 1.5

- `shift_hours`: Duration of shifts.  
  - Default: 8

- `lunch_start`: Hours of shift after which lunch starts  
  - Default: 4

- `objective_weight`: Staff-cost penalty weight in the objective function.
  - Default: 0.00005

### Responses

- `immediate_service_fraction`: Fraction of measured car and truck requests assigned to a compatible driver immediately, without waiting

- `staff_cost`: Average daily staffing cost over the seven measured days, including scheduled and overtime labor costs

- `daily_sum_car_wait_times`: Average daily sum of car waiting times over the seven measured days (days 2 through 8), reported in hours per day

- `daily_sum_truck_wait_times`: Average daily sum of truck waiting times over the seven measured days (days 2 through 8), reported in hours per day

- `daily_car_driver_overtime`: Average daily car-only driver service time delaying lunch or continuing past the scheduled shift end, counting overlapping time once, summed over measured shifts and reported in hours per day over the seven measured days

- `daily_truck_driver_overtime`: Average daily truck-only driver service time delaying lunch or continuing past the scheduled shift end, counting overlapping time once, summed over measured shifts and reported in hours per day over the seven measured days

- `daily_car_truck_driver_overtime`: Average daily flexible car+truck driver service time delaying lunch or continuing past the scheduled shift end, counting overlapping time once, summed over measured shifts and reported in hours per day over the seven measured days

- `objective`: `immediate_service_fraction - objective_weight * staff_cost`

### References

This model setup is similar to that of a call center. It was devised by David Eckman, Shane Henderson, Sara Shashaani and Cen Wang.

## Optimization Problem: Maximize Immediate Service Minus Staffing Cost (TELEOPS-1)

### Decision Variables

- `car_only_drivers`: 24 hourly car-only shift-start counts.
- `car_truck_drivers`: 24 hourly flexible car+truck shift-start counts.
- `truck_only_drivers`: 24 hourly truck-only shift-start counts.

The 72-dimensional decision vector stores these blocks in that order.

### Objectives

With the default `objective_weight` of 0.00005, the objective is to maximize

```text
immediate_service_fraction - 0.00005 * staff_cost
```

Other objective formulations may also be evaluated to avoid overfitting, including objectives based on waiting time.

### Constraints

- Each of the decision vectors above consists of nonnegative integers.

### Problem Factors

- `budget`: Maximum number of simulation replications. Each replication simulates nine days.
  - Default: 1000

- `rng_streams`: Separate streams for car arrivals, truck arrivals, car drive times, truck drive times  
  - Default: provided by the framework

### Starting Solution

- `initial_solution`: Concatenates the default `car_only_drivers`, `car_truck_drivers`, and `truck_only_drivers` vectors. Starts occur only at hours 0, 8, and 16, where the three role counts are `(6, 1, 2)`, `(5, 2, 1)`, and `(4, 1, 1)`, respectively.

### Random Solutions

Draw the total number of daily shifts from a geometric distribution on the nonnegative integers, with mean equal to the initial solution’s total shift count (23 by default). Assign each shift independently and uniformly to one of the 72 hourly role-specific slots. This sets the expected staffing level to a useful scale while giving every feasible integer schedule a positive probability of being sampled.

### Optimal Solution

Unknown

### Optimal Objective Function Value

Unknown
