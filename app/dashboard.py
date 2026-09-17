from pathlib import Path
from math import ceil, floor

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AfyaFlow",
    description="Hospital demand forecasting and capacity simulation system",
    version="1.0"
)


# Allow the local HTML frontend to call the API even when it is
# opened from a different development port (for example VS Code
# Live Server). In production, replace ["*"] with your real origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "static"


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(filename: str) -> pd.DataFrame:

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# LOAD FORECAST DATA
# ============================================================

hourly_forecast = load_csv(
    "hourly_future_forecast.csv"
)

daily_forecast = load_csv(
    "daily_7day_forecast.csv"
)

weekly_forecast = load_csv(
    "future_weekly_ensemble.csv"
)


# ============================================================
# LOAD BASELINE SIMULATION DATA
# ============================================================

hourly_simulation = load_csv(
    "hourly_capacity_simulation.csv"
)

daily_simulation = load_csv(
    "daily_capacity_simulation.csv"
)

weekly_simulation = load_csv(
    "weekly_capacity_simulation.csv"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

datasets = [
    hourly_forecast,
    daily_forecast,
    weekly_forecast,
    hourly_simulation,
    daily_simulation,
    weekly_simulation
]

for df in datasets:

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

# Normalize department names so values such as "GENERAL OPD "
# match the frontend/API department selection.
for df in datasets:
    if "department" in df.columns:
        df["department"] = (
            df["department"]
            .astype(str)
            .str.strip()
        )


# ============================================================
# DATE CONVERSIONS
# ============================================================

hourly_forecast["date"] = pd.to_datetime(
    hourly_forecast["date"],
    errors="coerce"
)

daily_forecast["date"] = pd.to_datetime(
    daily_forecast["date"],
    errors="coerce"
)

weekly_forecast["week"] = pd.to_datetime(
    weekly_forecast["week"],
    errors="coerce"
)

hourly_simulation["date"] = pd.to_datetime(
    hourly_simulation["date"],
    errors="coerce"
)

daily_simulation["date"] = pd.to_datetime(
    daily_simulation["date"],
    errors="coerce"
)

weekly_simulation["week"] = pd.to_datetime(
    weekly_simulation["week"],
    errors="coerce"
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "demand",
    "forecast_demand",
    "capacity",
    "daily_capacity",
    "weekly_capacity",
    "patients_served",
    "patients_remaining",
    "utilization_pct",
    "capacity_utilization_pct",
    "simulated_arrivals",
    "patients_arrivals",
    "service_time_mins",
    "staff_available",
    "rooms_available"
]

for df in datasets:

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "datasets": {
            "hourly_forecast": len(hourly_forecast),
            "daily_forecast": len(daily_forecast),
            "weekly_forecast": len(weekly_forecast),
            "hourly_simulation": len(hourly_simulation),
            "daily_simulation": len(daily_simulation),
            "weekly_simulation": len(weekly_simulation)
        }
    }


# ============================================================
# DEPARTMENTS
# ============================================================

@app.get("/departments")
def departments():

    departments = sorted(
        set(
            hourly_forecast["department"]
            .dropna()
            .astype(str)
        )
        |
        set(
            daily_forecast["department"]
            .dropna()
            .astype(str)
        )
        |
        set(
            weekly_forecast["department"]
            .dropna()
            .astype(str)
        )
    )

    return departments


# ============================================================
# FORECAST ENDPOINTS
# ============================================================

@app.get("/forecast/hourly")
def forecast_hourly(
    department: str | None = Query(default=None)
):

    df = hourly_forecast.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


@app.get("/forecast/daily")
def forecast_daily(
    department: str | None = Query(default=None)
):

    df = daily_forecast.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


@app.get("/forecast/weekly")
def forecast_weekly(
    department: str | None = Query(default=None)
):

    df = weekly_forecast.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


# ============================================================
# BASELINE SIMULATION ENDPOINTS
# ============================================================

@app.get("/simulation/hourly")
def simulation_hourly(
    department: str | None = Query(default=None)
):

    df = hourly_simulation.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


@app.get("/simulation/daily")
def simulation_daily(
    department: str | None = Query(default=None)
):

    df = daily_simulation.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


@app.get("/simulation/weekly")
def simulation_weekly(
    department: str | None = Query(default=None)
):

    df = weekly_simulation.copy()

    if department:
        df = df[
            df["department"] == department
        ]

    return df.to_dict(
        orient="records"
    )


# ============================================================
# HELPER: STATUS
# ============================================================

def get_scenario_status(
    utilization: float,
    remaining: float,
    operating: bool = True
):

    if not operating and remaining > 0:
        return "Closed"

    if remaining > 0:
        return "Overloaded"

    if utilization >= 85:
        return "High"

    if utilization >= 70:
        return "Moderate"

    return "Low"


# ============================================================
# HELPER: OPERATING FLAG
# ============================================================

def parse_time(value):

    if pd.isna(value):
        return 0

    value = str(value).strip()

    if ":" not in value:
        return 0

    parts = value.split(":")

    try:
        return (
            int(parts[0]) * 60
            + int(parts[1])
        )
    except ValueError:
        return 0


def is_operating_hourly(
    row
):

    is_24_7 = bool(
        row.get("is_24_7", False)
    )

    if is_24_7:
        return True

    hour = int(
        row["hour"]
    )

    start = parse_time(
        row.get(
            "operating_start",
            "08:00"
        )
    )

    end = parse_time(
        row.get(
            "operating_end",
            "16:00"
        )
    )

    current = hour * 60

    return (
        start <= current < end
    )


# ============================================================
# STAFFING SCENARIO
#
# This is deliberately a scenario re-simulation.
#
# It does NOT alter the trained forecasting model.
#
# It takes the forecast as fixed demand and changes:
#     staff
#     rooms
#
# Then it recalculates:
#     capacity
#     patients served
#     queue/backlog
#     utilization
#     congestion
#
# Backlog carries from one period to the next.
# ============================================================

@app.get("/scenario/staff")
def staff_scenario(
    level: str = Query(...),
    department: str = Query(...),
    staff: int = Query(..., ge=1, le=500),
    rooms: int = Query(..., ge=1, le=500)
):
    """
    Re-simulate the selected department with the requested staffing/room
    scenario. This is a queue-based FCFS simulation, not a simple
    demand-minus-capacity calculation.

    Returns waiting-time metrics as well as capacity, utilization,
    patients served, remaining patients and congestion status.
    """

    level = level.lower().strip()
    if level not in {"hourly", "daily", "weekly"}:
        raise HTTPException(
            status_code=400,
            detail="level must be hourly, daily, or weekly"
        )

    # --------------------------------------------------------
    # Select the appropriate forecast and baseline configuration
    # --------------------------------------------------------
    if level == "hourly":
        forecast = hourly_forecast[hourly_forecast["department"] == department].copy()
        config = hourly_simulation[hourly_simulation["department"] == department].copy()
        period_col = "date"
    elif level == "daily":
        forecast = daily_forecast[daily_forecast["department"] == department].copy()
        config = daily_simulation[daily_simulation["department"] == department].copy()
        period_col = "date"
    else:
        forecast = weekly_forecast[weekly_forecast["department"] == department].copy()
        config = weekly_simulation[weekly_simulation["department"] == department].copy()
        period_col = "week"

    if forecast.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {level} forecast for {department}"
        )

    if config.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation configuration for {department}"
        )

    # Prefer an operating row with positive capacity because hourly
    # simulation files often begin with a closed hour whose capacity is 0.
    config_row = config.iloc[0]
    if "capacity" in config.columns:
        positive_capacity = config[config["capacity"] > 0]
        if not positive_capacity.empty:
            config_row = positive_capacity.iloc[0]
    elif "daily_capacity" in config.columns:
        positive_capacity = config[config["daily_capacity"] > 0]
        if not positive_capacity.empty:
            config_row = positive_capacity.iloc[0]
    elif "weekly_capacity" in config.columns:
        positive_capacity = config[config["weekly_capacity"] > 0]
        if not positive_capacity.empty:
            config_row = positive_capacity.iloc[0]

    service = str(config_row.get("service", department))

    # The exported daily/weekly simulation files do not contain
    # service_time_mins. Recover it from the baseline capacity using
    # the project's baseline 6 staff / 6 rooms configuration. This keeps
    # the staffing scenario tied to the same service definition used by
    # the simulation (e.g. General OPD = 10 min/service).
    service_time = float(config_row.get("service_time_mins", 0) or 0)

    BASELINE_SERVERS = 6

    if service_time <= 0:
        baseline_capacity = None

        if level == "daily" and "daily_capacity" in config_row.index:
            baseline_capacity = float(config_row.get("daily_capacity", 0) or 0)
            raw_daily_24_7 = config_row.get("is_24_7", False)
            if isinstance(raw_daily_24_7, str):
                baseline_24_7 = raw_daily_24_7.strip().lower() in {"true", "1", "yes", "y"}
            else:
                baseline_24_7 = bool(raw_daily_24_7)
            if baseline_capacity > 0:
                if baseline_24_7:
                    minutes = 24 * 60
                else:
                    minutes = max(
                        parse_time(str(config_row.get("operating_end", "17:00")))
                        - parse_time(str(config_row.get("operating_start", "08:00"))),
                        0
                    )
                if minutes > 0:
                    service_time = BASELINE_SERVERS * minutes / baseline_capacity

        if service_time <= 0 and level == "weekly" and "weekly_capacity" in config_row.index:
            baseline_capacity = float(config_row.get("weekly_capacity", 0) or 0)
            raw_weekly_24_7 = config_row.get("is_24_7", False)
            if isinstance(raw_weekly_24_7, str):
                baseline_24_7 = raw_weekly_24_7.strip().lower() in {"true", "1", "yes", "y"}
            else:
                baseline_24_7 = bool(raw_weekly_24_7)
            if baseline_capacity > 0:
                if baseline_24_7:
                    minutes = 7 * 24 * 60
                else:
                    daily_minutes = max(
                        parse_time(str(config_row.get("operating_end", "17:00")))
                        - parse_time(str(config_row.get("operating_start", "08:00"))),
                        0
                    )
                    minutes = daily_minutes * 7
                if minutes > 0:
                    service_time = BASELINE_SERVERS * minutes / baseline_capacity

        if service_time <= 0 and level == "hourly" and "capacity" in config_row.index:
            baseline_capacity = float(config_row.get("capacity", 0) or 0)
            if baseline_capacity > 0:
                service_time = BASELINE_SERVERS * 60 / baseline_capacity

    if service_time <= 0:
        service_time = 30.0

    raw_24_7 = config_row.get("is_24_7", False)
    if isinstance(raw_24_7, str):
        is_24_7 = raw_24_7.strip().lower() in {"true", "1", "yes", "y"}
    else:
        is_24_7 = bool(raw_24_7)

    operating_start = str(config_row.get("operating_start", "08:00"))
    operating_end = str(config_row.get("operating_end", "17:00"))

    servers = max(1, min(int(staff), int(rooms)))

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def period_bounds(row):
        """Return start/end timestamps for the forecast period."""
        if level == "hourly":
            start = pd.Timestamp(row["date"])
            end = start + pd.Timedelta(hours=1)
        elif level == "daily":
            start = pd.Timestamp(row["date"]).normalize()
            end = start + pd.Timedelta(days=1)
        else:
            start = pd.Timestamp(row["week"]).normalize()
            end = start + pd.Timedelta(days=7)
        return start, end

    def operating_window(row):
        """Return the service window inside the forecast period."""
        period_start, period_end = period_bounds(row)

        if is_24_7:
            return period_start, period_end, True

        start_minutes = parse_time(operating_start)
        end_minutes = parse_time(operating_end)

        if level == "hourly":
            hour_start = period_start
            hour_end = period_end
            hour = period_start.hour
            current = hour * 60
            operating = start_minutes <= current < end_minutes
            return hour_start, hour_end, operating

        if level == "daily":
            service_start = period_start + pd.Timedelta(minutes=start_minutes)
            service_end = period_start + pd.Timedelta(minutes=end_minutes)
            if service_end <= service_start:
                return service_start, service_start, False
            return service_start, service_end, True

        # Weekly: apply the configured daily operating window to each
        # of the seven days. The weekly simulation therefore has seven
        # service windows rather than treating the entire week as one block.
        return period_start, period_end, True

    def weekly_service_windows(row):
        if is_24_7:
            start, end = period_bounds(row)
            return [(start, end)]

        week_start, _ = period_bounds(row)
        start_minutes = parse_time(operating_start)
        end_minutes = parse_time(operating_end)
        if end_minutes <= start_minutes:
            return []

        windows = []
        for day in range(7):
            day_start = week_start + pd.Timedelta(days=day)
            windows.append((
                day_start + pd.Timedelta(minutes=start_minutes),
                day_start + pd.Timedelta(minutes=end_minutes)
            ))
        return windows

    # --------------------------------------------------------
    # Sort forecast chronologically.
    # --------------------------------------------------------
    if level == "hourly":
        forecast = forecast.sort_values(["date", "hour"])
    else:
        forecast = forecast.sort_values(period_col)

    # Queue contains arrival timestamps. Server availability is represented
    # by timestamps for each parallel staff/room server.
    queue = []
    server_available = [pd.Timestamp.min] * servers

    results = []

    for _, row in forecast.iterrows():
        demand = max(float(row.get("demand", 0) or 0), 0.0)
        arrivals = int(ceil(demand))

        period_start, period_end = period_bounds(row)

        # Create deterministic arrival timestamps. This keeps the scenario
        # reproducible while avoiding the unrealistic assumption that all
        # forecast patients arrive at exactly the same instant.
        if level == "hourly":
            arrival_start = period_start
            arrival_end = period_end
            windows = [operating_window(row)[:2]] if operating_window(row)[2] else []
        elif level == "daily":
            day_start, day_end, operating = operating_window(row)
            arrival_start = day_start if operating else period_start
            arrival_end = day_end if operating else period_end
            windows = [(day_start, day_end)] if operating else []
        else:
            weekly_windows = weekly_service_windows(row)
            windows = weekly_windows
            arrival_start = period_start
            arrival_end = period_end

        if arrivals > 0:
            span_seconds = max((arrival_end - arrival_start).total_seconds(), 1.0)
            for i in range(arrivals):
                fraction = (i + 1) / (arrivals + 1)
                arrival_time = arrival_start + pd.Timedelta(seconds=span_seconds * fraction)
                queue.append(arrival_time)

        queue.sort()

        served_waits = []
        served_count = 0

        if level in {"hourly", "daily"}:
            service_windows = windows
        else:
            service_windows = windows

        # Process queued patients FCFS through the available staff/room servers.
        # A patient is only counted as served if the complete service finishes
        # inside the current period's operating window.
        remaining_queue = []

        for arrival_time in queue:
            best_server = None
            best_start = None
            best_finish = None

            for idx, available_at in enumerate(server_available):
                for window_start, window_end in service_windows:
                    service_start = max(arrival_time, available_at, window_start)
                    service_finish = service_start + pd.Timedelta(minutes=service_time)

                    if service_start >= window_start and service_finish <= window_end:
                        if best_start is None or service_start < best_start:
                            best_server = idx
                            best_start = service_start
                            best_finish = service_finish

            if best_server is not None:
                server_available[best_server] = best_finish
                served_count += 1
                served_waits.append(
                    max((best_start - arrival_time).total_seconds() / 60.0, 0.0)
                )
            else:
                remaining_queue.append(arrival_time)

        queue = remaining_queue

        # Capacity is the theoretical number of complete services possible
        # within the period under the selected staff/room scenario.
        if level == "hourly":
            operating_minutes = 60 if service_windows else 0
        elif level == "daily":
            operating_minutes = sum(
                max((end - start).total_seconds() / 60.0, 0)
                for start, end in service_windows
            )
        else:
            operating_minutes = sum(
                max((end - start).total_seconds() / 60.0, 0)
                for start, end in service_windows
            )

        capacity = servers * operating_minutes / service_time if service_time > 0 else 0
        capacity = max(capacity, 0.0)

        available_for_period = arrivals + len(queue) + served_count
        utilization = (
            available_for_period / capacity * 100
            if capacity > 0 else
            (100.0 if available_for_period > 0 else 0.0)
        )

        remaining = len(queue)
        avg_wait = sum(served_waits) / len(served_waits) if served_waits else 0.0
        max_wait = max(served_waits) if served_waits else 0.0

        # A period is closed only when its operating window is closed and
        # no patients could be served during it. Otherwise backlog drives
        # overload/congestion status.
        operating = bool(service_windows)
        status = get_scenario_status(
            utilization,
            remaining,
            operating
        )

        if level == "hourly":
            period_label = f"{period_start.strftime('%Y-%m-%d')} {period_start.hour:02d}:00"
        else:
            period_label = period_start.strftime("%Y-%m-%d")

        results.append({
            "period": period_label,
            "department": department,
            "service": service,
            "staff": int(staff),
            "rooms": int(rooms),
            "servers": int(servers),
            "service_time_mins": service_time,
            "operating": operating,
            "demand": demand,
            "simulated_arrivals": arrivals,
            "patients_served": served_count,
            "patients_remaining": remaining,
            "capacity": capacity,
            "utilization_pct": utilization,
            "average_wait_minutes": avg_wait,
            "maximum_wait_minutes": max_wait,
            "congestion_status": status
        })

    return results