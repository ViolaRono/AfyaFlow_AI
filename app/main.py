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


# Normalize department names consistently across every dataset.
# This is important because the source data contains values such as
# "GENERAL OPD " with a trailing space.
for df in datasets:
    if "department" in df.columns:
        df["department"] = (
            df["department"]
            .astype(str)
            .str.strip()
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
# SCENARIO CONFIGURATION
#
# The forecast resolution must not change the underlying service
# configuration.  The hourly baseline simulation contains the most
# direct evidence of service capacity, so use it to recover the
# configured service time and operating window for scenarios.
# ============================================================

def get_scenario_configuration(department: str):

    hourly = hourly_simulation[
        hourly_simulation["department"] == department
    ].copy()

    daily = daily_simulation[
        daily_simulation["department"] == department
    ].copy()

    source = hourly if not hourly.empty else daily

    if source.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation configuration for {department}"
        )

    first = source.iloc[0]

    service = str(first.get("service", department))

    # Recover service time from the baseline hourly capacity where
    # possible.  The baseline simulations were configured with six
    # staff and six rooms, so:
    #     service_time = 6 * 60 / hourly_capacity
    service_time = None

    if "capacity" in hourly.columns:
        hourly_capacity = pd.to_numeric(
            hourly["capacity"],
            errors="coerce"
        )

        operating_rows = hourly_capacity > 0
        if "operating" in hourly.columns:
            operating_rows = operating_rows & hourly["operating"].apply(
                lambda value: str(value).strip().lower()
                in {"true", "1", "yes", "y"}
            )

        valid_capacity = hourly_capacity[operating_rows]

        if not valid_capacity.empty and valid_capacity.max() > 0:
            service_time = (
                6 * 60 / float(valid_capacity.max())
            )

    # Fallback to daily baseline capacity.  Non-24/7 departments use
    # the configured nine-hour daytime window in the baseline model.
    if service_time is None and "daily_capacity" in daily.columns:
        daily_capacity = pd.to_numeric(
            daily["daily_capacity"],
            errors="coerce"
        ).dropna()

        positive = daily_capacity[daily_capacity > 0]

        if not positive.empty:
            service_time = (
                6 * 9 * 60 / float(positive.iloc[0])
            )

    if service_time is None or service_time <= 0:
        service_time = 30.0

    raw_24_7 = first.get("is_24_7", False)

    if isinstance(raw_24_7, str):
        is_24_7 = raw_24_7.strip().lower() in {
            "true", "1", "yes", "y"
        }
    else:
        is_24_7 = bool(raw_24_7)

    operating_start = None
    operating_end = None

    # Prefer explicitly stored operating times.
    for column, target in [
        ("operating_start", "start"),
        ("operating_end", "end")
    ]:
        if column in hourly.columns:
            values = hourly[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            if not values.empty:
                if target == "start":
                    operating_start = values.iloc[0]
                else:
                    operating_end = values.iloc[0]

    # Otherwise infer the window from hourly operating flags.
    if (
        not is_24_7
        and (operating_start is None or operating_end is None)
        and "hour" in hourly.columns
        and "operating" in hourly.columns
    ):
        operating_mask = hourly["operating"].apply(
            lambda value: str(value).strip().lower()
            in {"true", "1", "yes", "y"}
        )
        operating_hours = pd.to_numeric(
            hourly.loc[operating_mask, "hour"],
            errors="coerce"
        ).dropna()

        if not operating_hours.empty:
            start_hour = int(operating_hours.min())
            end_hour = int(operating_hours.max()) + 1
            operating_start = f"{start_hour:02d}:00"
            operating_end = f"{end_hour:02d}:00"

    operating_start = operating_start or "08:00"
    operating_end = operating_end or "17:00"

    return {
        "service": service,
        "service_time_mins": float(service_time),
        "is_24_7": is_24_7,
        "operating_start": operating_start,
        "operating_end": operating_end
    }


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

    level = level.lower().strip()

    if level not in {
        "hourly",
        "daily",
        "weekly"
    }:

        raise HTTPException(
            status_code=400,
            detail="level must be hourly, daily, or weekly"
        )

    # --------------------------------------------------------
    # Find department forecast
    # --------------------------------------------------------

    department = str(department).strip()

    if level == "hourly":
        forecast = hourly_forecast[
            hourly_forecast["department"] == department
        ].copy()

    elif level == "daily":
        forecast = daily_forecast[
            daily_forecast["department"] == department
        ].copy()

    else:
        forecast = weekly_forecast[
            weekly_forecast["department"] == department
        ].copy()

    if forecast.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {level} forecast for {department}"
        )

    # --------------------------------------------------------
    # Use one service configuration across all resolutions.
    # This prevents stale weekly simulation metadata from changing
    # the service time used by the staffing scenario.
    # --------------------------------------------------------

    scenario_config = get_scenario_configuration(department)

    service = scenario_config["service"]
    service_time = scenario_config["service_time_mins"]
    is_24_7 = scenario_config["is_24_7"]
    operating_start = scenario_config["operating_start"]
    operating_end = scenario_config["operating_end"]

    # Staff and rooms act as parallel constraints.
    servers = min(
        staff,
        rooms
    )

    # --------------------------------------------------------
    # Prepare forecast
    # --------------------------------------------------------

    if level == "hourly":

        forecast = forecast.sort_values(
            ["date", "hour"]
        )

    elif level == "daily":

        forecast = forecast.sort_values(
            "date"
        )

    else:

        forecast = forecast.sort_values(
            "week"
        )

    # --------------------------------------------------------
    # Persistent queue
    # --------------------------------------------------------

    backlog = 0

    results = []

    for _, row in forecast.iterrows():

        raw_demand = row.get("demand", None)

        if raw_demand is None or pd.isna(raw_demand):
            raw_demand = row.get("forecast_demand", None)
        if raw_demand is None or pd.isna(raw_demand):
            raw_demand = row.get("predicted_demand", None)
        if raw_demand is None or pd.isna(raw_demand):
            raw_demand = row.get("ensemble_prediction", 0)

        demand = float(raw_demand)

        demand = max(
            demand,
            0
        )

        # Forecast remains continuous.
        #
        # Simulation needs discrete patients.
        #
        arrivals = ceil(
            demand
        )

        # ----------------------------------------------------
        # Determine operating state
        # ----------------------------------------------------

        if level == "hourly":

            if is_24_7:

                operating = True

            else:

                hour = int(
                    row["hour"]
                )

                start = parse_time(
                    operating_start
                )

                end = parse_time(
                    operating_end
                )

                current_minutes = (
                    hour * 60
                )

                operating = (
                    start
                    <= current_minutes
                    < end
                )

            period_label = (
                f"{row['date'].isoformat()} "
                f"{int(row['hour']):02d}:00"
            )

        elif level == "daily":

            operating = True

            period_label = (
                pd.to_datetime(
                    row["date"]
                ).strftime(
                    "%Y-%m-%d"
                )
            )

        else:

            operating = True

            period_label = (
                pd.to_datetime(
                    row["week"]
                ).strftime(
                    "%Y-%m-%d"
                )
            )

        # ----------------------------------------------------
        # Calculate available service time
        # ----------------------------------------------------

        if level == "hourly":

            if operating:

                capacity = (
                    servers
                    * 60
                    / service_time
                )

            else:

                capacity = 0

        elif level == "daily":

            if is_24_7:

                operating_minutes = 24 * 60

            else:

                start = parse_time(
                    operating_start
                )

                end = parse_time(
                    operating_end
                )

                operating_minutes = max(
                    end - start,
                    0
                )

            capacity = (
                servers
                * operating_minutes
                / service_time
            )

        else:

            if is_24_7:

                daily_minutes = 24 * 60

            else:

                start = parse_time(
                    operating_start
                )

                end = parse_time(
                    operating_end
                )

                daily_minutes = max(
                    end - start,
                    0
                )

            weekly_minutes = (
                daily_minutes * 7
            )

            capacity = (
                servers
                * weekly_minutes
                / service_time
            )

        capacity = max(
            capacity,
            0
        )

        # ----------------------------------------------------
        # Queue before service
        # ----------------------------------------------------

        available_patients = (
            backlog + arrivals
        )

        # Only complete patient services
        # are counted as served.
        service_slots = floor(
            capacity
        )

        if operating:

            patients_served = min(
                available_patients,
                service_slots
            )

        else:

            patients_served = 0

        # ----------------------------------------------------
        # Remaining queue
        # ----------------------------------------------------

        patients_remaining = max(
            available_patients
            - patients_served,
            0
        )

        # ----------------------------------------------------
        # Utilization
        #
        # Includes patients waiting from previous
        # periods, which is important for congestion.
        # ----------------------------------------------------

        if capacity > 0:

            utilization = (
                available_patients
                / capacity
                * 100
            )

        else:

            utilization = (
                100
                if available_patients > 0
                else 0
            )

        status = get_scenario_status(
            utilization,
            patients_remaining,
            operating
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {
            "period": period_label,
            "department": department,
            "service": service,
            "staff": staff,
            "rooms": rooms,
            "servers": servers,
            "service_time_mins": service_time,
            "operating": operating,
            "demand": demand,
            "simulated_arrivals": arrivals,
            "patients_served": patients_served,
            "patients_remaining": patients_remaining,
            "capacity": capacity,
            "utilization_pct": utilization,
            "congestion_status": status
        }

        results.append(
            result
        )

        # Carry backlog into next period.
        backlog = patients_remaining

    return results