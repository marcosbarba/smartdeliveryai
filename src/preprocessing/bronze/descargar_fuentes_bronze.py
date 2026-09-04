import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
ROUTE_DATA = (
    ROOT_DIR
    / "data"
    / "raw_amazon"
    / "02 Dataset Original Amazon Last Mile"
    / "Dataset Entrenamiento 2021"
    / "Model Build Inputs"
    / "route_data.json"
)

BRONZE_DIR = ROOT_DIR / "data" / "bronze"
WEATHER_DIR = BRONZE_DIR / "Weather Open Meteo" / "Raw"
CALENDAR_DIR = BRONZE_DIR / "Calendar Nager" / "Raw"
MANIFEST_DIR = BRONZE_DIR / "Source Manifests"

START_DATE = "2018-07-19"
END_DATE = "2018-08-26"


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SmartDeliveryAI-TFM/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def station_coordinates(route_data):
    coords_by_station = {}
    route_counts = {}

    for route in route_data.values():
        station_code = route["station_code"]
        route_counts[station_code] = route_counts.get(station_code, 0) + 1

        station_stops = [
            (stop["lat"], stop["lng"])
            for stop in route["stops"].values()
            if stop.get("type") == "Station"
        ]
        coords_by_station.setdefault(station_code, []).extend(station_stops)

    stations = []
    for station_code, coords in sorted(coords_by_station.items()):
        lat = sum(item[0] for item in coords) / len(coords)
        lng = sum(item[1] for item in coords) / len(coords)
        stations.append(
            {
                "station_code": station_code,
                "route_count": route_counts.get(station_code, 0),
                "latitude": round(lat, 6),
                "longitude": round(lng, 6),
            }
        )
    return stations


def open_meteo_url(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "weather_code",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
            ]
        ),
        "timezone": "America/Los_Angeles",
    }
    return "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)


def download_weather(stations):
    downloaded = []
    for station in stations:
        url = open_meteo_url(station["latitude"], station["longitude"])
        data = fetch_json(url)
        payload = {
            "source": "Open-Meteo Historical Weather API",
            "source_url": url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "station": station,
            "raw_response": data,
        }
        output = WEATHER_DIR / f"Meteorologia {station['station_code']}.json"
        write_json(output, payload)
        downloaded.append({"station_code": station["station_code"], "path": output.relative_to(ROOT_DIR).as_posix()})
        time.sleep(0.2)
    return downloaded


def download_holidays():
    url = "https://date.nager.at/api/v3/PublicHolidays/2018/US"
    data = fetch_json(url)
    payload = {
        "source": "Nager.Date Public Holiday API",
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "country": "US",
        "year": 2018,
        "raw_response": data,
    }
    output = CALENDAR_DIR / "Festivos EEUU 2018.json"
    write_json(output, payload)
    return {"path": output.relative_to(ROOT_DIR).as_posix(), "records": len(data)}


def main():
    route_data = read_json(ROUTE_DATA)
    dates = sorted({route["date_YYYY_MM_DD"] for route in route_data.values()})
    stations = station_coordinates(route_data)

    write_json(
        MANIFEST_DIR / "Coordenadas Estaciones Amazon.json",
        {
            "description": "Station coordinates derived from Amazon route_data Station stops. Used only to request external data.",
            "source_file": ROUTE_DATA.relative_to(ROOT_DIR).as_posix(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "date_min": min(dates),
            "date_max": max(dates),
            "route_count": len(route_data),
            "station_count": len(stations),
            "stations": stations,
        },
    )

    holidays = download_holidays()
    weather = download_weather(stations)

    write_json(
        MANIFEST_DIR / "Manifiesto Fuentes Bronze.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Raw Bronze external sources for future enrichment of Amazon Last Mile routes.",
            "rules": [
                "Bronze stores original source responses.",
                "Do not join or transform external sources in Bronze.",
                "Clean each source separately in Silver.",
                "Join Amazon, weather and calendar features in Gold.",
            ],
            "amazon_training_routes": {
                "source_file": ROUTE_DATA.relative_to(ROOT_DIR).as_posix(),
                "route_count": len(route_data),
                "date_min": min(dates),
                "date_max": max(dates),
                "station_count": len(stations),
            },
            "external_sources": {
                "Calendar Nager": holidays,
                "Weather Open Meteo": weather,
            },
            "reliable_sources": [
                {
                    "name": "Open-Meteo Historical Weather API",
                    "url": "https://open-meteo.com/en/docs/historical-weather-api",
                    "reason": "Provides historical weather data using reanalysis/weather model datasets without API key.",
                },
                {
                    "name": "Nager.Date Public Holiday API",
                    "url": "https://date.nager.at/api",
                    "reason": "Open holiday API with country and subdivision holiday information.",
                },
            ],
        },
    )

    print("Bronze external sources downloaded.")
    print(f"Stations: {len(stations)}")
    print(f"Weather files: {len(weather)}")
    print(f"Holiday records: {holidays['records']}")


if __name__ == "__main__":
    main()
