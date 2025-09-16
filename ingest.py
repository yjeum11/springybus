import pandas as pd

def import_data():
    route = "61A"

    trips_cols = ['trip_id', 'route_id', 'trip_headsign', 'direction_id']
    trips = pd.read_csv("./GTFS/trips.txt", usecols=trips_cols, skipinitialspace=True)
    trips = trips[trips.trip_id == 10048010]

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'shape_dist_traveled', 'timepoint']
    stop_times = pd.read_csv("./GTFS/stop_times.txt", usecols=stop_times_cols, skipinitialspace=True)
    trip_times = pd.merge(trips, stop_times, on='trip_id')

    stops_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
    stops = pd.read_csv("./GTFS/stops.txt", usecols=stops_cols, skipinitialspace=True)
    df = pd.merge(trip_times, stops, on='stop_id')
    df.sort_values(by='stop_sequence', inplace=True, ignore_index=True)

    return df

def calculate_costs():
    pass
