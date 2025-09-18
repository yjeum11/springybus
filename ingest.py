import pandas as pd
import numpy as np
from datetime import datetime
from itertools import starmap

def parse_time(time_str):
    l = time_str.split(':')
    if len(l) != 3:
        raise ValueError("incorrect time string")
    h, m, s = l
    h = int(h)
    m = int(m)
    s = int(s)
    d = 0

    if h > 23:
        d += 1
        h -= 24

    return datetime(1900, 1, 1+d, h, m, s)

def import_data(minlon, minlat, maxlon, maxlat):
    # 10048010 is INBOUND, 10071010 is OUTBOUND

    trips_cols = ['trip_id', 'route_id', 'service_id', 'trip_headsign', 'direction_id']
    trips = pd.read_csv("./GTFS/trips.txt", usecols=trips_cols, skipinitialspace=True)

    # trips = trips[(trips.trip_id == 10048010) | (trips.trip_id == 10071010)]

    # Only use weekday service data
    trips = trips[ (trips.service_id == '2') \
                 | (trips.service_id == '3') \
                 | (trips.service_id == '6') \
                 | (trips.service_id == '7') ]

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'shape_dist_traveled', 'timepoint']
    stop_times = pd.read_csv("./GTFS/stop_times.txt", usecols=stop_times_cols, skipinitialspace=True)
    trip_times = pd.merge(trips, stop_times, on='trip_id')

    stops_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
    stops = pd.read_csv("./GTFS/stops.txt", usecols=stops_cols, skipinitialspace=True)
    df = pd.merge(trip_times, stops, on='stop_id')
    df.sort_values(by='stop_sequence', inplace=True, ignore_index=True)

    df = df[lambda x: (minlon < x['stop_lon']) & (x['stop_lon'] < maxlon) \
                    & (minlat < x['stop_lat']) & (x['stop_lat'] < maxlat)]

    return df

def avg_dispatch_interval(df, route, direction):
    # problem: only catches route whose endpoint is inside our legal range.
    # we just need the series of times at a specific bus stop across distinct trips.
    # this specific bus stop does not need to be the same across distinct routes.

    arrivals = df[(df.direction_id == direction) \
                 & (df.route_id == route)]
    print(arrivals)

    arrivals.to_csv('61A_dispatch.csv')

    # note that the stop_id column is of datatype str because the stop IDs are alphanumeric for some reason
    # maybe we can take a timestamp == 1 stop
    sample_stop = arrivals['stop_id'].min()

    arrivals = arrivals[arrivals.stop_id == sample_stop]

    t0 = datetime(1900, 1, 1)

    arrivals['arrival_time'] = arrivals['arrival_time'].map(parse_time).map(lambda x: (x - t0).total_seconds())

    arrivals.sort_values(by='arrival_time', inplace=True)
    print(arrivals)

    time_series = arrivals['arrival_time'].to_numpy()

    diffs = np.array(list( starmap((lambda a, b: b - a), zip(time_series[:-1], time_series[1:]))))

    return np.median(diffs)

# call this function outside in main.py in a loop
# in the loop we can loop over the common routes.
# Might need to keep track of nodes in a graph?
# So if there is an adjacency (i.e. stop_ids appear consecutive in any route) there is an edge
# we have to assume start_stop and end_stop are adjacent
def compute_length(df, start_stop, end_stop):
    pass

minlon = -79.96244
maxlon = -79.90803
maxlat = 40.45139
minlat = 40.42193
df = import_data(minlon, minlat, maxlon, maxlat)

print(avg_dispatch_interval(df, '61A', 1) / 60)

# return costs like (stop_id of src, stop_id of dest, target-length)
