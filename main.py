import ingest

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.transform import PiecewiseAffineTransform, ThinPlateSplineTransform
import pandas as pd

from PIL import Image
from datetime import datetime

TIME_TO_LENGTH = 8e-5
NUM_ITERS = 500
K_SPRING = 0.01
K_GEO = 0.005
K_REPULSION = 2e-12
K_DAMPING = 0.7


df = ingest.import_data()

# ---------------------- import image
minlon = -79.96244
maxlon = -79.90803
maxlat = 40.45139
minlat = 40.42193
rangelat = maxlat - minlat
rangelon = maxlon - minlon

lon2lat = rangelon/rangelat
image = np.array(Image.open("./mappic.png"))
image_rows, image_cols, numChannels = np.shape(image)
fig, (ax, ax2, ax3) = plt.subplots(3, 1)
ax.imshow(image, extent=(minlon, maxlon, minlat, maxlat))

legalpoints = df[lambda x: (minlon < x['stop_lon']) & (x['stop_lon'] < maxlon) \
                         & (minlat < x['stop_lat']) & (x['stop_lat'] < maxlat)]\
                [['stop_lon', 'stop_lat']].to_numpy()

timesteps   = df[lambda x: (minlon < x['stop_lon']) & (x['stop_lon'] < maxlon) \
                         & (minlat < x['stop_lat']) & (x['stop_lat'] < maxlat)]\
                [['arrival_time']].to_numpy().T[0]
timesteps   = np.array([datetime.strptime(x, "%H:%M:%S") for x in timesteps])

testing     = df[lambda x: (minlon < x['stop_lon']) & (x['stop_lon'] < maxlon) \
                         & (minlat < x['stop_lat']) & (x['stop_lat'] < maxlat)]\
                [['stop_name', 'arrival_time']]

num_points = len(legalpoints)
positions = legalpoints.copy()
velocities = np.zeros_like(legalpoints)
forces = np.zeros_like(legalpoints)

target_lengths = np.array([0.0 for _ in legalpoints])
for i in range(num_points-1):
    j = i + 1
    t1 = timesteps[i]
    t2 = timesteps[j]

    delta_t = (t2 - t1).total_seconds()

    # set goal length of the springs to be proportional to delta_t
    target_lengths[i] = TIME_TO_LENGTH * delta_t

# for _ in range(3):
    # for i, (p1, p2, t1, t2) in enumerate(zip(legalpoints[:-1], legalpoints[1:], timesteps[:-1], timesteps[1:])):
for _ in range(NUM_ITERS):
    for i in range(num_points-1):
        j = i + 1
        t1 = timesteps[i]
        t2 = timesteps[j]
        p1 = positions[i]
        p2 = positions[j]

        delta_t = (t2 - t1).total_seconds()

        # set goal length of the springs to be proportional to delta_t
        x = p2 - p1
        dist = max(1e-10, np.linalg.norm(x))
        F = K_SPRING * (dist - target_lengths[i]) * (x / dist)
        forces[i] += F
        forces[j] -= F

    for i in range(num_points):
        delta_geo = legalpoints[i] - positions[i]
        forces[i] += K_GEO * delta_geo

    # Optional repulsion
    for i in range(num_points):
        for j in range(i+1, num_points):
            delta = positions[i] - positions[j]
            dist = np.linalg.norm(delta) + 1e-6
            F = K_REPULSION / dist**2 * (delta / dist)
            forces[i] += F
            forces[j] -= F

    velocities = K_DAMPING * velocities + forces * 0.001
    positions += velocities

unhappiness = np.zeros_like(target_lengths)
dists = np.zeros_like(target_lengths)
distso = np.zeros_like(target_lengths)
for i in range(num_points-1):
    j = i+1
    p1 = positions[i]
    p2 = positions[j]
    x = p2 - p1

    p1o = legalpoints[i]
    p2o = legalpoints[j]
    xo = p2o - p1o

    dists[i] = max(1e-10, np.linalg.norm(x))
    distso[i] = max(1e-10, np.linalg.norm(xo))
    unhappiness[i] = dists[i] - target_lengths[i]

normalized_unhappiness = 0.5*np.clip(unhappiness/0.0027, min=-1.0, max=1.0) + 0.5

# unhappiness = 2.*(unhappiness - np.min(unhappiness))/np.ptp(unhappiness)-1
print(f"{unhappiness=}")
print(f"{normalized_unhappiness=}")
print(f"{dists=}")
print(f"{distso=}")
print(f"{target_lengths=}")
print(np.mean(abs(unhappiness)))
print(np.mean(abs(target_lengths)))

tform = ThinPlateSplineTransform()

def latlon_to_xy(lat, lon, lat_min, lat_max, lon_min, lon_max, width, height):
    """Map lat/lon into [0,width]x[0,height] screen coords"""
    x = (lon - lon_min) / (lon_max - lon_min) * width
    y = (1 - (lat - lat_min) / (lat_max - lat_min)) * height
    return np.array([x, y])

def xy_to_latlon(x, y, lat_min, lat_max, lon_min, lon_max, width, height):
    """Inverse mapping: screen coords back to lat/lon"""
    lon = x / width * (lon_max - lon_min) + lon_min
    lat = (1 - y / height) * (lat_max - lat_min) + lat_min
    return np.array([lon, lat])

corners = np.array([
    [0, 0],
    [image_cols-1, 0],
    [image_cols-1, image_rows-1],
    [0, image_rows-1]
])
legalpoints_img = np.vstack([np.array([
    latlon_to_xy(lat, lon, minlat, maxlat, minlon, maxlon, image_cols, image_rows) for lon, lat in legalpoints
]), corners])

positions_img = np.vstack([np.array([
    latlon_to_xy(lat, lon, minlat, maxlat, minlon, maxlon, image_cols, image_rows) for lon, lat in positions
]), corners])

res = tform.estimate(positions_img, legalpoints_img)
print(res)

out = ski.transform.warp(image, tform, output_shape=(image_rows, image_cols))
ax2.imshow(out, extent=(minlon, maxlon, minlat, maxlat))
ax3.imshow(out, extent=(minlon, maxlon, minlat, maxlat))
ax2.scatter(legalpoints[:, 0], legalpoints[:, 1])
ax2.scatter(positions[:, 0], positions[:, 1])

cmap = plt.cm.seismic

for i, ((x0, y0), (x1, y1)) in enumerate(zip(positions[:-1], positions[1:])):
    ax2.plot([x0, x1], [y0, y1], color=cmap(normalized_unhappiness[i]))  # 'k-' means black line

# plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label="value", ax=ax)

# ax.scatter(legalpoints[:, 0], legalpoints[:, 1])
# ax.scatter(positions[:, 0], positions[:, 1])
# for (x0, y0), (x1, y1) in zip(legalpoints, positions):
#     ax.plot([x0, x1], [y0, y1], 'k-')  # 'k-' means black line
# ax.plot()
# plt.axis('equal')
plt.show()

