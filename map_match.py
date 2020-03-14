# %%
import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, PReLU
from sklearn.model_selection import train_test_split

# Radius of Earth
R = 6371e3

columns_probes = ['sampleID', 'dateTime', 'sourceCode', 'latitude', 'longitude', 'altitude', 'speed', 'heading']
columns_links = ['linkPVID', 'refNodeID', 'nrefNodeID', 'length', 'functionalClass', 'directionOfTravel', 'speedCategory',
                 'fromRefSpeedLimit', 'toRefSpeedLimit', 'fromRefNumLanes', 'toRefNumLanes', 'multiDigitized', 'urban',
                 'timeZone', 'shapeInfo', 'curvatureInfo', 'slopeInfo']

df_probes = pd.read_pickle('probes_pickle.pkl')
df_links = pd.read_pickle('links_pickle.pkl')


# %%

# Opening the pickled probes coordinates
with open('probe_coord.pkl', 'rb') as f:
    probe_coord = pickle.load(f)


# %%
# Distinguishing the reference point and non-reference point of all links in two lists

link_shape = df_links['shapeInfo'].str.split('|')
ref_coord = [link_shape[i][0].split('/') for i in range(len(link_shape))]
nonref_coord = [link_shape[i][-1].split('/') for i in range(len(link_shape))]


# %%
# Replacing the elevation coordinate with 0 if its not present

for i in range(len(ref_coord)):
    for j in range(3):
        if ref_coord[i][j] == '':
            ref_coord[i][j] = '0'
ref_coord = list(map(lambda sl: list(map(float, sl)), ref_coord))

for i in range(len(nonref_coord)):
    for j in range(3):
        if nonref_coord[i][j] == '':
            nonref_coord[i][j] = '0'
nonref_coord = list(map(lambda sl: list(map(float, sl)), nonref_coord))


# %%
# Functions to get Bearing between two points, distance of a point from a Great Circle Path given by two points (reference and non reference in this case),
# and getting a list of great circle path distances for all the links from a certain point (probe coordinate).

def get_bearing(point_1, point_2):
    y = np.cos(point_2[0]) * np.sin(point_2[1] - point_1[1])
    x = np.cos(point_1[0]) * np.sin(point_2[0]) - np.sin(point_1[0]) * np.cos(point_2[0]) * np.cos(point_2[1] - point_1[1])
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def get_dist_from_path(point, ref, non_ref):
    ra_ref = np.radians(ref)
    ra_point = np.radians(point)

    angular_distance_13 = np.arccos(np.sin(ra_ref[0]) * np.sin(ra_point[0]) + np.cos(ra_ref[0]) * np.cos(ra_point[0]) * np.cos(abs(ra_point[1] - ra_ref[1])))
    theta_13 = get_bearing(ref, point)
    theta_12 = get_bearing(ref, non_ref)

    d = abs(np.arcsin(np.sin(angular_distance_13) * np.sin(theta_13 - theta_12)) * R)

    return d


def get_dist_list(point, ref, non_ref):
    dist_list = []
    for i in range(len(ref)):
        dist_list.append(get_dist_from_path(point, ref[i], non_ref[i]))
    return np.array(dist_list)


# %%
# Combining everything for probe matching for all probe points
N = 10

matched_links_indices = []
dist_from_ref = []
dist_from_link = []
p_index = 0

for sampleID in df_probes['sampleID'].unique():

    # Getting distance from all links for 1 probe point which will be used to calculate closest N links from it
    dist_list = get_dist_list(probe_coord[p_index], ref_coord, nonref_coord)

    # Getting the index of the closest N distances for 1 probe point which will remain constant for a Sample ID since
    # probe points should not go really far away from their recent links
    fake_dist_list = dist_list.copy()
    closest_n_ind = []
    for i in range(N):
        closest_n_ind.append(fake_dist_list.argmin())
        fake_dist_list[closest_n_ind[i]] = np.inf

    while (p_index < len(df_probes)) and (df_probes['sampleID'][p_index] == sampleID):

        # For each probe point of a sample ID use the heading and its distance from the N links calculated earlier
        # to create a selection criteria for the link
        headings = []
        updated_dist_list = []
        for close_index in closest_n_ind:
            ref = ref_coord[close_index]
            non_ref = nonref_coord[close_index]

            # Measuring point to great circle path distance
            updated_dist_list.append(get_dist_from_path(probe_coord[p_index], ref_coord[close_index], nonref_coord[close_index]))

            # Bearing Calculations from reference to non reference point on the link
            bearing = get_bearing(ref, non_ref)

            heading_factor = abs(df_probes['heading'][p_index] - bearing) / 720

            headings.append(heading_factor)

        headings = np.array(headings)

        # Creating the selection list
        selection_list = (updated_dist_list + headings) / 2

        # Selected link's index
        selected_link_index = closest_n_ind[selection_list.argmin()]

        # Info for the new dataframe to be formed as instructed in the readme
        d = get_dist_from_path(probe_coord[p_index], ref_coord[selected_link_index], nonref_coord[selected_link_index])

        # Calculating distance from the reference node to the map-matched probe point location on the link in decimal meters
        ra_ref = np.radians(ref_coord[selected_link_index])
        ra_point = np.radians(probe_coord[p_index])
        angular_distance = np.arccos(np.sin(ra_ref[0]) * np.sin(ra_point[0]) + np.cos(ra_ref[0]) * np.cos(ra_point[0]) * np.cos(abs(ra_point[1] - ra_ref[1])))

        along_track_d = np.arccos(np.cos(angular_distance) / np.cos(d / R)) * R

        matched_links_indices.append(selected_link_index)
        dist_from_ref.append(along_track_d)
        dist_from_link.append(d)

        p_index += 1


# %%
# Creating the matched points csv as described in the readme

matched_points_df = df_probes.copy()[:len(matched_links_indices)]
matched_points_df['linkPVID'] = df_links['linkPVID'][matched_links_indices].reset_index(drop=True)
matched_points_df['direction'] = df_links['directionOfTravel'][matched_links_indices].reset_index(drop=True)
matched_points_df = matched_points_df.replace('B', 'F')
matched_points_df['distFromRef'] = dist_from_ref
matched_points_df['distFromLink'] = dist_from_link

matched_points_df.to_csv('matched_points.csv', index=False, header=True)


# %%

df_matched_points = pd.read_csv('matched_points.csv')


# %%
df_avg_slope = df_links[['linkPVID']].copy()

# Distinguishing slopes and finding average slope of a link
link_slope = df_links['slopeInfo']
link_slope = link_slope.fillna('0/0|0/0').str.split('|')
avg_slope = np.array([(float(link_slope[i][0].split('/')[1]) + float(link_slope[i][1].split('/')[1])) / 2 for i in range(len(link_slope))])


# %%
# Replacing NAs with average of slope
avg_slope[avg_slope == 0] = avg_slope.mean(axis=0)
df_avg_slope['avg_slope'] = avg_slope


# %%
p_index = 0
X = []
Y = []

for sampleID in df_matched_points['sampleID'].unique():
    time = 0
    prev_alt = df_matched_points['altitude'][p_index]
    prev_speed = df_matched_points['speed'][p_index]

    while (p_index < len(df_matched_points) and (df_matched_points['sampleID'][p_index] == sampleID)):

        x = []

        alt_changed = df_matched_points['altitude'][p_index] - prev_alt
        prev_alt = df_matched_points['altitude'][p_index]

        spd_changed = df_matched_points['speed'][p_index] - prev_speed
        prev_speed = df_matched_points['speed'][p_index]

        distance = prev_speed * time
        time = 5

        # Adding these as features to X
        x.append(alt_changed)
        x.append(spd_changed)
        x.append(distance)
        x.append(float(df_links['length'][df_links['linkPVID'] == df_matched_points['linkPVID'][p_index]]))

        X.append(x)

        # Adding Y
        Y.append(round(float(df_avg_slope['avg_slope'][df_avg_slope['linkPVID'] == df_matched_points['linkPVID'][p_index]]), 5))

        p_index += 1

X = np.array(X)
Y = np.array(Y)


# %%
# Normalizing X
X = (X - X.mean(axis=0)) / X.std(axis=0)
print(X)


# %%
# Creating the NN model to predict average slope for the map matched probe points now
# Creating 80-20 split between Train and Test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

epochs = 1000


# %%
# NN Architecture creation
model = Sequential()
model.add(Dense(8, input_shape=x_train[0].shape))
model.add(PReLU())
model.add(Dense(16))
model.add(PReLU())
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, y_train, epochs=epochs)


# %%
# Adding predicted average slope and given average slope to the final csv
df_matched_points['predicted_avg_slope'] = model.predict(X)
df_matched_points['actual_avg_slope'] = Y

print(df_matched_points)


# %%
df_matched_points.to_csv('matched_points_final.csv', float_format='%g')
