import laspy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import time
import pandas as pd


class Point_Cloud:
    # object that stores the points in a point cloud, initialized from a las file
    # points can be enabled or disabled, disabled points are ignored when searching for stems
    def __init__(self, file_name, verbose=False):
        all_data = laspy.read(file_name)
        height_of_points_above_ground = np.array(
            [point[-1] for point in all_data.points.array])
        self.points = [Point(xyz, height) for xyz, height in zip(
            all_data.xyz, height_of_points_above_ground)]
        # internal variable, dont call it directly
        self.__enabled_points__ = self.points
        self.verbose = verbose

    def enabled_points(self):
        # filters the set of points for just those enabled, then resturns that set of enabled points
        self.__enabled_points__ = filter_for_enabled(self.__enabled_points__)
        return self.__enabled_points__

    def find_highest_point(self):
        # returns the point in the point cloud which has the greatest z coordinate
        return max(self.enabled_points(), key=lambda point: point.z())

    def find_stem(self, min_height=.5, fail_to_find_disable_radius=.2):
        # returns a stem from the point cloud if it exists, otherwise returns None
        # the stem will be a list of points in descending z value, starting from the min height and going down to a ground point
        stem_grounded = False
        while not(stem_grounded):
            # the stem starts at the highest point in the cloud
            stem = [self.find_highest_point()]
            if stem[0].height_above_ground < min_height:
                # if your stem is too short, we're done
                break
            while not(stem_grounded) and stem:
                # we try to find a point in a cylinder below this stem
                active_point = stem[-1]
                descendents = active_point.find_descendents(
                    self.enabled_points())
                if descendents:
                    # if there is a valid descendent, take one at random (we can try again later if it doesnt work)
                    next_point = random.choice(descendents)
                    stem.append(next_point)
                    if next_point.is_ground:
                        stem_grounded = True
                else:
                    # if the point has no descendent, it cannot have a path to the ground, so we disable it and all points within fail_to_find_disable_radius of it
                    n_points_start = len(self.enabled_points())
                    self.disable_stem_region(
                        [active_point], fail_to_find_disable_radius)
                    n_points_end = len(self.enabled_points())
                    if self.verbose:
                        print(
                            f"We exploded {n_points_start-n_points_end} points!")
                    while stem and not stem[-1].enabled:
                        stem.pop()
        if stem_grounded:
            # we've found a stem
            if self.verbose:
                print("Found a stem!")
            return stem
        else:
            # if the point cloud cannot find a new stem, we stop
            return None

    def disable_stem_region(self, stem, radius_to_delete=.7):
        # disables all points in the point cloud within radius_to_delete distance of a point in the stem
        squared_radius_to_delete = radius_to_delete**2
        for enabled_point in self.enabled_points():
            if any([squared_distance(enabled_point, stem_point) < squared_radius_to_delete for stem_point in stem]):
                enabled_point.enabled = False

    def plot(self, stem=None, disabled_color='green', theta=0):
        # displays the point cloud
        # inputs:
        #   - stem : a set of points to highlight as the stem
        #   - disabled_color : the color to draw the disabled points as. If None, will not draw them
        #   - theta : an angle (in degrees), which rotates the picture
        plt.close()
        enabled_point_cloud = self.enabled_points()
        sin = math.sin(theta*180/math.pi)
        cos = math.cos(theta*180/math.pi)
        x = [point.xyz[0]*cos+point.xyz[1]*sin for point in enabled_point_cloud]
        z = [point.xyz[2] for point in enabled_point_cloud]
        plt.scatter(x=x, y=z, color='blue')
        if disabled_color:
            disabled_point_cloud = filter_for_disabled(self.points)
            x = [point.xyz[0]*cos+point.xyz[1] *
                 sin for point in disabled_point_cloud]
            z = [point.xyz[2] for point in disabled_point_cloud]
            plt.scatter(x=x, y=z, color='green')
        if stem:
            x_stem = [stem_point.xyz[0]*cos +
                      stem_point.xyz[1]*sin for stem_point in stem]
            z_stem = [stem_point.xyz[2] for stem_point in stem]
            plt.scatter(x=x_stem, y=z_stem, color='red')

    def save_via_dataframe(self, file_name="point_clustering_output.csv", folder_name="cluster_csvs/"):
        pre_df = [list(point.xyz) + [point.height_above_ground,
                                     point.cluster] for point in self.points]
        df = pd.DataFrame(
            pre_df, columns=["x", "y", "z", "height_above_ground", "cluster"])

        out_file_name = f"{folder_name}{file_name}"
        with open(out_file_name, 'w') as f:
            df.to_csv(f)


class Point:
    # a point in the point cloud

    def __init__(self, xyz, height_above_ground, ):
        self.xyz = xyz
        self.height_above_ground = height_above_ground
        self.is_ground = height_above_ground < 0
        self.enabled = True
        self.descendents = None
        self.cluster = -1

    def __repr__(self):
        return str(list(self.xyz)+[self.height_above_ground, self.enabled])

    def find_descendents(self, points_to_check, horizontal_radius=.2, height_above_self=0, height_below_self=0.3):
        # finds enabled points in a cylinder below self
        if not self.descendents:
            self.descendents = self.find_points_in_cylinder(
                points_to_check, horizontal_radius=horizontal_radius, height_above_self=height_above_self, height_below_self=height_below_self)
        self.descendents = filter_for_enabled(self.descendents)
        return self.descendents

    def find_points_in_cylinder(self, points_to_check, horizontal_radius=.2, height_above_self=0, height_below_self=0.3):
        # finds points in a cylinder around self
        points_to_check = filter(
            lambda point: -1*height_below_self < point.z()-self.z(), points_to_check)
        points_to_check = list(
            filter(lambda point: height_above_self > point.z()-self.z(), points_to_check))
        squared_distances = np.zeros(len(points_to_check))
        for dim in range(2):
            distance_on_this_dimension = self.xyz[dim]-np.array(
                [point.xyz[dim] for point in points_to_check])
            squared_distances += distance_on_this_dimension**2
        points_in_cylinder = [point for point, squared_distance in zip(
            points_to_check, squared_distances) if squared_distance < horizontal_radius**2]
        return points_in_cylinder

    def z(self):
        return self.xyz[2]


def filter_for_enabled(points_list):
    return [point for point in points_list if point.enabled]


def filter_for_disabled(points_list):
    return [point for point in points_list if not point.enabled]


def filter_for_clustered(points_list):
    return [point for point in points_list if point.cluster >= 0]


def filter_for_clustered_non_ground(points_list):
    return [point for point in points_list if point.cluster > 0]


def filter_for_unclustered(points_list):
    return [point for point in points_list if point.cluster < 0]


def squared_horizontal_distance(point1, point2):
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2


def squared_distance(point1, point2):
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2+(point1.xyz[2]-point2.xyz[2])**2


def downsample(arr, fraction):
    np.random.shuffle(arr)
    cutoff_point = int(arr.shape[0]*fraction)
    return arr[:cutoff_point]


def assign_clusters_by_nearest(point_cloud, stems, height_cutoff=0.2):
    # assigns every point in the point cloud to its closest stem (as measured by horizontal distance to stem midpoint)
    # points below height_cutoff are sent to a separate cluster 0 for the ground
    stem_centers = [
        Point(sum([point.xyz for point in stem])/len(stem), 0) for stem in stems]
    for point in point_cloud.points:
        if point.height_above_ground < height_cutoff:
            point.cluster = 0
        else:
            squared_horizontal_distance_to_stem_centers = np.array(
                [squared_horizontal_distance(point, stem_center) for stem_center in stem_centers])
            point.cluster = 1 + \
                np.argmin(squared_horizontal_distance_to_stem_centers)


def assign_clusters_by_growing(point_cloud, stems, grow_radius=.2, grow_height=.4, height_cutoff=0.2):
    # assigns points to clusters based on "growing" the stem
    # points below height_cutoff are sent to a separate cluster 0 for the ground
    for point in point_cloud.points:
        if point.height_above_ground < height_cutoff:
            point.cluster = 0
    for i, stem in enumerate(stems):
        growing_points = [stem_point for stem_point in stem]
        while growing_points:
            unclustered_points = filter_for_unclustered(point_cloud.points)
            point_to_grow = growing_points.pop()
            upwards_points = point_to_grow.find_points_in_cylinder(
                unclustered_points, horizontal_radius=grow_radius, height_above_self=grow_height, height_below_self=0)
            if point_cloud.verbose:
                print(
                    f"Found {len(upwards_points)} new points for cluster {i+1}/{len(stems)}")
            for upward_point in upwards_points:
                upward_point.cluster = i+1
                growing_points.append(upward_point)
        print(f"Finished first pass at assigning points to cluster {i+1}")
    # if any points are not assigned to a cluster, this method finishes the process:
    print("Proceeding to cluster remaining points!")
    cluster_remaining_points_to_nearest_neighbor(point_cloud)


def cluster_remaining_points_to_nearest_neighbor(point_cloud):
    # assigns points to existing clusters based on which cluster it is currently closest to
    out_of_cluster_points = filter_for_unclustered(point_cloud.points)
    in_cluster_points = filter_for_clustered_non_ground(point_cloud.points)
    for point in out_of_cluster_points:
        nearest_point = min(in_cluster_points,
                            key=lambda x: squared_distance(x, point))
        point.cluster = nearest_point.cluster


def plot_stem_centers(point_cloud, stems, include_ground=True, save_title=None, show_unclustered=False):
    # visualization method
    plt.close()

    stem_centers = [
        Point(sum([point.xyz for point in stem])/len(stem), 0) for stem in stems]

    colors = ['blue', 'green', 'purple', 'yellow',
              'black', 'orange', 'cyan', 'maroon', 'brown']
    num_colors = len(colors)

    stem_center_x = [stem_center.xyz[0] for stem_center in stem_centers]
    stem_center_y = [stem_center.xyz[1] for stem_center in stem_centers]
    plt.scatter(x=stem_center_x, y=stem_center_y,
                color='red', marker='x', label='Stem Centers')

    for i in range(len(stem_centers)):
        neighborhood = [
            point for point in point_cloud.points if point.cluster == i+1]
        x = [point.xyz[0] for point in neighborhood]
        y = [point.xyz[1] for point in neighborhood]
        plt.scatter(
            x=x, y=y, color=colors[i % num_colors], alpha=.1, label=f'Tree {i+1}')
    if show_unclustered:
        neighborhood = [
            point for point in point_cloud.points if point.cluster == -1]
        x = [point.xyz[0] for point in neighborhood]
        y = [point.xyz[1] for point in neighborhood]
        plt.scatter(x=x, y=y, color='grey', alpha=.1,
                    label=f'Unclustered Points')

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.title(save_title)
    if save_title:
        plt.savefig(fname=f"saved_images/{save_title}")
    return


if __name__ == "__main__":
    file_names = [
        # "Test_data/treeID_12210.las", # 1 stem
        # "Test_data/treeID_19707.las", # 1 stem
        # "Test_data/treeID_33009.las", # 1 stem
        # "Test_data/treeID_34926_merged.las", # >10 stems
        # "Test_data/treeID_35618_merged.las", # >10 stems
        # "Test_data/treeID_40038_merged.las", # 2 stems
        # "Test_data/treeID_40061_merged.las", # 13 stems
        # "Test_data/treeID_40113_merged.las", # 3 stems
        "Test_data/treeID_40645_merged.las",  # 2 stems
        # "Test_data/treeID_40803_merged.las", # 9ish stems
        # "Test_data/treeID_42113_merged.las", # 11 stems
    ]

    for file_name in file_names:
        random.seed(42)
        point_cloud = Point_Cloud(file_name)

        # point_cloud.plot()
        overall_t_start = time.time()
        stems = []
        stem = True
        while stem:
            t_start = time.time()
            stem = point_cloud.find_stem()
            t_end = time.time()
            if stem:
                stems.append(stem)
                print(
                    f"I found the {len(stems)}th stem in {t_end-t_start:.2f} seconds!")
                # point_cloud.plot(stem=stem, disabled_color='green', theta=0)
                point_cloud.disable_stem_region(stem)
            else:
                print(
                    f"I found that there were no more stems in {t_end-t_start:.2f} seconds!")
        # assign_clusters_by_nearest(point_cloud,stems)
        assign_clusters_by_growing(point_cloud, stems)

        point_cloud.save_via_dataframe()
        save_file_name = file_name.split(
            ".")[0].split("/")[1]+"_stem_centers.png"
        # plot_stem_centers(point_cloud, stems=stems,
        #                   include_ground=False, save_title=save_file_name)
        overall_t_end = time.time()
        print(
            f"Done! Found {len(stems)} stems in {overall_t_end-overall_t_start:.2f} seconds!")
    print("foo")
