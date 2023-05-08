import laspy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import time
import pandas as pd
import argparse
import re
import os

class Point_Cloud:
    # object that stores the points in a point cloud, initialized from a las file
    # points can be enabled or disabled, disabled points are ignored when searching for stems
    def __init__(self, file_name, verbose=False):
        all_data = laspy.read(file_name)
        height_of_points_above_ground = np.array(
            [point[-1] for point in all_data.points.array])
        self.points = [Point(xyz, height) for xyz, height in zip(
            all_data.xyz, height_of_points_above_ground)]
        self.__enabled_points__ = self.points # internal, to access this use enabled_points()
        self.verbose = verbose

    def enabled_points(self):
        # filters the set of points for just those enabled, then returns that set of enabled points
        self.__enabled_points__ = filter_for_enabled(self.__enabled_points__)
        return self.__enabled_points__

    def find_highest_point(self):
        # returns the highest point in the point cloud, ie the one with the greatest z coordinate
        return max(self.enabled_points(), key=lambda point: point.z())

    def find_stem(self, min_height=.5, fail_to_find_disable_radius=.2, descendents_height=.3):
        # returns a stem from the point cloud if it exists, otherwise returns None
        # the stem will be a list of points in descending z value, starting from above the min height and going down to a ground point
        # inputs:
        #   - min_height : the minimum height (in m) for a valid stem
        #   - fail_to_find_disable_radius : when a point has no descendents, we disable each point within this radius of it
        #   - descendents_height : the maximum vertical distance allowed to a descendent of this point
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
                    self.enabled_points(), height_below_self=descendents_height)
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
                            f"We disabled {n_points_start-n_points_end} points near this inactive point!")
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

    def plot(self, stem=None, disabled_color='green', theta=0, save_location=None):
        # displays the point cloud
        # inputs:
        #   - stem : a set of points to highlight as the stem
        #   - disabled_color : the color to draw the disabled points as. If None, will not draw them
        #   - theta: an angle (in degrees), which rotates the picture
        plt.close()
        enabled_point_cloud = self.enabled_points()
        sin = math.sin(theta*180/math.pi)
        cos = math.cos(theta*180/math.pi)
        x = [point.xyz[0]*cos+point.xyz[1]*sin for point in enabled_point_cloud]
        z = [point.xyz[2] for point in enabled_point_cloud]
        plt.scatter(x=x, y=z, color='blue')
        if disabled_color:
            # plot disabled points in a separate color
            disabled_point_cloud = filter_for_disabled(self.points)
            x = [point.xyz[0]*cos+point.xyz[1] *
                 sin for point in disabled_point_cloud]
            z = [point.xyz[2] for point in disabled_point_cloud]
            plt.scatter(x=x, y=z, color=disabled_color)
        if stem:
            # plot the stem in red
            x_stem = [stem_point.xyz[0]*cos +
                      stem_point.xyz[1]*sin for stem_point in stem]
            z_stem = [stem_point.xyz[2] for stem_point in stem]
            plt.scatter(x=x_stem, y=z_stem, color='red')
        if save_location:
            # save the figure
            plt.savefig(fname=f"saved_images/side_views/{save_location}")

    def save_via_dataframe(self, file_name="point_clusters.csv", folder_name="cluster_csvs", stems=None):
        # saves this point cloud as a csv with their cluster information
        # inputs:
        #   file_name : the file that we save the data to (this will overwrite an old file if it is there)
        #   folder_name : the folder to save the data to
        #   stems : optional, a list of list of points, where we'll save the average position of each list into a separate file
        pre_df = [list(point.xyz) + [point.height_above_ground,
                                     point.cluster] for point in self.points]
        # df has one row per point in the pointcloud, with these columns:
        df = pd.DataFrame(
            pre_df, columns=["x", "y", "z", "height_above_ground", "cluster"])

        # save the df
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        out_file_name = f"{folder_name}/{file_name}"
        with open(out_file_name, 'w') as f:
            df.to_csv(f)

        # if we also want to save stem information, we do that here
        if stems:
            stem_centers = compute_stem_centers(stems)
            center_point = np.average(
                np.array([point.xyz for point in self.points]), axis=0)
            stem_centers = [Point(center_point, -1)]+stem_centers
            pre_df_stems = [list(point.xyz) + [point.height_above_ground,
                                               point.cluster] for point in stem_centers]
            df_stems = pd.DataFrame(
                pre_df_stems, columns=["x", "y", "z", "height_above_ground", "cluster"])
            df_stems["cluster"] = [n for n in range(len(stem_centers))]
            df_stems["cluster_name"] = ["Ground"] + \
                [f"Tree {n+1}" for n in range(len(stem_centers)-1)]
            stem_out_file_name = out_file_name.split("clusters.csv")[
                0]+"stem_centers.csv"
            with open(stem_out_file_name, 'w') as f_stems:
                df_stems.to_csv(f_stems)


class Point:
    # a point in the point cloud

    def __init__(self, xyz, height_above_ground):
        # inputs:
        #   xyz - a 3-item list of x, y, and z coordinates
        #   height_above_ground - the point's height above ground
        self.xyz = xyz
        self.height_above_ground = height_above_ground
        self.is_ground = height_above_ground < 0
        self.enabled = True
        self.descendents = None
        self.cluster = -1
        # -1 is the code for not-yet-clustered points

    def __repr__(self):
        return str(list(self.xyz)+[self.height_above_ground, self.enabled])

    def find_descendents(self, points_to_initialize_descendents, horizontal_radius=.2, height_above_self=0, height_below_self=0.3):
        # finds enabled points in a cylinder below self
        if not self.descendents:
            self.descendents = self.find_points_in_cylinder(
                points_to_initialize_descendents, horizontal_radius=horizontal_radius, height_above_self=height_above_self, height_below_self=height_below_self)
        self.descendents = filter_for_enabled(self.descendents)
        return self.descendents

    def find_points_in_cylinder(self, points_to_check, horizontal_radius=.2, height_above_self=0, height_below_self=0.3):
        # finds points in a cylinder around self
        # screen for points not too far below
        points_to_check = filter(
            lambda point: -1*height_below_self < point.z()-self.z(), points_to_check)
        # screen for points not too far above
        points_to_check = list(
            filter(lambda point: height_above_self > point.z()-self.z(), points_to_check))
        # screen for points within the correct horizontal distance
        squared_distances = np.zeros(len(points_to_check))
        for dim in range(2):
            distance_on_this_dimension = self.xyz[dim]-np.array(
                [point.xyz[dim] for point in points_to_check])
            squared_distances += distance_on_this_dimension**2
        points_in_cylinder = [point for point, squared_distance in zip(
            points_to_check, squared_distances) if squared_distance < horizontal_radius**2]
        return points_in_cylinder

    def z(self):
        # returns the z coordinate of the point
        return self.xyz[2]


def filter_for_enabled(points_list):
    #filters a list for just the points which are enabled
    return [point for point in points_list if point.enabled]


def filter_for_disabled(points_list):
    #filters a list for just the points which are NOT enabled
    return [point for point in points_list if not point.enabled]


def filter_for_clustered(points_list):
    #filters a list for just the points which are in a cluster
    return [point for point in points_list if point.cluster >= 0]


def filter_for_clustered_non_ground(points_list):
    #filters a list for just the points which are in a non-ground cluster
    return [point for point in points_list if point.cluster > 0]


def filter_for_unclustered(points_list):
    #filters a list for just the points which are NOT in a cluster
    return [point for point in points_list if point.cluster < 0]


def squared_horizontal_distance(point1, point2):
    # computes the squared horizontal distance between the two points
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2


def squared_distance(point1, point2):
    # computes the squared distance between the two points
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2+(point1.xyz[2]-point2.xyz[2])**2


def downsample(arr, fraction):
    # selects a random subset of fraction elements in arr
    # not used, but could be used if you need to speed up the algorithms
    np.random.shuffle(arr)
    cutoff_point = int(arr.shape[0]*fraction)
    return arr[:cutoff_point]


def compute_stem_centers(stems):
    # creates a list of points, one point as the "stem center" of each stem in stems
    stem_centers = [
        Point(sum([point.xyz for point in stem])/len(stem), max([point.height_above_ground for point in stem])) for stem in stems]
    return stem_centers


def assign_clusters_by_nearest(point_cloud, stems, height_cutoff=0.2):
    # this method is old, and not recommended, assign_clusters_by_growing() is preferred
    # assigns every point in the point cloud to its closest stem (as measured by horizontal distance to stem midpoint)
    # points below height_cutoff are sent to a separate cluster 0 for the ground
    stem_centers = compute_stem_centers(stems)
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
    # now we "grow" each stem to form the core of our clusters
    # for each point in the cluster, we add the unclustered points in a cylinder above it to the same cluster, 
    # repeating until this adds no more new points to the cluster
    for i, stem in enumerate(stems):
        t_start = time.time()
        cluster_number=i+1
        growing_points = [stem_point for stem_point in stem]
        while growing_points:
            unclustered_points = filter_for_unclustered(point_cloud.points)
            point_to_grow = growing_points.pop()
            # we grow points one at a time
            upwards_points = point_to_grow.find_points_in_cylinder(
                unclustered_points, horizontal_radius=grow_radius, height_above_self=grow_height, height_below_self=0)
            if point_cloud.verbose:
                print(
                    f"Found {len(upwards_points)} new points for cluster {cluster_number}/{len(stems)}")
            for upward_point in upwards_points:
                upward_point.cluster = cluster_number
                growing_points.append(upward_point)
        t_end = time.time()
        print(
            f"Finished initial assignment of points to cluster {cluster_number}/{len(stems)} in {t_end-t_start:.2f} seconds!")
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

    stem_centers = compute_stem_centers(stems)

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


def initialize_arguments():
    parser = argparse.ArgumentParser(
        description='Clusters points into trees a .las file')
    parser.add_argument('-f', '--file_name', required=False, type=str, default="",
                        help='the name of the input file')
    parser.add_argument('-p', '--make_plots', action='store_const', required=False, const=True,
                        help='whether to save plots')
    parser.add_argument('-sv', '--side_view', required=False, type=int, default=0,
                        help='whether to save a plot with a side view, and at what angle to view it')
    parser.add_argument('-dh', '--descendents_height', required=False, type=float, default=.3,
                        help='how far to search vertically for descendents')
    parser.add_argument('-of', '--output_folder', required=False, type=str, default="cluster_csvs",
                        help='the name of the folder to save the outputs to')
    args = parser.parse_args()

    if args.file_name:
        if re.match("\d+", args.file_name):
            file_names = [f"Test_data/treeID_{args.file_name}_merged.las"]
        else:
            file_names = [args.file_name]
    else:
        file_names = [
            # "Test_data/treeID_10717_merged.las", # 1 stem
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
    return args, file_names


if __name__ == "__main__":

    args, file_names = initialize_arguments()

    for file_name in file_names:
        random.seed(42)
        try:
            point_cloud = Point_Cloud(file_name)
            print(
                f"Loaded file {file_name}, attempting to cluster {len(point_cloud.points)} points.")
        except:
            print(
                f"I could not find the file named {file_name}, please try again with a new file.")
            break

        overall_t_start = time.time()
        stems = []
        stem = True
        while stem:
            t_start = time.time()
            stem = point_cloud.find_stem(
                descendents_height=args.descendents_height)
            t_end = time.time()
            if stem:
                stems.append(stem)
                print(
                    f"I found stem #{len(stems)} in {t_end-t_start:.2f} seconds!")
                # point_cloud.plot(stem=stem, disabled_color='green', theta=0)
                point_cloud.disable_stem_region(stem)
            else:
                print(
                    f"I found that there were no more stems in {t_end-t_start:.2f} seconds!")
        # assign_clusters_by_nearest(point_cloud,stems)
        assign_clusters_by_growing(point_cloud, stems)

        file_name_prefix = file_name.split(".")[0].split("/")[1]
        csv_file_name = f"{file_name_prefix}_clusters.csv"
        point_cloud.save_via_dataframe(file_name=csv_file_name, stems=stems, folder_name=args.output_folder)
        if args.make_plots:
            plot_file_name = f"{file_name_prefix}_cluster_plot.png"
            plot_stem_centers(point_cloud, stems=stems,
                              include_ground=False, save_title=plot_file_name)
        if args.side_view:
            plot_file_name = f"{file_name_prefix}_side_view_angle_{args.side_view}.png"
            point_cloud.plot(theta=args.side_view,
                             disabled_color="blue", save_location=plot_file_name)

        overall_t_end = time.time()
        if stems:
            print(
                f"Done! Assigned {len(point_cloud.points)} points to {len(stems)} stems in {overall_t_end-overall_t_start:.2f} seconds!")
        else:
            print(f"Quit after failing to find any stems in file {file_name}.")