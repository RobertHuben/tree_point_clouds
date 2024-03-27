"""Segments point cloud data from a LAS file into individual trees.

This program reads lidar data from a LAS file, identifies clusters of points corresponding to individual trees, and outputs the resulting clusters in a csv file.

In order to identify trees, this algorithm first identifies vertical "stems" which go from the top of a tree to the ground. The remaining points are matched to nearby stems, resulting in one output cluster per stem which should correspond to a single tree.

Example Usage:
python3 tree_segment.py --file_name sample_data/treeID_40645_merged.las
"""

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
import logging

# logging.basicConfig(level=logging.INFO)


class Point_Cloud:
    """Object that stores the points in a point cloud, initialized from a las file.

    Points can be enabled or disabled, disabled points are ignored when searching for stems
    """

    def __init__(self, file_name):
        all_data = laspy.read(file_name)
        height_of_points_above_ground = np.array(
            [point[-1] for point in all_data.points.array])
        self.points = [Point(xyz, height) for xyz, height in zip(
            all_data.xyz, height_of_points_above_ground)]
        self.enabled_points = self.points

    def find_highest_point(self):
        """ Returns the highest point in the point cloud, ie the one with the greatest z coordinate."""
        return max(self.enabled_points, key=lambda point: point.z())

    def find_stem(self, minimum_stem_height=.5, found_stem_disable_radius=.7, fail_to_find_disable_radius=.2, descendents_height=.3, descendents_radius=.2):
        """Returns a stem from the point cloud if one exists, otherwise returns None. Disables points during the process.

        The stem will be a list of points in descending z value, starting from above the min height and going down to a ground point
        inputs:
          - minimum_stem_height : the minimum height (in m) for a valid stem
          - found_stem_disable_radius : when we find a stem, we disable each point within this distance of a stem point
          - fail_to_find_disable_radius : when a point has no descendents, we disable each point within this radius of it
          - descendents_height : the maximum vertical distance allowed to a descendent of this point
          - descendents_radius : the maximum horizontal distance allowed to a descendent of this point
        """
        stem = []
        while True:
            """ loop has 6 possible outcomes, in order:
                1. The highest point is below the stem cutoff height, so there are no more stems and we return None.
                2. We start a new stem from the highest point (going to top of loop).
                3. If the active point is disabled, we remove it from the stem (going to top of loop)
                4. We've found a grounded stem, so we disable the points near to it and return it.
                5. The latest point has no descendents, so we disable it and nearby points (going to top of loop)
                6. We add a descendent to the stem (going to the top of loop)
            """
            # if the stem is empty, we try to start a new one from the highest point
            if not stem:
                highest_point = self.find_highest_point()
                if highest_point.height_above_ground < minimum_stem_height:
                    # every point in the point cloud is below the cutoff, so there are no more stems
                    # outcome 1:
                    return None
                # outcome 2:
                stem.append(highest_point)
                continue

            active_point = stem[-1]
            # check if the active point is ground or disabled, for outcomes 3 or 4:
            if not active_point.enabled:
                # outcome 3:
                stem.pop()
                continue
            if active_point.is_ground:
                logging.info("Found a stem!")
                # outcome 4:
                self.disable_region_near_points(
                    points_to_center_disabling=stem,
                    radius_to_disable=found_stem_disable_radius)
                return stem

            # look for a descendent
            descendents = active_point.find_descendents(
                points_to_initialize_descendents= self.enabled_points, 
                height_below_self=descendents_height,
                horizontal_radius=descendents_radius
                )
            if not descendents:
                # if the point has no descendents, it cannot have a path to the ground, so we disable it
                # we also disable all point within fail_to_find_disable_radius of it, which makes the code run fast, though this could accidentally delete valid stem paths
                # outcome 5:
                self.disable_region_near_points(
                    points_to_center_disabling=[active_point],
                    radius_to_disable=fail_to_find_disable_radius)
                continue

            # if there is a valid descendent, take one at random (we will try again later if it doesnt work)
            # outcome 6:
            next_point = random.choice(descendents)
            stem.append(next_point)

    def disable_region_near_points(self, points_to_center_disabling, radius_to_disable):
        """Disables all points in the point cloud within radius_to_delete distance of a point in the points_to_center_disabling."""
        n_points_start = len(self.enabled_points)
        squared_radius_to_disable = radius_to_disable**2
        for enabled_point in self.enabled_points:
            if any([squared_distance(enabled_point, disabling_point) < squared_radius_to_disable for disabling_point in points_to_center_disabling]):
                enabled_point.enabled = False
        self.enabled_points = filter_for_enabled(self.enabled_points)
        n_points_end = len(self.enabled_points)
        logging.info(
            f"We disabled {n_points_start-n_points_end} points near this inactive point!")

    def save_via_dataframe(self, file_name="point_clusters.csv", folder_name="cluster_csvs", stems=None):
        """Saves this point cloud as a csv with their cluster information.

        inputs:
          file_name : the file that we save the data to (this will overwrite an old file if it is there)
          folder_name : the folder to save the data to
          stems : optional, a list of list of points, where we'll save the average position of each list into a separate file
        """
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
    """An object representing a point in the point cloud."""

    def __init__(self, xyz, height_above_ground):
        """ Initializes a Point object.

        inputs:
          xyz - a 3-item list of x, y, and z coordinates
          height_above_ground - the point's height above ground
        """
        self.xyz = xyz
        self.height_above_ground = height_above_ground
        self.is_ground = height_above_ground < 0
        self.enabled = True
        self.descendents = None
        self.cluster = -1
        # -1 is the code for not-yet-clustered points

    def __repr__(self):
        return str(list(self.xyz)+[self.height_above_ground, self.enabled])

    def find_descendents(self, points_to_initialize_descendents, horizontal_radius, height_below_self):
        """Finds enabled points in a cylinder below self."""
        if not self.descendents:
            self.descendents = self.find_points_in_cylinder(
                points_to_initialize_descendents, horizontal_radius=horizontal_radius, height_above_self=0, height_below_self=height_below_self)
        self.descendents = filter_for_enabled(self.descendents)
        return self.descendents

    def find_points_in_cylinder(self, points_to_check, horizontal_radius, height_above_self, height_below_self):
        """Finds points in a cylinder around self.

        inputs:
            self - the point to form the cylinder around
            points_to_check - iterable of points to search over
            horizontal_radius - radius, in meters, of the cylinder to check for
            height_above_self - the height, in meters, that the cylinder extends vertically above self
            height_below_self - the height, in meters, that the cylinder extends vertically below self
        """

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
        """Returns the z coordinate of the point."""
        return self.xyz[2]


def filter_for_enabled(points_list):
    """Filters a list for just the points which are enabled."""
    return [point for point in points_list if point.enabled]


def filter_for_disabled(points_list):
    """Filters a list for just the points which are NOT enabled."""
    return [point for point in points_list if not point.enabled]


def filter_for_clustered(points_list):
    """Filters a list for just the points which are in a cluster."""
    return [point for point in points_list if point.cluster >= 0]


def filter_for_clustered_non_ground(points_list):
    """Filters a list for just the points which are in a non-ground cluster."""
    return [point for point in points_list if point.cluster > 0]


def filter_for_unclustered(points_list):
    """Filters a list for just the points which are NOT in a cluster."""
    return [point for point in points_list if point.cluster < 0]


def squared_horizontal_distance(point1, point2):
    """Computes the squared horizontal distance between the two points."""
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2


def squared_distance(point1, point2):
    """Computes the squared distance between the two points."""
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2+(point1.xyz[2]-point2.xyz[2])**2


def downsample(arr, fraction):
    """Selects a random subset of fraction elements in arr.

    Not used, but could be used if you need to speed up the algorithms
    """
    np.random.shuffle(arr)
    cutoff_point = int(arr.shape[0]*fraction)
    return arr[:cutoff_point]


def compute_stem_centers(stems):
    """Creates a list of points, one point as the "stem center" of each stem in stems."""
    stem_centers = [
        Point(sum([point.xyz for point in stem])/len(stem), max([point.height_above_ground for point in stem])) for stem in stems]
    return stem_centers


def assign_clusters_by_nearest(point_cloud, stems, ground_height_cutoff=0.2):
    """Deprecated method, assign_clusters_by_growing() is preferred.

    assigns every point in the point cloud to its closest stem (as measured by horizontal distance to stem midpoint)
    points below ground_height_cutoff are sent to a separate cluster 0 for the ground
    """
    stem_centers = compute_stem_centers(stems)
    for point in point_cloud.points:
        if point.height_above_ground < ground_height_cutoff:
            point.cluster = 0
        else:
            squared_horizontal_distance_to_stem_centers = np.array(
                [squared_horizontal_distance(point, stem_center) for stem_center in stem_centers])
            point.cluster = 1 + \
                np.argmin(squared_horizontal_distance_to_stem_centers)


def assign_clusters_by_growing(point_cloud, stems, grow_radius=.2, grow_height=.4, ground_height_cutoff=0.2):
    """Assigns points to clusters based on "growing" the stem.

    Points below ground_height_cutoff are sent to a separate cluster 0 for the ground
    """
    for point in point_cloud.points:
        if point.height_above_ground < ground_height_cutoff:
            point.cluster = 0
    # now we "grow" each stem to form the core of our clusters
    # for each point in the cluster, we add the unclustered points in a cylinder above it to the same cluster,
    # repeating until this adds no more new points to the cluster
    for i, stem in enumerate(stems):
        t_start = time.time()
        cluster_number = i+1
        growing_points = [stem_point for stem_point in stem]
        while growing_points:
            unclustered_points = filter_for_unclustered(point_cloud.points)
            point_to_grow = growing_points.pop()
            # we grow points one at a time
            upwards_points = point_to_grow.find_points_in_cylinder(
                unclustered_points, horizontal_radius=grow_radius, height_above_self=grow_height, height_below_self=0)
            logging.info(
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
    """Assigns points to existing clusters based on which cluster it is currently closest to."""
    out_of_cluster_points = filter_for_unclustered(point_cloud.points)
    in_cluster_points = filter_for_clustered_non_ground(point_cloud.points)
    for point in out_of_cluster_points:
        nearest_point = min(in_cluster_points,
                            key=lambda x: squared_distance(x, point))
        point.cluster = nearest_point.cluster


def plot_from_angle(point_cloud, side_view_rotation_angle=0, stems=[], save_title=None, save_folder="saved_images", ground_points_color="", unclustered_points_color=""):
    """Creates and saves a plot from a top-down perspective of the points and their stem centers.

    inputs:
      - point_cloud : the Point_Cloud object to plot
      - side_view_rotation_angle : number which controls whether the plot is top-down or a side view, and if a side view which angle
           - if 0: top-down plot
           - if >0: side view rotated that many degrees (use side_view_rotation_angle = 360 for "unrotated" side view)
      - stems : the stems (list of list of points) whose centers will be drawn
      - save_title : the name to save the image as. if you leave it as None the image won't be saved
      - save_folder : the folder to save the image to
      - ground_points_color : the name of the color to make the points in the 'ground' cluster. leave as the empty string to not draw it
      - unclustered_points_color : the name of the color to make the points in the 'unclustered' cluster. leave as the empty string to not draw it"""

    colors = ['yellow', 'green', 'blue', 'purple', 'red',
              'orange', 'cyan', 'brown']
    num_colors = len(colors)
    num_clusters = max([point.cluster for point in point_cloud.points])
    # plot the clusters:
    for i in range(-1, num_clusters+1):
        if i > 0:
            # this cluster is a tree
            label = f'Tree {i}'
            color = colors[i % num_colors]
        elif i == -1:
            # unclustered points
            if not unclustered_points_color:
                continue
            label = "Unclustered Points"
            color = unclustered_points_color
        elif i == 0:
            # ground
            if not ground_points_color:
                continue
            label = "Ground"
            color = ground_points_color
        points_in_this_cluster = [
            point for point in point_cloud.points if point.cluster == i]
        if not side_view_rotation_angle:
            x = [point.xyz[0] for point in points_in_this_cluster]
            y = [point.xyz[1] for point in points_in_this_cluster]
            alpha = .1
        else:
            sin = math.sin(side_view_rotation_angle*math.pi/180)
            cos = math.cos(side_view_rotation_angle*math.pi/180)
            x = [point.xyz[0]*cos+point.xyz[1] *
                 sin for point in points_in_this_cluster]
            y = [point.xyz[2] for point in points_in_this_cluster]
            alpha = .2
        plt.scatter(
            x=x, y=y, color=color, alpha=alpha, label=label)

    # plot the stem centers
    if stems:
        stem_centers = compute_stem_centers(stems)
        stem_center_x = [stem_center.xyz[0] for stem_center in stem_centers]
        stem_center_y = [stem_center.xyz[1] for stem_center in stem_centers]
        plt.scatter(x=stem_center_x, y=stem_center_y,
                    color='black', marker='x', label='Stem Centers')

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.title(save_title)
    if save_title:
        plt.savefig(fname=f"{save_folder}/{save_title}")
    plt.close()


def initialize_arguments():
    parser = argparse.ArgumentParser(
        description='Clusters points from a .las file into trees.')
    parser.add_argument('-f', '--file_name', required=False, type=str, default="",
                        help=f'the name of the input file')
    parser.add_argument('-td', '--top_down', action='store_const', required=False, const=True,
                        help=f'whether to save a top-down plot of the clustering results')
    parser.add_argument('-sv', '--side_view', required=False, type=int, default=0,
                        help=f'whether to save a side-view plot of the clustering results, and at what angle to view it. must be >0 (default: 0)')
    parser.add_argument('-dh', '--descendents_height', required=False, type=float, default=.3,
                        help=f'how far to search vertically for descendents when searching for stems (default: .3m)')
    parser.add_argument('-dr', '--descendents_radius', required=False, type=float, default=.2,
                        help=f'how far to search horizontally for descendents when searching for stems (default: .2m)')
    parser.add_argument('-msh', '--minimum_stem_height', required=False, type=float, default=.5,
                        help=f'minimum height required for a stem to be valid (default: .5m)')
    parser.add_argument('-fsdr', '--found_stem_disable_radius', required=False, type=float, default=.7,
                        help=f'points within this distance of a found stem will be disabled (default: .7m)')
    parser.add_argument('-ftfdr', '--fail_to_find_disable_radius', required=False, type=float, default=.2,
                        help=f'points within this distance of a point with no path to the ground will be disabled (default: .2m)')
    parser.add_argument('-gr', '--grow_radius', required=False, type=float, default=.2,
                        help=f'how far to search horizontally for other points in the tree when "growing" (default: .2m)')
    parser.add_argument('-gh', '--grow_height', required=False, type=float, default=.4,
                        help=f'how far to search vertically for other points in the tree when "growing" (default: .4)')
    parser.add_argument('-ghc', '--ground_height_cutoff', required=False, type=float, default=.2,
                        help=f'points below this height will be clustered into the ground cluster (default: .2m)')
    parser.add_argument('-of', '--output_folder', required=False, type=str, default="cluster_csvs",
                        help=f'the name of the folder where csv outputs are saved')
    args = parser.parse_args()

    if args.file_name:
        if re.match("\d+", args.file_name):
            file_names = [f"Test_data/treeID_{args.file_name}_merged.las"]
        else:
            file_names = [args.file_name]
    else:
        file_names = [
            "sample_data/treeID_40645_merged.las",  # 2 stems
            # "sample_data/treeID_40113_merged.las", # 3 stems
        ]
        # file_names = [
        # "Test_data/treeID_40038_merged.las", # 2 stems
        # "Test_data/treeID_40061_merged.las", # 13 stems
        # "Test_data/treeID_40113_merged.las", # 3 stems
        # "Test_data/treeID_40803_merged.las", # 9ish stems
        # ]
    return args, file_names


def cluster_point_cloud(point_cloud, args):
    stems = []
    stem = True
    while stem:
        t_start = time.time()
        stem = point_cloud.find_stem(
            minimum_stem_height=args.minimum_stem_height,
            found_stem_disable_radius=args.found_stem_disable_radius,
            fail_to_find_disable_radius=args.fail_to_find_disable_radius,
            descendents_height=args.descendents_height,
            descendents_radius=args.descendents_radius
            )
        t_end = time.time()
        if stem:
            stems.append(stem)
            print(
                f"I found stem #{len(stems)} in {t_end-t_start:.2f} seconds!")
        else:
            print(
                f"I found that there were no more stems in {t_end-t_start:.2f} seconds!")
    # assign_clusters_by_nearest(point_cloud,stems)
    assign_clusters_by_growing(point_cloud, stems,
                               grow_radius=args.grow_radius,
                               grow_height=args.grow_height,
                               ground_height_cutoff=args.ground_height_cutoff)
    return point_cloud, stems


if __name__ == "__main__":

    args, file_names = initialize_arguments()

    for file_name in file_names:
        random.seed(42)
        try:
            point_cloud = Point_Cloud(file_name)
            print(
                f"Loaded file {file_name}, attempting to cluster {len(point_cloud.points)} points.")
        except FileNotFoundError:
            print(
                f"I could not find the file named {file_name}, please try again with a new file.")
            break

        overall_t_start = time.time()
        point_cloud, stems = cluster_point_cloud(point_cloud, args)

        file_name_prefix = file_name.split(".")[0].split("/")[-1]
        csv_file_name = f"{file_name_prefix}_clusters.csv"
        point_cloud.save_via_dataframe(
            file_name=csv_file_name, stems=stems, folder_name=args.output_folder)
        if args.top_down:
            plot_file_name = f"{file_name_prefix}_cluster_plot.png"
            plot_from_angle(point_cloud, stems=stems, side_view_rotation_angle=0,
                            ground_points_color="", save_title=plot_file_name, save_folder="saved_images/top_downs")
        if args.side_view:
            plot_file_name = f"{file_name_prefix}_side_view_angle_{args.side_view}.png"
            plot_from_angle(point_cloud, ground_points_color='black',
                            save_title=plot_file_name, side_view_rotation_angle=args.side_view, save_folder="saved_images/side_views")

        overall_t_end = time.time()
        if stems:
            print(
                f"Done! Assigned {len(point_cloud.points)} points to {len(stems)} stems in {overall_t_end-overall_t_start:.2f} seconds!")
        else:
            print(f"Quit after failing to find any stems in file {file_name}.")
