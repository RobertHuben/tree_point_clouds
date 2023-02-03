import laspy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import time
import cProfile

class Point_Cloud:

    def __init__(self, file_name):
        all_data=laspy.read(file_name)

        height_above_ground=np.array([point[-1] for point in all_data.points.array])
        # height_above_ground=np.expand_dims(height_above_ground, 1)
        # foo=np.array(all_data.xyz)
        # data=np.concatenate((foo, height_above_ground), axis=1)
        self.points=[Point(xyz, height) for xyz, height in zip(all_data.xyz, height_above_ground)]
        self.__enabled_points__=self.points

    def enabled_points(self):
        self.__enabled_points__=filter_for_enabled(self.__enabled_points__)
        return self.__enabled_points__

    def find_highest_point(self):
        return max(self.enabled_points(), key=lambda point: point.z())

    def find_stem(self, min_height=.5):
        stem_grounded=False
        while not(stem_grounded):
            stem=[self.find_highest_point()]
            if stem[0].height_above_ground<min_height:
                #if your stem is too short, we're done
                break
            while not(stem_grounded) and stem:
                active_point=stem[-1]
                descendents=active_point.find_descendents(self.enabled_points())
                if descendents:
                    # if there is a valid descendent, take one at random (we can try again if it doesnt work)
                    next_point=random.choice(descendents)
                    stem.append(next_point)
                    if next_point.is_ground:
                        stem_grounded=True
                else:
                    # if the point has no descendent, it cannot have a path to the ground, so we disable it
                    active_point.enabled=False
                    stem.pop(-1)
        if stem_grounded:
            # we've found a stem
            return stem
        else:
            # if the point cloud cannot find a new stem, we stop
            return None

    def disable_stem_region(self, stem, radius_to_delete=.7):
        squared_radius_to_delete=radius_to_delete**2
        for enabled_point in self.enabled_points():
            if any([squared_distance(enabled_point, stem_point)<squared_radius_to_delete for stem_point in stem]):
                enabled_point.enabled=False



    def plot(self, stem=None, disabled_color='green'):
        plt.close()
        enabled_point_cloud=self.enabled_points()
        x=[point.xyz[0] for point in enabled_point_cloud]
        z=[point.xyz[2] for point in enabled_point_cloud]
        plt.scatter(x=x, y=z, color='blue')
        if disabled_color:
            disabled_point_cloud=filter_for_disabled(self.points)
            x=[point.xyz[0] for point in disabled_point_cloud]
            z=[point.xyz[2] for point in disabled_point_cloud]
            plt.scatter(x=x, y=z, color='green')
        if stem:
            x_stem=[stem_point.xyz[0] for stem_point in stem]
            z_stem=[stem_point.xyz[2] for stem_point in stem]
            plt.scatter(x=x_stem, y=z_stem, color='red')


class Point:

    def __init__(self, xyz, height_above_ground, ):
        self.xyz=xyz
        self.height_above_ground=height_above_ground
        self.is_ground=height_above_ground<0
        self.enabled=True
        self.descendents=None

    def __repr__(self):
        return str(list(self.xyz)+[self.height_above_ground, self.enabled])

    def find_descendents(self, points_to_check, horizontal_radius=.2, height_difference=.3):
        if not self.descendents:
            points_to_check=filter(lambda point: point.z()<self.z(), points_to_check)
            points_to_check=list(filter(lambda point: point.z()+height_difference>self.z(), points_to_check))
            squared_distances=np.zeros(len(points_to_check))
            for dim in range(2):
                distance_on_this_dimension=self.xyz[dim]-np.array([point.xyz[dim] for point in points_to_check])
                squared_distances+=distance_on_this_dimension**2
            self.descendents=[point for point, squared_distance in zip(points_to_check, squared_distances) if squared_distance<horizontal_radius**2]

        self.descendents=filter_for_enabled(self.descendents)
        return self.descendents

    def z(self):
        return self.xyz[2]


def filter_for_enabled(points_list):
    return [point for point in points_list if point.enabled]

def filter_for_disabled(points_list):
    return [point for point in points_list if not point.enabled]


def squared_horizontal_distance(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

def squared_distance(point1, point2):
    return (point1.xyz[0]-point2.xyz[0])**2+(point1.xyz[1]-point2.xyz[1])**2+(point1.xyz[2]-point2.xyz[2])**2



def downsample(arr, fraction):
    np.random.shuffle(arr)
    cutoff_point=int(arr.shape[0]*fraction)
    return arr[:cutoff_point]

def choose_guide_points(point_cloud, radius_of_density=1.4):
    downsample_fraction=1/np.sqrt(point_cloud.shape[0])
    downsample_fraction=1
    guide_points=downsample(point_cloud, downsample_fraction)
    squared_distances=np.zeros((point_cloud.shape[0], guide_points.shape[0]))
    for dim in range(3):
        points_one_dimension=point_cloud[:, dim]
        points_one_dimension=np.expand_dims(points_one_dimension, 1)

        guide_points_one_dimension=guide_points[:,dim]
        guide_points_one_dimension=np.expand_dims(guide_points_one_dimension, 0)

        differences=np.subtract(points_one_dimension,guide_points_one_dimension)
        squared_distances+=differences**2
    adjacencies=squared_distances<radius_of_density**2
    return adjacencies

def plot_stem_centers(point_cloud, stem_centers):
    plt.close()
    enabled_point_cloud=point_cloud.points
    x=[point.xyz[0] for point in enabled_point_cloud]
    y=[point.xyz[1] for point in enabled_point_cloud]
    plt.scatter(x=x, y=y, color='blue', alpha=.005)
    stem_center_x=[stem_center[0] for stem_center in stem_centers]
    stem_center_y=[stem_center[1] for stem_center in stem_centers]
    plt.scatter(x=stem_center_x, y=stem_center_y, color='red', marker='x')
    plt.show()
    return


def find_n_stems():
    n=4
    file_name="Test_data/treeID_35618_merged.las"
    random.seed(42)

    point_cloud=Point_Cloud(file_name)
    stems=[]
    for _ in range(n):
        stem=point_cloud.find_stem()
        if stem:
            stems.append(stem)
            point_cloud.disable_stem_region(stem)
        else:
            break



if __name__=="__main__":

    cProfile.run('find_n_stems()')


    # file_name="Test_data/treeID_12210.las"
    # file_name="Test_data/treeID_19707.las"
    # file_name="Test_data/treeID_33009.las"
    # file_name="Test_data/treeID_34926_merged.las"
    file_name="Test_data/treeID_35618_merged.las"
    sample_fraction=.5
    random.seed(42)

    # stem_centers_file='stem_centers_output.txt'
    # stem_centers=[]
    # with open(stem_centers_file, 'r') as infile:
    #     for line in infile.read().split("array"):
    #         if line:
    #             stem_centers.append(np.array([float(s) for s in line[2:-4].split(",  ")]))

    point_cloud=Point_Cloud(file_name)
    # plot_stem_centers(point_cloud, stem_centers)

    # point_cloud.plot()
    stems=[]
    while True:
        t_start=time.time()
        stem=point_cloud.find_stem()
        t_end=time.time()
        if stem:
            stems.append(stem)
            print(f"I found the {len(stems)}th stem in {t_end-t_start} seconds!")
            point_cloud.plot(stem=stem, disabled_color='green')
            point_cloud.disable_stem_region(stem)
            # plot_point_cloud(point_cloud, stem)
        else:
            print(f"I found that there were no more stems in {t_end-t_start} seconds!")
            break
    print(f"Done! Found {len(stems)} stems")
    print("foo")

