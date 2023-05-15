# tree_point_clouds
Methods for identifying individual trees from point cloud data

Segments a point cloud from a LAS file into individual trees.

## _Brief description_

This script works with Python 3. It takes as input a LAS file and outputs two csvs assigning the points from the LAS file into clusters, one cluster per tree plus one cluster for the ground. In our testing it clustered ~6000 points in ~16 seconds, with runtimes increasing as O(n^2) in the number of points.

There are 3 steps to the script:

1. Identifying "stems", which are a list of points with descending z-value, that go from the top of a tree to the ground. In the final clustering, each stem will generate one cluster, which should consist of one tree.
2. Do an initial clustering pass by "growing" each stem. For each point in the cluster, we add the unclustered points in a cylinder above it to the same cluster, repeating until this adds no more new points to the cluster.
3. Cluster the remaining points by finding its nearest neighbor that was clustered in step (2) and assigning it to the same cluster.

Please don't hesitate to open an [`Issue`](https://github.com/RobertHuben/tree_point_clouds/issues) if you find any problem or suggestions for a new feature.



## _Usage_

Before using this script, you must install laspy:
```
pip3 install laspy
```

You can run the script from the command line as `python3 tree_segment.py`. You will likely want to use optional arguments to set the input file and some configuration options. You can see the optional arguments by running `python3 tree_segment.py -h`:

```
usage: tree_segment.py [-h] [-f FILE_NAME] [-td] [-sv SIDE_VIEW] [-dh DESCENDENTS_HEIGHT] [-of OUTPUT_FOLDER]

Clusters points from a .las file into trees.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_NAME, --file_name FILE_NAME
                        the name of the input file
  -td, --top_down       whether to save a top-down plot of the clustering results
  -sv SIDE_VIEW, --side_view SIDE_VIEW
                        whether to save a side-view plot of the clustering results, and at what angle to view it. must be >0
  -dh DESCENDENTS_HEIGHT, --descendents_height DESCENDENTS_HEIGHT
                        how far to search vertically for descendents when searching for stems
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        the name of the folder where csv outputs are saved
```


## _Examples_

_Example 1:_ Use default parameters to cluster the points in sample_data/treeID_40645_merged.las:

```
python3 tree_segment.py
# which is equivalent to
python3 tree_segment.py -f sample_data/treeID_40645_merged.las
# or
python3 tree_segment.py --file_name sample_data/treeID_40645_merged.las
```

_Example 2:_ Saving plots with a top-down perspective:

```
python3 tree_segment.py -f sample_data/treeID_40645_merged.las -td
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -td
# these should appear in the saved_images/ folder
```

_Example 3:_ Saving plots from a side-view perspective:

```
# create a side view of the trees rotated 90 degrees:
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -sv 45
# these should appear in the saved_images/side_views/ folder
# the image is somewhat unclear because a the angle makes two trees overlap. let's try this angle instead:
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -sv 135
# that's clearer! some other angles:
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -sv 180
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -sv 185
python3 tree_segment.py -f sample_data/treeID_40645_merged.las -sv 360
```

_Example 4:_ Saving the output csvs to another folder:

```
# create a side view of the trees rotated 90 degrees:
python3 tree_segment.py -f sample_data/treeID_40113_merged.las -of example_folder
```

## _Credits_
- Code: [Robert Huben](mailto:rvhuben@gmail.com)
- Data and testing: [Zhengyang Wang](mailto:zhengyangwang@g.harvard.edu)
