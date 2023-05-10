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



## _Examples_


## _Credits_
- Code: [Robert Huben](mailto:rvhuben@gmail.com)
- Data and testing: [Zhengyang Wang](mailto:zhengyangwang@g.harvard.edu)
