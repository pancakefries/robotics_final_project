#!/usr/bin/env python3

import rospy 
import numpy as np
from nav_msgs.msg import OccupancyGrid
from robotics_final_project.msg import NodeRow, NodeMatrix

class Node(object):
    def __init__(self, x: int, y: int):
        self.fcost = 1000
        self.hcost = 1000
        self.gcost = 1000
        self.x = x
        self.y = y
        self.parent = None

class Pathfinder(object):
    def __init__(self):
        self.current = None
        self.start = None
        self.goal = None
        self.grid = []

        self.map = None
        self.width = 0
        self.height = 0
        self.map_initialized = False

        rospy.init_node("target_path")
        # /target_nodes is where start and end positions are received
        rospy.Subscriber("target_nodes", NodeMatrix, self.get_target)
        rospy.Subscriber("map", OccupancyGrid, self.get_map)
        # Path calculated and sent to robot_controller
        self.path_pub = rospy.Publisher("target_path", NodeMatrix, queue_size=10)

    def get_target(self, data):
        # When a new path request arrives, initialize the starting, goal, and current nodes
        self.start = Node(data.matrix[0].row[0], data.matrix[0].row[1])
        self.start.parent = [self.start.x, self.start.y]
        self.goal = Node(data.matrix[1].row[0], data.matrix[1].row[1])
        self.current = Node(data.matrix[0].row[0], data.matrix[0].row[1])
        self.current.parent = [self.current.x, self.current.y]
        # Make sure that the map is initialized before starting the algorithm
        while not self.map_initialized:
            rospy.sleep(0.1)
        self.a_star()

    def downsample_map(self):
        # An attempt to stop the robot from navigating through walls by downsampling
        # the map by a factor of 4. Ultimately seems to be unsuccessful
        new_map_data = np.reshape(self.map_data, (self.width, self.height))
        a = new_map_data[::4, ::4]
        b = new_map_data[1::4, 1::4]
        c = new_map_data[2::4, 2::4]
        d = new_map_data[3::4, 3::4]
        new_map_data = a | b 
        new_map_data = new_map_data | c
        new_map_data = new_map_data | d
        self.map_data = new_map_data.flatten()
        self.width = int(self.width / 4)
        self.height = int(self.height / 4)

    def pad_map(self):
        # Another attempt to stop the robot from navigating through walls, this time
        # by padding the walls (I think the issue is that paths hug walls)
        new_map_data = np.reshape(self.map_data, (self.width, self.height))
        for i in np.reshape(self.map_data, (self.width, self.height)):
            for j in i:
                if j == 100:
                    for k in [0, 1, 2]:
                        for h in [0, 1, 2]:
                            new_map_data[i + k][j + h] = 100
                            new_map_data[i + k][j - h] = 100
                            new_map_data[i - k][j + h] = 100
                            new_map_data[i - k][j - h] = 100
        self.map_data = new_map_data.flatten()

    def get_map(self, data):
        self.map = data
        map_info = self.map.info
        self.map_data = self.map.data
        self.map_resolution = map_info.resolution
        self.width = map_info.width
        self.height = map_info.height
        self.map_origin = map_info.origin
        # Initialize the pathfinding grid with appropriate dimensions
        for i in range(self.width):
            row = []
            for j in range(self.height):
                row.append(Node(i, j))
            self.grid.append(row)
        # Use one of the attempted solutions, neither are great as of right now
        # self.downsample_map()
        self.pad_map()
        self.map_initialized = True
        print("Pathfinding: map initialized")

    def get_neighbors(self, current):
        # Get the 8 coordinates surrounding the current cell
        offsets = [-1, 0, 1]
        neighbors = []
        for x_offset in offsets:
            for y_offset in offsets:
                # Ensure that they fall within the bounds of the board and are valid positions
                if (x_offset != y_offset or x_offset != 0) and \
                    (current.x + x_offset < self.width) and \
                    (current.y + y_offset < self.height) and \
                    (current.x + x_offset >= 0) and (current.y + y_offset >= 0) and \
                    (self.map_data[(current.x + x_offset) + (current.y + y_offset)*self.width] == 0):
                    neighbors.append([current.x + x_offset, current.y + y_offset])
        return neighbors

    def update_gcost(self, node: Node):
        if node.x == self.start.x and node.y == self.start.y:
            return 0
        # If the parent is adjacent, add 10, if diagonal, add 14
        diff = abs(node.parent[0] - node.x) + abs(node.parent[1] - node.y)
        if diff == 2:
            return 14 + self.update_gcost(self.grid[node.parent[0]][node.parent[1]])
        if diff == 1:
            return 10 + self.update_gcost(self.grid[node.parent[0]][node.parent[1]])
        else:
            return 0

    def update_hcost(self, node: Node):
        dist_x = abs(node.x - self.goal.x)
        dist_y = abs(node.y - self.goal.y)
        return (14 * min([dist_x, dist_y])) + (10 * abs(dist_x - dist_y))
    
    def a_star(self):
        open_list = []
        closed_list = []
        open_list.append(self.current)
        update_fcost = lambda node : node.gcost + node.hcost

        while self.current.x != self.goal.x or self.current.y != self.goal.y:
            lowest = open_list[0]
            for cell in open_list:
                if self.grid[cell.x][cell.y].fcost < self.grid[lowest.x][lowest.y].fcost:
                    lowest = cell
            self.current = lowest
            open_list.remove(self.current)
            closed_list.append(self.current)

            if self.current.x == self.goal.x and self.current.y == self.goal.y:
                break
            
            ns = self.get_neighbors(self.current)
            # print("n neighbors: ", len(ns))
            for nb in ns:
                if self.grid[nb[0]][nb[1]] not in closed_list:
                    old_parent = self.grid[nb[0]][nb[1]].parent
                    test_node = self.grid[nb[0]][nb[1]]
                    test_node.parent = [self.current.x, self.current.y]
                    new_gcost = self.update_gcost(test_node)

                    if new_gcost < self.grid[nb[0]][nb[1]].gcost or \
                       (self.grid[nb[0]][nb[1]] not in open_list):
                        self.grid[nb[0]][nb[1]].parent = [self.current.x, self.current.y]
                        self.grid[nb[0]][nb[1]].gcost = new_gcost
                        self.grid[nb[0]][nb[1]].hcost = \
                            self.update_hcost(self.grid[nb[0]][nb[1]])
                        self.grid[nb[0]][nb[1]].fcost = \
                            update_fcost(self.grid[nb[0]][nb[1]])

                        if self.grid[nb[0]][nb[1]] not in open_list:
                            open_list.append(self.grid[nb[0]][nb[1]])
                    else:
                        self.grid[nb[0]][nb[1]].parent = old_parent

        path = []
        pub_path = NodeMatrix()
        while self.current.x != self.start.x or self.current.y != self.start.y:
            path.append(self.current)
            pub_path.matrix.append(NodeRow(row=[self.current.x, self.current.y]))
            self.current = self.grid[self.current.parent[0]][self.current.parent[1]]

        self.path_pub.publish(pub_path)

        return path


if __name__ == '__main__':
    # m = 10
    # n = 12
    # start = Node(1, 1)
    # end = Node(7, 9)
    pf = Pathfinder()
    rospy.spin()
    # path = pf.a_star()
    # for p in path:
    #     print(p.x, p.y, p.gcost, p.hcost)
    # # print(path)
    # print(path[-1].x, path[-1].y)