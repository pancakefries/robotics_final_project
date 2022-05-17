#!/usr/bin/env python3

from cgi import test
import rospy 
import numpy as np
from nav_msgs.msg import OccupancyGrid

class Node(object):
    def __init__(self, x: int, y: int):
        self.fcost = 1000
        self.hcost = 1000
        self.gcost = 1000
        self.x = x
        self.y = y
        self.parent = None

class Pathfinder(object):
    def __init__(self, width: int, height: int, start: Node, end: Node):
        self.current = Node(start.x, start.y)
        self.current.parent = [self.current.x, self.current.y]
        self.start = Node(start.x, start.y)
        self.start.parent = [start.x, start.y]
        self.goal = end
        self.grid = []
        for i in range(width):
            row = []
            for j in range(height):
                row.append(Node(i, j))
            self.grid.append(row)

        self.map = None
        self.width = width
        self.height = height

        # TODO: create map using SLAM and import it for obstacle data
        # rospy.Subscriber("map", OccupancyGrid, self.get_map)
        # rospy.sleep(2)

    def get_map(self, data):
        self.map = data
        # self.width = data.info.width
        # self.height = data.info.height

    def get_neighbors(self, current):
        # Get the 8 coordinates surrounding the current cell
        offsets = [-1, 0, 1]
        neighbors = []
        for x_offset in offsets:
            for y_offset in offsets:
                # Ensure that they fall within the bounds of the board
                if (x_offset != y_offset or x_offset != 0) and \
                    (current.x + x_offset < self.width) and \
                    (current.y + y_offset < self.height) and \
                    (current.x + x_offset >= 0) and (current.y + y_offset >= 0):
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
            
            for nb in self.get_neighbors(self.current):
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
        while self.current.x != self.start.x or self.current.y != self.start.y:
            path.append(self.current)
            self.current = self.grid[self.current.parent[0]][self.current.parent[1]]
        return path


if __name__ == '__main__':
    m = 10
    n = 12
    start = Node(1, 1)
    end = Node(7, 9)
    pf = Pathfinder(m, n, start, end)
    path = pf.a_star()
    for p in path:
        print(p.x, p.y, p.gcost, p.hcost)
    print(path)