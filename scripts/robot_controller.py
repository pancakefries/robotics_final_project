#!/usr/bin/env python3

import rospy 
import numpy as np
from robotics_final_project.msg import NodeRow, NodeMatrix
from pathfinding import Node
import math
from geometry_msgs.msg import Vector3, Twist


class RobotController(object):
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.theta = 0

        self.goal_x = 0
        self.goal_y = 0
        self.goal_theta = 0

        self.linspeed = 0
        self.linspeed_max = 0.7
        self.angspeed = 0
        self.angspeed_scale = 3

        self.received_response = 1

        rospy.init_node("target_nodes")
        rospy.Subscriber("target_path", NodeMatrix, self.update_goal)
        # TODO: implement subscriber for node that uses PFL replacement library
        self.node_pub = rospy.Publisher("target_nodes", NodeMatrix, queue_size=10)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        print("RC: initialized")
    
    def update_goal(self, data):
        self.received_response = 1
        self.goal_x = data.matrix[-1].row[0]
        self.goal_y = data.matrix[-1].row[1]
        self.goal_theta = math.atan(self.goal_y - self.y, self.goal_x - self.x)
        # print("updated_goal")

    def update_current_pos(self):
        # TODO: update current x and y 
        pass

    def decide_current_target(self):
        # TODO: if attempting to tag opponent takes a shorter time than
        # attempting to reach the goal then try to tag opponent
        #
        # publish location of which target to pursue, then pathfinding.py will
        # return the proper goal path, i.e. send goal to pathfinding, then
        # update_goal will make sure it's always accurate
        #
        # ensure that it only gets called after it receives a path from
        # pathfinding, set a boolean value and check each call, since it will
        # be called over and over again in run()
        # if self.received_response:
        #     self.received_response = 0
        #     execute above
        # print("decided target")
        pass

    def path_to_map(self, data):
        # TODO: implement coordinate transform between pathfinding and map
        return data

    def map_to_path(self, data):
        # TODO: implement coordinate transform between map and pathfinding
        return data

    def update_movement(self):
        # Use proportional control to navigate to goal, accuracy is not super 
        # important since it is constantly being updated, so odom is not necessary
        # TODO: experiment with values for self.angspeed_scale
        self.angspeed = (self.theta - self.goal_theta) * self.angspeed_scale
        dist = math.sqrt(pow(abs(self.x - self.goal_x), 2) + pow(abs(self.y - self.goal_y), 2))
        # Use a threshold to stop so that the bot doesn't overshoot
        if dist >= 0 and dist <= 0.2:
            self.linspeed = 0
        else:
            self.linspeed = self.linspeed_max
        # print("updated movement")

    def run(self):
        l = Vector3(x = self.linspeed, y = 0, z = 0)
        a = Vector3(x = 0, y = 0, z = self.angspeed)
        while not rospy.is_shutdown():
            self.decide_current_target()
            self.update_current_pos()
            self.update_movement()
            l.x = self.linspeed
            a.z = self.angspeed
            self.vel_pub.publish(Twist(linear = l, angular = a))
            # print("published")
        

if __name__ == "__main__":
    rc = RobotController()
    rc.run()