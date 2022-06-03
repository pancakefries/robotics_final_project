#!/usr/bin/env python3

import rospy 
import numpy as np
from robotics_final_project.msg import NodeRow, NodeMatrix
from pathfinding import Node
import math
from geometry_msgs.msg import Vector3, Twist, Point, PoseWithCovarianceStamped
import cv2, cv_bridge
import moveit_commander
from sensor_msgs.msg import Image,LaserScan
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid


def convert_hsv (h, s, v):
    new_h = h/2 - 1
    new_s = s * 256 - 1
    new_v = v * 256 - 1
    return np.array([new_h, new_s, new_v])

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

color_index = {"green" : 0, "blue": 1, "pink": 2}
lower_colors = [convert_hsv(75, 0.40, 0.40), convert_hsv(180, 0.40, 0.40), convert_hsv(300, 0.40, 0.40)] 
upper_colors = [convert_hsv(80, 1, 1), convert_hsv(200, 1, 1), convert_hsv(330, 1, 1)]

class RobotController(object):
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.theta = 0

        # Dummy coords for goal for testing
        self.goal_x = 1
        self.goal_y = 1
        self.goal_theta = 0

        self.player_x = 0
        self.player_y = 0
        self.end_x = 0
        self.end_y = 0

        self.linspeed = 0
        self.linspeed_max = 0.02
        self.angspeed = 0
        self.angspeed_scale = 0.2

        self.received_response = 1
        self.arrived_at_goal = 0
        self.goal_type = "flag" # options are flag, player, and goal
        self.player_returning = 0

        # Set to whatever color "flag" is being used (green, blue, pink)
        self.current_target = "green"
        self.target_in_view = False
        self.horizontal_error = 0
        self.distance_error = 0
        self.arrived_at_target_counter = 0

        self.holding_item = False

        self.map_initialized = False

        lin = Vector3()
        ang = Vector3()
        self.twist = Twist(linear=lin,angular=ang)

        rospy.init_node("target_nodes")
        rospy.Subscriber("target_path", NodeMatrix, self.update_goal)
        rospy.Subscriber("returned", Point, self.player_returned)
        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.update_current_pos)
        rospy.Subscriber("map", OccupancyGrid, self.get_map)

        self.node_pub = rospy.Publisher("target_nodes", NodeMatrix, queue_size=10)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.return_pub = rospy.Publisher("player_return", Point, queue_size=10)

        # Comment out below if using turtlebot without arm (for testing)
        # self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        # self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.bridge = cv_bridge.CvBridge()

        print("RC: initialized")

    def get_map(self, data):
        self.map = data
        map_info = self.map.info
        self.map_data = self.map.data
        self.map_resolution = map_info.resolution
        self.map_width = map_info.width
        self.map_height = map_info.height
        self.map_origin = map_info.origin
        print("RC: map initialized")
        self.map_initialized = True

    # Most of the code for finding, picking up, and placing the flag is
    # adapted from my group's implementatino of the Q-Learning Project

    def update_movement_flag(self):
        # If can see target (cam), turn toward it
        if (self.target_in_view):
            turn_speed = (- self.horizontal_error) * 0.4
            self.twist.angular.z = turn_speed
            self.twist.linear.x = 0
        else:
            self.twist.angular.z = 0.4
            self.twist.linear.x = 0

        # If target in center, approach
        if (self.target_in_view and self.horizontal_error < 0.1):
            forward_speed = (self.distance_error + 0.021) * 0.5
            self.twist.linear.x = forward_speed

        # If centered and close, increment distance
        if (self.horizontal_error < 0.1 and self.distance_error == 0):
            self.arrived_at_target_counter += 1
        else:
            self.arrived_at_target_counter = max(0, self.arrived_at_target_counter - 1)
        
        print("Counter: ",self.arrived_at_target_counter)

        # Will need to tinker with threshold to stop at right distance
        if (self.arrived_at_target_counter > 100 and self.holding_item == False and self.goal_type == 'flag'):    
            self.holding_item = True
            self.pick_up_baton()
        elif (self.arrived_at_target_counter > 100 and self.holding_item == True and self.goal_type == 'goal'):
            self.holding_item = False
            self.place_baton()

    def image_callback(self, msg):
        if self.goal_type == "flag" and self.arrived_at_goal:
            # Converts the ROS message to OpenCV format and HSV (hue, saturation, value)
            image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            h, w, d = image.shape
                    
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            i = color_index[self.current_target]

            # Generate mask
            mask = cv2.inRange(hsv, lower_colors[i], upper_colors[i])

            M = cv2.moments(mask)

            # If any target color pixels are found
            if M['m00'] > 0:
                # Center of the target color pixels in the image
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # Visualize red circle in center
                cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
                self.target_in_view = True

                self.horizontal_error = (cx - w/2) / (w /2)

            else:
                self.target_in_view = False
                self.horizontal_error = 10
            
            print("RC: image callback")
            self.update_movement_flag()

        else:
            self.horizontal_error = 0

    def pick_up_baton(self):
        print("Picking up baton")
        # Move arm back
        arm_goal = [0.0, np.radians(-93), np.radians(62), np.radians(33)]
        self.move_group_arm.go(arm_goal, wait=True)
        self.move_group_arm.stop()

        # Open gripper
        gripper_goal = [0.019, -0.019]
        self.move_group_gripper.go(gripper_goal, wait=True)
        self.move_group_gripper.stop()

        # Stretch arm out to grab 
        arm_goal = [0.0, np.radians(38), np.radians(-3), np.radians(-29)]
        self.move_group_arm.go(arm_goal, wait=True)
        self.move_group_arm.stop()
        # Move forward to avoid bumping dumbbell in gazebo
        rospy.sleep(1)

        # Grab dumbbell
        gripper_goal = [-0.008, 0.008]
        self.move_group_gripper.go(gripper_goal, wait=True)
        self.move_group_gripper.stop()

        # Lift up
        arm_goal = [0.0, np.radians(-23), np.radians(-33), np.radians(33)]
        self.move_group_arm.go(arm_goal, wait=True)
        self.move_group_arm.stop()

        self.goal_type = "goal"
        self.target_in_view = False
        self.arrived_at_target_counter = 0
        self.arrived_at_goal = False

    def place_baton(self):
        print("Placing Baton")
        # Will have to tinker with sleep to approach at proper distance
        rospy.sleep(3)
        # Extend to place
        arm_goal = [0.0, np.radians(38), np.radians(-3), np.radians(-29)]
        self.move_group_arm.go(arm_goal, wait=True)
        self.move_group_arm.stop()
        
        # Open gripper
        gripper_goal = [0.019, -0.019]
        self.move_group_gripper.go(gripper_goal, wait=True)
        self.move_group_gripper.stop()

        # Move arm back to neutral
        arm_goal = [0.0, 0.0, 0.0, 0.0]
        self.move_group_arm.go(arm_goal, wait=True)
        self.move_group_arm.stop()

        gripper_goal = [0, 0]
        self.move_group_gripper.go(gripper_goal, wait=True)
        self.move_group_gripper.stop()

        self.holding_item = False
        self.arrived_at_target_counter = 0
        self.target_in_view = False
        self.current_target = None
        # Updating goal type probably not necessary since robot will have won
        self.goal_type = "player"
        self.arrived_at_goal = False

    def scan_callback(self, data):
        if self.goal_type == "flag" and self.arrived_at_goal:
            center_average = 0
            angles = [359, 0, 1]
            for i in angles:
                if data.ranges[i] == 0.0:
                    center_average += 0.01
                if data.ranges[i] > 4:
                    center_average += 0.01
                else:
                    center_average += data.ranges[i]
            center_average = center_average / len(angles)
            if center_average != 0.0 and center_average > 0.4:
                self.distance_error = center_average / 3.5
            else:
                self.distance_error = 0
            self.update_movement_flag()
            print("RC: scan callback")

    def update_goal(self, data):
        self.received_response = 1
        print("matrix length: ", len(data.matrix))
        # The commented code can be used to follow the received path for 
        # longer than one cell
        # l = 3
        # if len(data.matrix) < l:
        #     l = len(data.matrix)
        # for i in range(l):
        #     rospy.sleep(0.3)
        self.goal_x = data.matrix[-1].row[0]
        self.goal_y = data.matrix[-1].row[1]
        self.goal_theta = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        self.update_movement()

    def update_current_pos(self, data):
        pos = data.pose.pose.position
        self.x = pos.x
        self.y = pos.y
        self.theta = get_yaw_from_pose(data.pose.pose)

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
        # print("RC: deciding target")
        # 
        # The odd numbers added to the positions are offsets for converting ot map coords
        if self.received_response and self.map_initialized:
            dist_player = math.sqrt(pow(abs(self.x - self.player_x), 2) + pow(abs(self.y - self.player_y), 2))
            dist_end = math.sqrt(pow(abs(self.x - self.end_x), 2) + pow(abs(self.y - self.end_y), 2))
            if dist_player < dist_end:
                self.goal_type = "player"
                self.received_response = 0
                rospy.sleep(0.25)
                self.node_pub.publish(NodeMatrix([NodeRow([int((self.y/self.map_resolution) + 244.837626), \
                                                       int((self.x/self.map_resolution) + 172.824563)]), \
                                              NodeRow([int((self.player_y/self.map_resolution) + 244.837626), \
                                                       int((self.player_x/self.map_resolution) + 172.824563)])]))
                while not self.received_response:
                    rospy.sleep(0.1)
            else:
                self.goal_type = "goal"
                self.received_response = 0
                rospy.sleep(0.25)
                self.node_pub.publish(NodeMatrix([NodeRow([int((self.y/self.map_resolution) + 244.837626), \
                                                       int((self.x/self.map_resolution) + 172.824563)]), \
                                              NodeRow([int((self.end_y/self.map_resolution) + 244.837626), \
                                                       int((self.end_x/self.map_resolution) + 172.824563)])]))
                while not self.received_response:
                    rospy.sleep(0.1)


            # self.received_response = 0
            # rospy.sleep(1)
            # self.node_pub.publish(NodeMatrix([NodeRow([int((self.y/self.map_resolution) + 244.837626), \
            #                                            int((self.x/self.map_resolution) + 172.824563)]), \
            #                                   NodeRow([int((self.player_y/self.map_resolution) + 244.837626), \
            #                                            int((self.player_x/self.map_resolution) + 172.824563)])]))
            # while not self.received_response:
            #     rospy.sleep(0.1)
            # dist_player = math.sqrt(pow(abs(self.x - self.goal_x), 2) + pow(abs(self.y - self.goal_y), 2))
            # self.received_response = 0
            # self.node_pub.publish(NodeMatrix([NodeRow([int((self.y/self.map_resolution) + 244.837626), \
            #                                            int((self.x/self.map_resolution) + 172.824563)]), \
            #                                   NodeRow([int((self.end_y/self.map_resolution) + 244.837626), \
            #                                            int((self.end_x/self.map_resolution) + 172.824563)])]))
            # while not self.received_response:
            #     rospy.sleep(0.1)
            # dist_end = math.sqrt(pow(abs(self.x - self.goal_x), 2) + pow(abs(self.y - self.goal_y), 2))

            # if dist_player < dist_end:
            #     self.goal_type = "player"
            # else:
            #     self.goal_type = "goal"
            # # print("RC: decided target")

    def update_movement(self):
        # Use proportional control to navigate to goal, accuracy is not super 
        # important since it is constantly being updated
        # TODO: experiment with values for self.angspeed_scale
        # Too high values of angspeed_scale causes overshootingS
        # print("RC: updating movement")
        if self.received_response and self.map_initialized:
            diff = self.goal_theta - self.theta
            # diff *= -1
            # Use piecewise function to control angular speed, if difference between
            # goal and current theta is > 0.6rad, then set to max speed, otherwise
            # scale linearly with the difference
            if abs(diff) > 0.6:
                self.angspeed = self.angspeed_scale * (diff / abs(diff))
            else:
                self.angspeed = (diff / np.pi) * self.angspeed_scale
            dist = math.sqrt(pow(abs(self.x - self.goal_x), 2) + pow(abs(self.y - self.goal_y), 2))
            # Use a threshold to stop so that the bot doesn't overshoot
            if dist >= 0 and dist <= 0.2:
                print("RC: arrived at goal")
                self.linspeed = 0
                self.arrived_at_goal = 1
            else:
                print("RC: not yet at goal")
                self.linspeed = self.linspeed_max
                self.arrived_at_goal = 0
            # print("RC: updated movement")

    def player_returned(self, data):
        self.player_returning = 0

    def run(self):
        l = Vector3(x = self.linspeed, y = 0, z = 0)
        a = Vector3(x = 0, y = 0, z = self.angspeed)
        while not rospy.is_shutdown():
            self.decide_current_target()
            self.update_movement()
            if self.arrived_at_goal:
                print("RC: reached goal")
                if self.goal_type == "player":
                    print("RC: player return")
                    l.x = 0
                    a.z = 0
                    self.vel_pub.publish(Twist(linear = l, angular = a))
                    # Calling this while running player_return.py will display a
                    # prompt for the player to accept once they've returned to start
                    self.return_pub.publish(Point(1, 1))
                    self.player_returning = 1
                    while self.player_returning:
                        rospy.sleep(0.2)
                else:
                    print("RC: waiting to pickup/drop")
                    # Pause main loop and navigation to flag and pickup/drop will take
                    # over due to scan and camera callback functions, resume when done
                    while self.arrived_at_goal:
                        rospy.sleep(0.5)
                self.arrived_at_goal = 0
            # The angular and linear speeds are constantly updated asynchronously
            l.x = self.linspeed
            a.z = self.angspeed
            self.vel_pub.publish(Twist(linear = l, angular = a))
            # print("published")
        

if __name__ == "__main__":
    rc = RobotController()
    rc.run()
    