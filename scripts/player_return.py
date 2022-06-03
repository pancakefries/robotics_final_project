#!/usr/bin/env python3

import rospy 
from geometry_msgs.msg import Point

class Player_Return(object):
    def __init__(self):
        rospy.init_node("player_return")
        rospy.Subscriber("player_return", Point, self.ask)
        self.pub = rospy.Publisher("returned", Point, queue_size=10)

    def ask(self, data):
        i = 'n'
        while i == 'n':
            input("Have you returned to the start? y/n: ")
        self.pub.publish(Point(0, 0))
        
if __name__ == "__main__":
    pr = Player_Return()
    rospy.spin()
