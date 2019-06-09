from .graph import Graph

'''
1: head
2: neck
3: spine
4: left shoulder
5: left elbow
6: left wrist
7: right shoulder
8: right elbow
9: right wrist
10: left hip
11: left knee
12: left ankle
13: right hip
14: right knee
15: right ankle
'''

num_node = 15
inward_ori_index = [(2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 2), (8, 7),
                    (9, 8), (10, 3), (11, 10), (12, 11), (13, 3), (14, 13),
                    (15, 14)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# Head: 1, 2, 4, 7
head = [(1, 0), (2, 1), (3, 1), (6, 1)]
# LHand: 4, 5, 6
lefthand = [(4, 3), (5, 4)]
# RHand: 7, 8, 9
righthand = [(7, 6), (8, 7)]
hands = lefthand + righthand
# Torso: 2, 3, 4, 7, 10, 13
torso = [(2, 1), (3, 1), (6, 1), (9, 2), (12, 2)]
# Lleg: 3, 10, 11, 12
leftleg = [(9, 2), (10, 9), (11, 10)]
# Rleg: 3, 13, 14, 15
rightleg = [(12, 2), (13, 12), (14, 13)]
legs = leftleg + rightleg

class Flo3DGraph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(Flo3DGraph, self).__init__(num_node=num_node,
                                         inward=inward,
                                         outward=outward,
                                         parts=[head, hands, torso, legs],
                                         labeling_mode=labeling_mode)
