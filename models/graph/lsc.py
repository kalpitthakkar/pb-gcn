from graph import Graph

''' 20 joints
1: base of spine
2: middle of spine
3: neck
4: head
5: left shoulder
6: left elbow
7: left wrist
8: left hand
9: right shoulder
10: right elbow
11: right wrist
12: right hand
13: left hip
14: left knee
15: left ankle
16: left foot
17: right hip
18: right knee
19: right ankle
20: right foot
'''

''' 15 joints (Like CAD-60)
1 -> HEAD
2 -> NECK
3 -> TORSO
4 -> LEFT_SHOULDER
5 -> LEFT_ELBOW
6 -> RIGHT_SHOULDER
7 -> RIGHT_ELBOW
8 -> LEFT_HIP
9 -> LEFT_KNEE
10 -> RIGHT_HIP
11 -> RIGHT_KNEE
12 -> LEFT_HAND
13 -> RIGHT_HAND
14 -> LEFT_FOOT
15 -> RIGHT_FOOT
'''

# In order to use similar batchnorm and masks, keep the size of Adj same -> (20, 20)
num_node = [20, 20]
inward_ori_index = [[(1, 2), (2, 3), (3, 4), (5, 3), (6, 5), (7, 6), (8, 7),
                    (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                    (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)],
                    [(3, 2), (2, 1), (4, 2), (5, 4), (12, 5), (6, 2), (7, 6), (13, 7),
                    (3, 8), (8, 9), (9, 14), (3, 10), (10, 11), (11, 15)]]
inward = [[(i - 1, j - 1) for (i, j) in inward_ori_index[k]] for k in range(len(num_node))] 
outward = [[(j, i) for (i, j) in inward[k]] for k in range(len(num_node))]
neighbor = [inward[i] + outward[i] for i in range(len(num_node))]

# Head: 3, 4, 5, 9 / 1, 2, 4, 6
head = [[(2, 3), (2, 4), (2, 8)], [(0, 1), (1, 3), (1, 5)]]
# LHand: 5, 6, 7, 8 / 4, 5, 12
lefthand = [[(4, 5), (5, 6), (6, 7)], [(3, 4), (4, 11)]]
# RHand: 9, 10, 11, 12 / 6, 7, 13
righthand = [[(8, 9), (9, 10), (10, 11)], [(5, 6), (6, 12)]]
hands = [lefthand[i] + righthand[i] for i in range(len(num_node))]
# Torso: 1, 2, 3, 5, 9, 13, 17 / 2, 3, 4, 6, 8, 10
torso = [[(0, 1), (1, 2), (2, 4), (2, 8), (0, 12), (0, 16)], [(1, 2), (1, 3), (1, 5), (2, 7), (2, 9)]]
# Lleg: 1, 13, 14, 15, 16 / 3, 8, 9, 14
leftleg = [[(0, 12), (12, 13), (13, 14), (14, 15)], [(2, 7), (7, 8), (8, 13)]]
# Rleg: 1, 17, 18, 19, 20 / 3, 10, 11, 15
rightleg = [[(0, 16), (16, 17), (17, 18), (18, 19)], [(2, 9), (9, 10), (10, 14)]]
legs = [leftleg[i] + rightleg[i] for i in range(len(num_node))]

class LSCGraph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(LSCGraph, self).__init__(num_node=num_node,
                                         inward=inward,
                                         outward=outward,
                                         parts=[[head[i], hands[i], torso[i], legs[i]] for i in range(len(num_node))],
                                         labeling_mode=labeling_mode)