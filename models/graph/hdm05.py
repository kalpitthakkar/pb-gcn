from .graph import Graph

'''
1: root
2: lhipjoint
3: lfemur
4: ltibia
5: lfoot
6: ltoes
7: rhipjoint
8: rfemur
9: rtibia
10: rfoot
11: rtoes
12: lowerback
13: upperback
14: thorax
15: lowerneck
16: upperneck
17: head
18: lclavicle
19: lhumerus
20: lradius
21: lwrist
22: lhand
23: lfingers
24: lthumb
25: rclavicle
26: rhumerus
27: rradius
28: rwrist
29: rhand
30: rfingers
31: rthumb
'''

num_node = 31
inward_ori_index = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 1), (8, 7),
                    (9, 8), (10, 9), (11, 10), (1, 12), (12, 13), (13, 14),
                    (14, 15), (15, 16), (16, 17), (18, 14), (19, 18), (20, 19),
                    (21, 20), (22, 21), (23, 22), (24, 21), (25, 14), (26, 25),
                    (27, 26), (28, 27), (29, 28), (30, 29), (31, 28)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# Head: 14, 15, 16, 17, 18, 19, 25, 26
head = [(13, 14), (14, 15), (15, 16), (17, 13), (18, 17), (24, 13), (25, 24)]
# LHand: 18, 19, 20, 21, 22, 23, 24
lefthand = [(18, 17), (19, 18), (20, 19), (21, 20), (22, 21), (23, 20)]
# RHand: 25, 26, 27, 28, 29, 30, 31
righthand = [(25, 24), (26, 25), (27, 26), (28, 27), (29, 28), (30, 27)]
hands = lefthand + righthand
# Torso: 1, 2, 7, 12, 13, 14, 18, 25
torso = [(1, 0), (6, 0), (0, 11), (11, 12), (12, 13), (17, 13), (24, 13)]
# Lleg: 1, 2, 3, 4, 5, 6
leftleg = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]
# Rleg: 1, 7, 8, 9, 10, 11
rightleg = [(6, 0), (7, 6), (8, 7), (9, 8), (10, 9)]
legs = leftleg + rightleg

class HDM05Graph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(HDM05Graph, self).__init__(num_node=num_node,
                                         inward=inward,
                                         outward=outward,
                                         parts=[head, hands, torso, legs],
                                         labeling_mode=labeling_mode)
