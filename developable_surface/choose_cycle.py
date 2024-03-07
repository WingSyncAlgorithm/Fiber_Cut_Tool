def choose_cycle(cycles, high_curvature):
    '''
    Parameter:
    cycles : 二維list,第一維是環的index,第二維存環中的點
    high_curvature : 1D list 存所有點的高斯曲率
    
    Return:
    1D list : 高斯曲率總和最大的環
    '''
    max_cost = 0
    max_cycle_idx = 0
    for idx, cycle in enumerate(cycles):
        cost = 0
        for point_idx in cycle:
            cost += high_curvature[point_idx]
        if cost > max_cost:
            max_cost = cost
            max_cycle_idx = idx
    return cycles[max_cycle_idx]