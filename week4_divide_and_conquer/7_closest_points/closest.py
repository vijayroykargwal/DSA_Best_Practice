#Uses python3
import copy
import sys
import math
def dist(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
def brute_force(points,n):
    min_val = sys.maxsize
    for i in range(n):
        for j in range(i+1,n):
            if dist(points[i],points[j]) < min_val:
                min_val = dist(points[i],points[j])
    return min_val
def strip_closest(strip,size,d):
    min_val = d
    for i in range(size):
        j = i+1
        while j < size and (strip[j][1]-strip[i][1]) < min_val:
            min_val = dist(strip[i],strip[j])
            j+=1
    return min_val
def closet_util(points,re_points,n):
    if n <= 3:
        return brute_force(points,n)
    mid = n//2
    mid_point = points[mid]
    points_l = points[:mid]
    points_r = points[mid:]
    dl = closet_util(points_l,re_points,mid)
    dr = closet_util(points_r, re_points, n-mid)
    d = min(dl,dr)
    strip_P  = []
    strip_Q = []
    lr = points_l+points_r
    for i in range(n):
        if abs(lr[i][0]-mid_point[0]) < d:
            strip_P.append(lr[i])
        if abs(re_points[i][0]-mid_point[0]) < d:
            strip_Q.append(re_points[i])
    strip_P.sort(key=lambda x:x[1])
    min_a = min(d, strip_closest(strip_P,len(strip_P),d))
    min_b = min(d, strip_closest(strip_Q,len(strip_Q),d))
    return min(min_a,min_b)
def minimum_distance(points,n):
    points.sort(key=lambda x:x[0])
    re_points = copy.deepcopy(points)
    re_points.sort(key=lambda x:x[1])
    return closet_util(points,re_points,len(points))
if __name__ == '__main__':
    n = int(input())
    data = []
    for i in range(n):
        data.append(list(map(int, input().split())))
    print(minimum_distance(data,len(data)))
