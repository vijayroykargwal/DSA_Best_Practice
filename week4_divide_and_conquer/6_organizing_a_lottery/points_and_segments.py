# Uses python3
import sys

def fast_count_segments(starts, ends, points):
    cnt = [0] * len(points)
    #write your code here
    point = []
    seg = []
    for i in range(len(points)):
        point.append((points[i],i))
    for i in range(len(starts)):
        seg.append((starts[i],1))
        seg.append((ends[i],-1))
    seg.sort(reverse=True)
    point.sort()
    count = 0
    for i in range(len(points)):
        x = point[i][0]
        while len(seg)!=0 and seg[len(seg)-1][0] <= x:
            count += seg[len(seg)-1][1]
            seg.remove(seg[len(seg)-1])
        cnt[point[i][1]] = count
    return cnt

def naive_count_segments(starts, ends, points):
    cnt = [0] * len(points)
    for i in range(len(points)):
        for j in range(len(starts)):
            if starts[j] <= points[i] <= ends[j]:
                cnt[i] += 1
    return cnt

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    m = data[1]
    starts = data[2:2 * n + 2:2]
    ends = data[3:2 * n + 2:2]
    points = data[2 * n + 2:]
    # use fast_count_segments
    cnt = fast_count_segments(starts, ends, points)
    for x in cnt:
        print(x, end=' ')
