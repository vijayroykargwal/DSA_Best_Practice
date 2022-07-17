Q.28.Given n integers, which form a circle. Find the minimal absolute value of any adjacent pair. If there are many optimum solutions, output any of them. 
A = [10, 12, 13, 15, 10]
B = [10, 20, 30, 40]

def min_AdjDiff(arr):
    if len(arr) < 2:
        return -1
    diff = abs(arr[1]-arr[0])
    for i in range(2,len(arr)):
        diff = min(abs(arr[i]-arr[i-1]), diff)
    diff = min(diff, abs(arr[0]-arr[len(arr)-1]))
    return diff
print(min_AdjDiff(A))
print(min_AdjDiff(B))

Q.29.Given two arrays a[] and b[], we need to build an array c[] such that every element c[i] of c[] contains a value from a[] which is greater than b[i] and is closest to b[i]. If a[] has no greater element than b[i], then value of c[i] is -1. All arrays are of same size.

A = [2, 6, 5, 7, 0]
B = [1, 3, 2, 5, 8]
import bisect
def closest_ele(arr1,arr2):
    arr1.sort()
    c = []
    for i in range(len(arr1)):
        up = bisect.bisect_right(arr1,arr2[i])
        if up==len(arr1):
            c.append(-1)
        else:
            c.append(arr1[up])
    return c
print(closest_ele(A,B))

Q.30.Given an unsorted array of n integers that can contain integers from 1 to n. Some elements can be repeated multiple times and some other elements can be absent from the array. Count the frequency of all elements that are present and print the missing elements.
A = [2, 3, 3, 2, 5]
B = [4, 4, 4, 4]

def count_freq(arr):
    for i in range(len(arr)):
        arr[i] = arr[i]-1
    for j in range(len(arr)):
        arr[arr[j]%len(arr)] = arr[arr[j]%len(arr)]+len(arr)
    for l in range(len(arr)):
        print(l+1, arr[l]//len(arr))
print(count_freq(A))
print(count_freq(B))

Q.31.Given an array of N integers and an integer K, pick two distinct elements whose sum is K and find the maximum shortest distance of the picked elements from the endpoints.
A = [2, 4, 3, 2, 1]
B = [2, 4, 1, 9, 5]
from sys import maxsize
def max_shortest_dist(arr,k):
    map = {}
    for i in range(len(arr)):
        x = arr[i]
        d = min(1+i,len(arr)-i)
        if x not in map:
            map[x] = d
        else:
            map[x] = min(d,map[x])
    ans = maxsize
    for j in range(len(arr)):
        x = arr[j]
        if x != k-x and k-x in map:
            ans = min(max(map[x],map[k-x]),ans)
    return ans

k=5
print(max_shortest_dist(A,k))
k=3
print(max_shortest_dist(B,k))

Q.32.Given an array arr[] of size n. Three elements arr[i], arr[j] and arr[k] form an inversion of size 3 if a[i] > a[j] >a[k] and i < j < k. Find total number of inversions of size 3.
A = [8, 4, 2, 1]
B = [9, 6, 4, 5, 8]

def inv_count(arr):
    count = 0
    for i in range(1,len(arr)-1):
        small = 0
        for j in range(i+1,len(arr)):
            if arr[i] > arr[j]:
                small+=1
        great = 0
        for k in range(i-1,-1,-1):
            if arr[i] < arr[k]:
                great+=1
        count += great*small
    return count

print(inv_count(A))
print(inv_count(B))

Q.33.Given an array of integers, and a number ‘sum’, find the number of pairs of integers in the array whose sum is equal to ‘sum’.
A = [1, 5, 7, -1]
B = [1, 5, 7, -1, 5]

def count_sum(arr,sum):
    count = 0
    map = {}
    for i in range(len(arr)):
        if sum-arr[i] in map:
            count += map[sum-arr[i]]
        if arr[i] in map:
            map[arr[i]] += 1
        else:
            map[arr[i]] = 1
    return count
sum = 6
print(count_sum(A,sum))
print(count_sum(B,sum))

Q.34.Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
A = [3, 0, 2, 0, 4]
B = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]

def max_rainWater(arr):
    left = 0
    right = len(arr)-1
    l_max = 0
    r_max = 0
    result = 0
    while left <= right:
        if r_max <= l_max:
            result+= max(0, r_max - arr[right])
            r_max = max(r_max, arr[right])
            right -= 1
        else:
            result += max(0, l_max-arr[left])
            l_max = max(l_max,arr[left])
            left += 1
    return result


print(max_rainWater(A))
print(max_rainWater(B))

Q.35.Given an array of positive distinct integers. We need to find the only element whose replacement with any other value makes array elements distinct consecutive. If it is not possible to make array elements consecutive, return -1.
A = [45, 42, 46, 48, 47]
B = [5, 6, 7, 9, 10]
from bisect import bisect_left
def binary_search(arr,x):
    i = bisect_left(arr,x)
    if i != len(arr) and arr[i]==x:
        return i
    else:
        return -1
def replace_ele(arr):
    arr.sort()
    res= 0
    next_ele = arr[len(arr)-1]-len(arr)+1
    count_starting = 0
    for i in range(len(arr)-1):
        if (binary_search(arr,next_ele)==-1):
            res = arr[0]
            count_starting +=1
        next_ele += 1
    if count_starting == 1:
        return res
    if count_starting ==0:
        return 0
    count_ending = 0
    next_ele = arr[0]+len(arr)-1
    for j in range(len(arr)-1,0,-1):
        if(binary_search(arr,next_ele)==-1):
            res = arr[len(arr)-1]
            count_ending += 1
        next_ele -=1
    if count_ending == 1:
        return res
    return -1

print(replace_ele(A))
print(replace_ele(B))

Q.36.Given an increasing sequence a[], we need to find the K-th missing contiguous element in the increasing sequence which is not present in the sequence. If no k-th missing element is there output -1. 
using hashmap:
A = [2, 3, 5, 9, 10]
B = [2, 3, 5, 9, 10, 11, 12]
from sys import maxsize
def kth_missing(arr,k):
    map ={}
    maxi = -maxsize-1
    mini = 1
    missing_count = 0
    for i in range(len(arr)):
        map[arr[i]]=1
        if arr[i] > maxi:
            maxi = arr[i]
        if arr[i] < mini:
            mini = arr[i]
    for j in range(mini,maxi+1):
        if j not in map:
            missing_count += 1
        if missing_count == k:
            return j
    return -1


print(kth_missing(A,1))
print(kth_missing(B,4))

Using Binary Search:
A = [2, 3, 5, 9, 10]
B = [2, 3, 5, 9, 10, 11, 12]
from sys import maxsize
def kth_missing(arr,k):
    if k < arr[0]:
        return k
    l = 0
    r = len(arr) - 1
    while l <=r:
        mid = (l+r)//2
        if arr[mid]-(mid+1) < k:
            l = mid+1
        else:
            r = mid-1
    return l+k

print(kth_missing(A,1))
print(kth_missing(B,4))

Q.37.Given two sorted arrays of distinct elements, we need to print those elements from both arrays that are not common. The output should be printed in sorted order. 
using merge sort:o(n1+n2):
A = [10, 20, 30]
B = [20, 25, 30, 40, 50]

def uncommon_ele(arr1,arr2):
    i = 0
    j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            print(arr1[i])
            i += 1
        elif arr2[j] < arr1[i]:
            print(arr2[j])
            j+=1
        else:
            i+=1
            j+=1
    while i < len(arr1):
        print(arr1[i])
        i += 1
    while j < len(arr2):
        print(arr2[j])
        j+=1

print(uncommon_ele(A,B))

Q.38.Given an array of n integers and a number m, find the maximum possible difference between two sets of m elements chosen from given array.
A = [1 ,2, 3, 4 ,5]
B = [5, 8, 11, 40, 15]

def max_diff(arr,m):
    arr.sort()
    j = len(arr)-1
    min_sum,max_sum=0,0
    for i in range(m):
        min_sum += arr[i]
        max_sum += arr[j]
        j -= 1
    return max_sum - min_sum
print(max_diff(A,4))
print(max_diff(B,2))

Q.39.Given n arrays of size m each. Find the maximum sum obtained by selecting a number from each array such that the elements selected from the i-th array are more than the element selected from (i-1)-th array. If maximum sum cannot be obtained then return 0.
A = [[1, 7, 3, 4],[4, 2, 5, 1],[9, 5, 1, 8]]
B = [[9, 8, 7],[6, 5, 4],[3, 2, 1]]
from sys import maxsize
def max_sum(arr,m):
    prev_max = max(max(arr))
    maxsum = prev_max
    for i in range(len(arr)-2,-1,-1):
        max_val = -maxsize-1
        for j in range(m-1,-1,-1):
            if arr[i][j] < prev_max and arr[i][j] > max_val:
                max_val = arr[i][j]
        if max_val == -maxsize-1:
            return 0
        prev_max = max_val
        maxsum += max_val
    return maxsum
m =4
print(max_sum(A,m))
m = 3
print(max_sum(B,m))

Q.40.Given an array A[ ]            of size N. Find the number of pairs (i, j) such that A_i            XOR A_j            = 0, and 1 <= i < j <= N.
A = [1, 3, 4, 1, 4]
B = [2, 2, 2]

def XOR_Pair(arr):
    map = {}
    for i in arr:
        if i in map:
            map[i] += 1
        else:
            map[i] = 1
    pair_count = 0
    for i in map:
        pair_count += map[i]*(map[i]-1)//2
    return pair_count

print(XOR_Pair(A))
print(XOR_Pair(B))

Q.41.You are given an array of n-elements with a basic condition that occurrence of greatest element is more than once. You have to find the minimum distance between maximums. (n>=2).
A = [3, 5, 2, 3, 5, 3, 5]
B = [1, 1, 1, 1, 1, 1]

def min_dist(arr):
    max_ele = arr[0]
    minDist = len(arr)
    idx = 0
    for i in range(1,len(arr)):
        if max_ele == arr[i]:
            minDist = min(minDist, i - idx)
            idx = i
        elif max_ele < arr[i]:
            max_ele = arr[i]
            minDist = len(arr)
            idx = i
        else:
            continue
    return minDist

print(min_dist(A))
print(min_dist(B))

Q.42.Given an array of numbers, find the number among them such that all numbers are divisible by it. If not possible print -1.
A = [25, 20, 5, 10, 100]
B = [9, 3, 6, 2, 15]

def smallest_div(arr):
    smallest = min(arr)
    for i in arr:
        if i % smallest != 0:
            return -1
    return smallest

print(smallest_div(A))
print(smallest_div(B))

Q.43.Given heights of consecutive buildings, find the maximum number of consecutive steps one can put forward such that he gain a increase in altitude while going from roof of one building to next adjacent one.
A = [1, 2, 2, 3, 2]
B = [1, 2, 3, 4]

def max_cons_step(arr):
    count, maxi =0,0
    for i in range(1,len(arr)):
        if arr[i] > arr[i-1]:
            count += 1
        else:
            maxi = max(maxi,count)
            count = 0
    return max(maxi,count)

print(max_cons_step(A))
print(max_cons_step(B))

Q.44.Given an array of even number of elements, form groups of 2 using these array elements such that the difference between the group with highest sum and the one with lowest sum is maximum.
A = [1, 4, 9, 6]
B = [6, 7, 1, 11]
from sys import maxsize
def max_diff(arr):
    first_min = maxsize
    second_min = maxsize
    for i in range(len(arr)):
        if arr[i] < first_min:
            second_min = first_min
            first_min = arr[i]
        elif arr[i] < second_min and arr[i] != first_min:
            second_min = arr[i]
    first_max = -maxsize-1
    second_max = -maxsize-1
    for j in range(len(arr)):
        if arr[j] > first_max:
            second_max = first_max
            first_max = arr[j]
        elif arr[j] > second_max and arr[j] != first_max:
            second_max = arr[j]
    return abs(first_max+second_max-first_min-second_min)

print(max_diff(A))
print(max_diff(B))

Q.45.Given an array of even number of elements, form groups of 2 using these array elements such that the difference between the group with the highest sum and the one with the lowest sum is minimum. 
A = [2, 6, 4, 3]
B = [11, 4, 3, 5, 7, 1]
from sys import maxsize
def min_diff(arr):
    arr.sort()
    i = 0
    j = len(arr)-1
    sum_pair = []
    while i < j:
        sum_pair.append(arr[i]+arr[j])
        i += 1
        j -=1
    min_sum = min(sum_pair)
    max_sum = max(sum_pair)
    return abs(max_sum-min_sum)

print(min_diff(A))
print(min_diff(B))

Q.46.Given a list of distinct unsorted integers, find the pair of elements that have the smallest absolute difference between them? If there are multiple pairs, find them all
A = [10, 50, 12, 100]
B = [5, 4, 3, 2]
from sys import maxsize
def minDiff_pair(arr):
    if len(arr) <= 1:
        return -1
    arr.sort()
    min_diff = arr[1]-arr[0]
    for i in range(2,len(arr)):
        min_diff = min(min_diff,arr[i]-arr[i-1])
    for j in range(1,len(arr)):
        if arr[j]-arr[j-1]==min_diff:
            print(arr[j-1],arr[j])


print(minDiff_pair(A))
print(minDiff_pair(B))

Q.47.Given an unsorted array A of N integers, A_{1}, A_{2}, ...., A_{N}.        Return maximum value of f(i, j) for all 1 ≤ i, j ≤ N. 
f(i, j) or absolute difference of two elements of an array A is defined as |A[i] – A[j]| + |i – j|, where |A| denotes 
the absolute value of A.
A = [1, 3, -1]
B = [3, -2, 5, -4]
from sys import maxsize
def max_absDiff(arr):
    max1 = -maxsize-1
    max2 = -maxsize-1
    min1 = maxsize
    min2 = maxsize
    for i in range(len(arr)):
        max1 = max(max1, arr[i]+i)
        min1 = min(min1, arr[i]+i)
        max2 = max(max2, arr[i]-i)
        min2 = min(min2,arr[i]-i)
    return max(max1-min1, max2-min2)


print(max_absDiff(A))
print(max_absDiff(B))

Q.48.Given an array of size n and a number k, we need to print first k natural numbers that are not there in the given array.
A = [2, 3, 4]
B = [-2, -3, 4]
from sys import maxsize
def missing_K(arr,k):
    map ={}
    for i in arr:
        map[i] = i
    count = 1
    flag = 0
    for j in range(len(arr)+k):
        if count not in map:
            flag+=1
            print(count)
            if flag == k:
                break
        count += 1

print(missing_K(A,3))
print(missing_K(B,2))

Q.49.Given an array of n integers, we need to find the no. of ways of choosing pairs with maximum difference. 
A = [3, 2, 1, 1, 3]
B = [2, 4, 1, 1]
from sys import maxsize
def choose_maxDiff_Pairs(arr):
    mini = maxsize
    maxi = -maxsize-1
    for i in range(len(arr)):
        maxi = max(maxi,arr[i])
        mini = min(mini, arr[i])
    mini_count = 0
    maxi_count = 0
    for j in range(len(arr)):
        if arr[j] == maxi:
            maxi_count+=1
        if arr[j] == mini:
            mini_count +=1
    if mini == maxi:
        return len(arr)*(len(arr)-1)//2
    else:
        return mini_count*maxi_count

print(choose_maxDiff_Pairs(A))
print(choose_maxDiff_Pairs(B))

Q.50.Given an array a, we have to find the minimum product possible with the subset of elements present in the array. The minimum product can be a single element also.
A = [-1, -1, -2, 4, 3]
B = [-1, 0]
from sys import maxsize
def min_subProduct(arr):
    if len(arr) == 1:
        return arr[0]
    max_neg = -maxsize-1
    min_pos = maxsize
    count_neg = 0
    count_zero = 0
    product =1
    for i in range(0,len(arr)):
        if arr[i] == 0:
            count_zero += 1
            continue
        if arr[i] < 0:
            count_neg += 1
            max_neg = max(max_neg,arr[i])
        if arr[i] > 0:
            min_pos +=1
            min_pos = min(min_pos,arr[i])
        product *= arr[i]
    if count_zero == len(arr) or (count_neg == 0 and count_zero > 0):
        return 0
    if count_neg == 0:
        return min_pos
    if count_neg & 1 ==0 and count_neg != 0:
        product = product // max_neg
    return product

print(min_subProduct(A))
print(min_subProduct(B))

Q.51.Given an array of n elements. Find maximum sum of pairwise multiplications. Sum can be larger so take mod with 10^9+7. If there are odd elements, then we can add any one element (without forming a pair) to the sum.
A = [-1, 4, 5, -7, -4, 9, 0]
B = [8, 7, 9]
from sys import maxsize
def maxSum_pairProduct(arr):
    sum = 0
    arr.sort()
    i = 0
    while i < len(arr) and arr[i] < 0:
        if i != len(arr)-1 and arr[i+1]<=0:
            sum = sum + (arr[i]*arr[i+1])
            i+=2
        else:
            break
    j = len(arr)-1
    while j >=0 and arr[j] > 0:
        if j != 0 and arr[j-1] > 0:
            sum += arr[j]*arr[j-1]
            j-=2
        else:
            break
    if j > i:
        sum = sum + arr[i]*arr[j]
    elif i ==j:
        sum = sum + arr[i]
    return sum



print(maxSum_pairProduct(A))
print(maxSum_pairProduct(B))


