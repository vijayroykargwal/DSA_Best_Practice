Range Queries Problems in Arrays:
Q1.Given a set of time intervals in any order, merge all overlapping intervals into one and output the result which should have only mutually exclusive intervals.
A = [[1,3],[2,4],[6,8],[9,10]]
B = [[6,8],[1,9],[2,4],[4,7]]
def merge_intervals(arr):
    arr.sort(key=lambda x:x[0])
    idx = 0
    for i in range(1,len(arr)):
        if arr[idx][1] >= arr[i][0]:
            arr[idx][1] = max(arr[idx][1], arr[i][1])
        else:
            idx += 1
            arr[idx] = arr[i]
    for i in range(idx+1):
        print(arr[i], end = ' ')

print(merge_intervals(A))
print(merge_intervals(B))

Order Statistics Problems in Arrays:
 Q1.Given an n x n matrix, where every row and column is sorted in non-decreasing order. Find the kth smallest element in the given 2D array.
 normal Approach :O(n^2)
 A = [[10, 20, 30, 40],[15, 25, 35, 45],[24, 29, 37, 48],[32, 33, 39, 50]]
B = [[10, 20, 30, 40],[15, 25, 35, 45],[24, 29, 37, 48],[32, 33, 39, 50]]
def k_smallest_2D(arr,k):
    temp = [0]*(len(arr)*len(arr))
    v = 0
    for i in range(len(arr)):
        for j in range(len(arr)):
            temp[v] = arr[i][j]
            v += 1
    temp.sort()
    return temp[k-1]

print(k_smallest_2D(A,3))
print(k_smallest_2D(B,7))

Q.2.Given an array of n distinct elements, the task is to find all elements in array which have at-least two greater elements than themselves.
using sorting:
A = [2, 8, 7, 1, 5]
B = [7, -2, 3, 4, 9, -1]
def atLeast2_Greater(arr):
    arr.sort()
    for i in range(len(arr)-2):
        print(arr[i], end=' ')

print(atLeast2_Greater(A))
print(atLeast2_Greater(B))

Efficient:
A = [2, 8, 7, 1, 5]
B = [7, -2, 3, 4, 9, -1]
from sys import maxsize
def atLeast2_Greater(arr):
    first_max = -maxsize-1
    second_max = -maxsize-1
    for i in range(len(arr)):
        if arr[i] > first_max:
            second_max = first_max
            first_max = arr[i]
        elif arr[i] > second_max:
            second_max = arr[i]
    for i in range(len(arr)):
        if arr[i] < second_max:
            print(arr[i], end = ' ')

print(atLeast2_Greater(A))
print(atLeast2_Greater(B))

Q.3.Given an array of integers. Write a program to find the K-th largest sum of contiguous subarray within the array of numbers which has negative and positive numbers.
A = [20, -5, -1]
B = [10, -10, 20, -40]
import heapq
def largest_contSub_Sum(arr,k):
    sum = []
    sum.append(0)
    sum.append(arr[0])
    for i in range(2,len(arr)+1):
        sum.append(sum[i-1]+arr[i-1])
    queue = []
    heapq.heapify(queue)
    for i in range(1,len(arr)+1):
        for j in range(i,len(arr)+1):
            x = sum[j] - sum[i-1]
            if len(queue) < k:
                heapq.heappush(queue,x)
            else:
                if queue[0] < x:
                    heapq.heappop(queue)
                    heapq.heappush(queue,x)
    return queue[0]

print(largest_contSub_Sum(A,3))
print(largest_contSub_Sum(B,6))

Q.4.Given two equally sized arrays (A, B) and N (size of both arrays). 
A sum combination is made by adding one element from array A and another element of array B. Display the maximum K valid sum combinations from all the possible sum combinations. 
A = [4, 2, 5, 1]
B = [8, 0, 3, 5]
from heapq import *
def max_sum_comb(arr1,arr2,k):
    arr1.sort(reverse=True)
    arr2.sort(reverse=True)
    visited = set()
    ans =[]
    heap = []
    heappush(heap, (-(arr1[0]+arr2[0]),(0,0)))
    visited.add((0,0))
    for i in range(len(arr1)):
        sum,(i_arr1,j_arr2) = heappop(heap)
        ans.append(-sum)
        tuple1 = (i_arr1+1,j_arr2)
        if i_arr1 < len(arr1)-1 and tuple1 not in visited:
            heappush(heap, (-(arr1[i_arr1 + 1] + arr2[j_arr2]), tuple1))
            visited.add(tuple1)
        tuple2 = (i_arr1,j_arr2+1)
        if j_arr2 < len(arr2)-1 and tuple2 not in visited:
            heappush(heap,(-(arr1[i_arr1] + arr2[j_arr2+1]), tuple2))
            visited.add(tuple2)
    return ans[:k]

print(max_sum_comb(A,B,3))

Q.5.Given an array of Integers and an Integer value k, find out k sub-arrays(may be overlapping), which have k maximum sums
A = [4, -8, 9, -4, 1, -8, -1, 6]
B = [-2, -3, 4, -1, -2, 1, 5, -3]
from sys import maxsize
def prefix_sum(arr):
    pre_sum = []
    pre_sum.append(arr[0])
    for i in range(1, len(arr)):
        pre_sum.append(pre_sum[i-1] + arr[i])
    return pre_sum
def maxMerge(maxi,cand):
    j=0
    for i in range(len(maxi)):
        if cand[j] > maxi[i]:
            maxi.insert(i,cand[j])
            del maxi[-1]
            j+=1
def insert_Mini(mini,pre_sum):
    for i in range(len(mini)):
        if pre_sum < mini[i]:
            mini.insert(i,pre_sum)
            del mini[-1]
            break

def max_sum_overlapCont_Sub(arr,k):
    pre_sum = prefix_sum(arr)
    mini = [maxsize]*k
    mini[0]=0
    maxi=[-maxsize-1]*k
    cand = [0]*k
    for i in range(len(arr)):
        for j in range(k):
            cand[j]=pre_sum[i]-mini[j]
        maxMerge(maxi,cand)
        insert_Mini(mini,pre_sum[i])
    for ele in maxi:
        print(ele, end=' ')


print(max_sum_overlapCont_Sub(A,4))
print(max_sum_overlapCont_Sub(B,3))

Q.6.Given an Array of Integers and an Integer value k, find out k non-overlapping sub-arrays which have k maximum sums.
A = [4, 1, 1, -1, -3, -5, 6, 2, -6, -2]
B = [5, 1, 2, -6, 2, -1, 3, 1]
from sys import maxsize
def max_sum_nonoverlapCont_Sub(arr,k):
    for j in range(k):
        max_so_far = -maxsize-1
        curr_sum = 0
        start,end,s = 0,0,0
        for i in range(len(arr)):
            curr_sum += arr[i]
            if max_so_far < curr_sum:
                max_so_far = curr_sum
                start = s
                end = i
            if curr_sum < 0:
                curr_sum = 0
                s = i+1
        print(max_so_far, end=' ')
        for l in range(start, end+1):
            arr[l]=-maxsize-1

print(max_sum_nonoverlapCont_Sub(A,3))
print(max_sum_nonoverlapCont_Sub(B,2))

Q.7.Given two integer arrays arr1[] and arr2[] sorted in ascending order and an integer k. Find k pairs with smallest sums such that one element of a pair belongs to arr1[] and other element belongs to arr2[]
A = [1, 7, 11]
B = [2, 4, 6]
from heapq import *
def min_sum_comb(arr1,arr2,k):
    arr1.sort()
    arr2.sort()
    visited = set()
    ans =[]
    heap = []
    heappush(heap, ((arr1[0]+arr2[0]),(0,0)))
    visited.add((0,0))
    for i in range(len(arr1)):
        sum,(i_arr1,j_arr2) = heappop(heap)
        ans.append([arr1[i_arr1],arr2[j_arr2]])
        tuple1 = (i_arr1+1,j_arr2)
        if i_arr1 < len(arr1)-1 and tuple1 not in visited:
            heappush(heap, ((arr1[i_arr1 + 1] + arr2[j_arr2]), tuple1))
            visited.add(tuple1)
        tuple2 = (i_arr1,j_arr2+1)
        if j_arr2 < len(arr2)-1 and tuple2 not in visited:
            heappush(heap,((arr1[i_arr1] + arr2[j_arr2+1]), tuple2))
            visited.add(tuple2)
    return ans[:k]

print(min_sum_comb(A,B,3))

Q.8.We are given an array of size n containing positive integers. The absolute difference between values at indices i and j is |a[i] – a[j]|. There are n*(n-1)/2 such pairs and we are asked to print the kth (1 <= k <= n*(n-1)/2) as the smallest absolute difference among all these pairs.
A = [1, 2, 3, 4]
B = [10, 10]
from bisect import bisect as upper_bound
def count_pairs(arr,mid):
    ans = 0
    for i in range(len(arr)):
        ans += upper_bound(arr, arr[i]+mid)
    return ans
def kthSmallest_absDiff(arr,k):
    arr.sort()
    low = arr[1]-arr[0]
    high = arr[len(arr)-1]-arr[0]
    while low < high:
        mid = (low+high)//2
        if count_pairs(arr,mid) < k :
            low = mid + 1
        else:
            high = mid
    return low

print(kthSmallest_absDiff(A,3))
print(kthSmallest_absDiff(B,1))

Q.9.Given an array of n numbers and a positive integer k. The problem is to find k numbers with most occurrences, i.e., the top k numbers having the maximum frequency. If two numbers have the same frequency then the larger number should be given preference. The numbers should be displayed in decreasing order of their frequencies. It is assumed that the array consists of k numbers with most occurrences.
A = [3, 1, 4, 4, 5, 2, 6, 1]
B = [7, 10, 11, 5, 2, 5, 5, 7, 11, 8, 9]

def k_most_occur(arr,k):
    map={}
    for i in arr:
        if i in map:
            map[i]+=1
        else:
            map[i]=1
    j=0
    ans = [0] * len(map)
    for i in map:
        ans[j]=[i, map[i]]
        j += 1
    ans = sorted(ans,key=lambda x: x[0], reverse=True)
    ans = sorted(ans,key=lambda x: x[1], reverse=True)
    for i in range(k):
        print(ans[i][0], end=' ')


print(k_most_occur(A,2))
print(k_most_occur(B,4))

Q.10.Given a sorted array of n distinct integers where each integer is in the range from 0 to m-1 and m > n. Find the smallest number that is missing from the array.
A = [4, 5, 10, 11]
B = [0, 1, 2, 3, 4, 5, 6, 7, 10]

def smallest_missing(arr,start,end):
    if start > end:
        return end+1
    if start != arr[start]:
        return start
    mid = (start+end)//2
    if arr[mid]==mid:
        return smallest_missing(arr,mid+1,end)
    return smallest_missing(arr,start,mid)
print(smallest_missing(A,0,len(A)-1))
print(smallest_missing(B,0,len(B)-1))

Q.11.Given an array arr[] of positive numbers, the task is to find the maximum sum of a subsequence with the constraint that no 2 numbers in the sequence should be adjacent in the array.
A = [5, 5, 10, 100, 10, 5]
B = [3, 2, 5, 10, 7]

def maxSum_noadj_subseq(arr):
    include = 0
    exclude = 0
    for i in arr:
        new_exclude = max(exclude,include)
        include = exclude + i
        exclude = new_exclude
    return max(include,exclude)
print(maxSum_noadj_subseq(A))
print(maxSum_noadj_subseq(B))

Q.12.Given an array arr[] of integers, find out the maximum difference between any two elements such that larger element appears after the smaller number.
A = [2, 3, 10, 6, 4, 8, 1]
B = [7, 9, 5, 6, 3, 2]

def max_Diff(arr):
    max_diff = arr[1]-arr[0]
    min_ele = arr[0]
    for i in range(1,len(arr)):
        if arr[i]-min_ele > max_diff:
            max_diff = arr[i]-min_ele
        if arr[i] < min_ele:
            min_ele = arr[i]
    return max_diff
print(max_Diff(A))
print(max_Diff(B))

Q.13.Given an array and an integer K, find the maximum for each and every contiguous subarray of size k
A = [1, 2, 3, 1, 4, 5, 2, 3, 6]
B = [8, 5, 10, 7, 9, 4, 15, 12, 90, 13]
from collections import deque
def max_cont_subarray(arr,k):
    queue = deque()
    for i in range(k):
        while queue and arr[i] >= arr[queue[-1]]:
            queue.pop()
        queue.append(i)
    for i in range(k,len(arr)):
        print(str(arr[queue[0]]), end=' ')
        while queue and queue[0] <= i-k:
            queue.popleft()
        while queue and arr[i] >= arr[queue[-1]]:
            queue.pop()
        queue.append(i)
    print(str(arr[queue[0]]))
print(max_cont_subarray(A,3))
print(max_cont_subarray(B,4))

Q.14.Given an array of random numbers. Find longest increasing subsequence (LIS) in the array. I know many of you might have read recursive and dynamic programming (DP) solutions. There are few requests for O(N log N) algo in the forum posts.
A = [2, 5, 3, 7, 11, 8, 10, 13, 6]
B = [2, 5, 3, 5, 4, 4, 2, 3]
from bisect import bisect_left
def longest_inc_subsequences(arr):
    if len(arr)==0:
        return 0
    tail = [0]*(len(arr)+1)
    length = 1
    tail[0] = arr[0]
    for i in range(1,len(arr)):
        if arr[i]>tail[length-1]:
            tail[length]=arr[i]
            length += 1
        else:
            tail[bisect_left(tail,arr[i],0,length-1)] = arr[i]
    return length
print(longest_inc_subsequences(A))
print(longest_inc_subsequences(B))

Q,15.You are given an unsorted array with both positive and negative elements. You have to find the smallest positive number missing from the array in O(n) time using constant extra space. You can modify the original array.
A = [2, 3, 7, 6, 8, -1, -10, 15]
B = [2, 3, -7, 6, 8, 1, -10, 15]

def missing_number(arr):
    for i in range(len(arr)):
        while arr[i] >= 1 and arr[i] <= len(arr) and arr[i] != arr[arr[i]-1]:
            temp = arr[i]
            arr[i] = arr[arr[i]-1]
            arr[temp-1] = temp
    for i in range(len(arr)):
        if arr[i] != i+1:
            return i+1
    return len(arr)+1


print(missing_number(A))
print(missing_number(B))

Q.16.Given an array of size n, the array contains numbers in the range from 0 to k-1 where k is a positive integer and k <= n. Find the maximum repeating number in this array. For example, let k be 10 the given array be arr[] = {1, 2, 2, 2, 0, 2, 0, 2, 3, 8, 0, 9, 2, 3}, the maximum repeating number would be 2. The expected time complexity is O(n) and extra space allowed is O(1). Modifications to array are allowed. 
A = [2, 3, 3, 5, 3, 4, 1, 7]
B = [1, 2, 2, 2, 0, 2, 0, 2, 3, 8, 0, 9, 2, 3]

def max_repeating(arr,k):
    for i in range(len(arr)):
        arr[arr[i]%k] += k
    max = arr[0]
    for i in range(1,len(arr)):
        if arr[i] > max:
            max = arr[i]
            ans = i
    return ans

print(max_repeating(A,8))
print(max_repeating(B,10))

Q.17.Given two sorted arrays, such that the arrays may have some common elements. Find the sum of the maximum sum path to reach from the beginning of any array to end of any of the two arrays. We can switch from one array to another array only at common elements. 
Note: The common elements do not have to be at the same indexes.
Expected Time Complexity: O(m+n), where m is the number of elements in ar1[] and n is the number of elements in ar2[].
A = [2, 3, 7, 10, 12]
B = [1, 5, 7, 8]

def max_sum_path(arr1,arr2):
    i,j = 0,0
    res,sum1,sum2=0,0,0
    while i < len(arr1) and j < len(arr2):
        if arr1[i]<arr2[j]:
            sum1 += arr1[i]
            i+=1
        elif arr1[i]>arr2[j]:
            sum2 += arr2[j]
            j +=1
        else:
            res += max(sum1,sum2)+arr1[i]
            sum1 = 0
            sum2 = 0
            i+=1
            j+=1
    while i < len(arr1):
        sum1 += arr1[i]
        i+=1
    while j < len(arr2):
        sum2 += arr2[j]
        j+=1
    res += max(sum1,sum2)
    return res

print(max_sum_path(A,B))
A = [2, 3, 7, 10, 12, 15, 30, 34]
B = [1, 5, 7, 8, 10, 15, 16, 19]
print(max_sum_path(A,B))

Q.18.Given two sorted arrays and a number x, find the pair whose sum is closest to x and the pair has an element from each array. 
We are given two arrays ar1[0…m-1] and ar2[0..n-1] and a number x, we need to find the pair ar1[i] + ar2[j] such that absolute value of (ar1[i] + ar2[j] – x) is minimum.
A = [1, 4, 5, 7]
B = [10, 20, 30, 40]
from sys import maxsize
def closest_pair(arr1,arr2,x):
    diff = maxsize
    l = 0
    r = len(arr2)-1
    while l < len(arr1) and r >= 0:
        if abs(arr1[l]+arr2[r]-x) < diff:
            res_l = l
            res_r = r
            diff = abs(arr1[l]+arr2[r]-x)
        if arr1[l]+arr2[r] > x:
            r -= 1
        else:
            l += 1
    return([arr1[res_l],arr2[res_r]])

print(closest_pair(A,B,32))
A = [1, 4, 5, 7]
B = [10, 20, 30, 40]
print(closest_pair(A,B,50))

Q.19.Given two arrays of integers, compute the pair of values (one value in each array) with the smallest (non-negative) difference. Return the difference.
A = [1, 3, 15, 11, 2]
B = [23, 127, 235, 19, 8]
from sys import maxsize
def smallest_DiffPair_Val(arr1,arr2,x):
    arr1.sort()
    arr2.sort()
    i,j = 0,0
    res = maxsize
    while i < len(arr1) and j < len(arr2):
        if abs(arr1[i]-arr2[j]) < res:
            res = abs(arr1[i]-arr2[j])
        if arr1[i] < arr2[j]:
            i+=1
        else:
            j +=1
    return res

print(smallest_DiffPair_Val(A,B,32))
A = [10, 5, 40]
B = [50, 90, 80]
print(smallest_DiffPair_Val(A,B,50))

Q.20.Given an unsorted of distinct integers, find the largest pair sum in it. For example, the largest pair sum in {12, 34, 10, 6, 40} is 74.
A = [12, 34, 10, 6, 40]
B = [23, 127, 235, 19, 8]
from sys import maxsize
def largest_sum_pair(arr):
    first = -maxsize-1
    second = -maxsize-1
    for i in range(len(arr)):
        if arr[i] > first:
            second = first
            first = arr[i]
        elif arr[i] > second:
            second = arr[i]
    return first + second

print(largest_sum_pair(A))
print(largest_sum_pair(B))

Q.21.Given a binary array and an integer m, find the position of zeroes flipping which creates maximum number of consecutive 1’s in array.
A = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
B = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
from sys import maxsize
def flip_zeros(arr,m):
    wl = wr = 0
    bestl=bestw=0
    zero_count = 0
    while wr < len(arr):
        if zero_count <= m:
            if arr[wr]==0:
                zero_count += 1
            wr += 1
        if zero_count > m:
            if arr[wl] == 0:
                zero_count -= 1
            wl += 1
        if wr-wl > bestw and zero_count <= m:
            bestw = wr - wl
            bestl = wl
    for i in range(0,bestw):
        if arr[bestl+i] == 0:
            print(bestl+i, end=' ')

print(flip_zeros(A,2))
print(flip_zeros(B,1))

Q.22.Given an array of integers, count number of subarrays (of size more than one) that are strictly increasing.
A = [1, 2, 3, 4]
B = [1, 2, 2, 4]
from sys import maxsize
def count_inc_subarray(arr):
    count = 0
    length =1
    for i in range(0,len(arr)-1):
        if arr[i+1] > arr[i]:
            length += 1
        else:
            count += length*(length-1)//2
            length = 1
    if length > 1:
        count += length*(length-1)//2
    return count
print(count_inc_subarray(A))
print(count_inc_subarray(B))

Q.23.You are given an array of n elements. You have to divide the given array into two group such that one group consists exactly k elements and second group consists rest of elements. Your result must be maximum possible difference of sum of elements of these two group.
A = [1, 5, 2, 6, 3]
B = [1, -1, 3, -2, -3]
from sys import maxsize
def array_sum(arr,n):
    sum = 0
    for i in range(n):
        sum += arr[i]
    return sum
def max_diff_btwk(arr,k):
    arr.sort()
    arr_sum = array_sum(arr,len(arr))
    diff1 = abs(arr_sum - 2*array_sum(arr,k))
    arr.reverse()
    diff2 = abs(arr_sum - 2*array_sum(arr,k))
    return  max(diff1,diff2)

print(max_diff_btwk(A,2))
print(max_diff_btwk(B,2))

