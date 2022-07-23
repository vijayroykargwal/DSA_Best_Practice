Array Rotation Problems:
Q1.Given an array of integers arr[] of size N and an integer, the task is to rotate the array elements to the left by d positions.
A = [1, 2, 3, 4, 5, 6, 7]
B = [3, 4, 5, 6, 7, 1, 2]
from sys import maxsize
def array_rotate(arr,d):
    return arr[d:]+arr[:d]

print(array_rotate(A,2))
print(array_rotate(B,2))

Q2.Given an array, cyclically rotate the array clockwise by one.
A = [1, 2, 3, 4, 5]
B = [3, 4, 5, 6, 7, 1, 2]

def cyclic_rotate(arr):
    return arr[-1:]+arr[:-1]

print(cyclic_rotate(A))
print(cyclic_rotate(B))

Array Rearrangement Problems:
Q1.Given an array of random numbers, Push all the zero’s of a given array to the end of the array. For example, if the given arrays is {1, 9, 8, 4, 0, 0, 2, 7, 0, 6, 0}, it should be changed to {1, 9, 8, 4, 2, 7, 6, 0, 0, 0, 0}. The order of all other elements should be same. Expected time complexity is O(n) and extra space is O(1).
A = [1, 2, 0, 4, 3, 0, 5, 0]
B = [1, 2, 0, 0, 0, 3, 6]

def move_zeroes_end(arr):
    j=0
    for i in range(len(arr)):
        if arr[i] != 0:
            arr[j],arr[i]=arr[i],arr[j]
            j+=1
    return arr

print(move_zeroes_end(A))
print(move_zeroes_end(B))

Q2.Given an array of n positive integers and a number k. Find the minimum number of swaps required to bring all the numbers less than or equal to k together. 
A = [2, 1, 5, 6, 3]
B = [2, 7, 9, 5, 8, 7, 4]

def min_swap(arr,k):
    count = 0
    for i in range(len(arr)):
        if arr[i] <= k:
            count += 1
    bad =0
    for i in range(0,count):
        if arr[i] > k:
            bad += 1
    ans = bad
    j = count
    for i in range(0,len(arr)):
        if j == len(arr):
            break
        if arr[i] > k:
            bad -= 1
        if arr[j] > k:
            bad += 1
        ans = min(ans,bad)
        j += 1
    return ans
print(min_swap(A,3))
print(min_swap(B,5))

Q3.Given an array of numbers, arrange them in a way that yields the largest value. For example, if the given numbers are {54, 546, 548, 60}, the arrangement 6054854654 gives the largest value. And if the given numbers are {1, 34, 3, 98, 9, 76, 45, 4}, then the arrangement 998764543431 gives the largest value.
Naive Approach:
A = [54, 546, 548, 60]
B = [1, 34, 3, 98, 9, 76, 45, 4]

def largest_number(arr):
    if len(arr) == 1:
        return str(arr[0])
    for i in range(len(arr)):
        arr[i] = str(arr[i])
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if arr[j]+arr[i]>arr[i]+arr[j]:
                arr[i],arr[j]=arr[j],arr[i]
    res = ''.join(arr)
    if res == '0'*len(res):
        return '0'
    else:
        return res

print(largest_number(A))
print(largest_number(B))

Another Approach:

A = [54, 546, 548, 60]
B = [1, 34, 3, 98, 9, 76, 45, 4]
from itertools import permutations
def largest_number(arr):
    lst = []
    for i in permutations(arr,len(arr)):
        lst.append(''.join(map(str,i)))
    return max(lst)

print(largest_number(A))
print(largest_number(B))

Q.4.Given a sorted array of positive integers, rearrange the array alternately i.e first element should be the maximum value, second minimum value, third-second max, fourth-second min and so on.
A = [1, 2, 3, 4, 5, 6, 7]
B = [1, 2, 3, 4, 5, 6]

def rearrange_max_min(arr):
    max_ele = arr[len(arr)-1]
    min_ele = arr[0]
    for i in range(len(arr)):
        if i %2 == 0:
            arr[i] = max_ele
            max_ele -= 1
        else:
            arr[i] = min_ele
            min_ele += 1
    return arr

print(rearrange_max_min(A))
print(rearrange_max_min(B))

Q.5.Given an array, write a program to generate a random permutation of array elements. This question is also asked as “shuffle a deck of cards” or “randomize a given array”. Here shuffle means that every permutation of array element should be equally likely.
A = [1, 2, 3, 4, 5, 6, 7]
B = [1, 2, 3, 4, 5, 6]
from random import randint
def shuffle_arr(arr):
    for i in range(len(arr)-1,0,-1):
        j = randint(0,i+1)
        arr[i],arr[j]=arr[j],arr[i]
    return arr

print(shuffle_arr(A))
print(shuffle_arr(B))

Q.6.Given an array arr[] of integers, segregate even and odd numbers in the array. Such that all the even numbers should be present first, and then the odd numbers.
A = [1, 9, 5, 3, 2, 6, 7, 11]
B = [1, 3, 2, 4, 7, 6, 9, 10]

def seg_even_odd(arr):
    i = -1
    j = 0
    while j != len(arr):
        if arr[j]%2 == 0:
            i += 1
            arr[i],arr[j]=arr[j],arr[i]
        j += 1
    return  arr

print(seg_even_odd(A))
print(seg_even_odd(B))

Q.7.You are given an array of 0s and 1s in random order. Segregate 0s on left side and 1s on right side of the array [Basically you have to sort the array]. Traverse array only once.
A = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
B = [0, 1, 0, 1, 1, 1]

def seg_0s_1s(arr):
    i = 0
    j = len(arr)-1
    while i < j:
        if arr[i] == 1:
            if arr[j] != 1:
                arr[i],arr[j]=arr[j],arr[i]
            j -= 1
        else:
            i += 1
    return  arr

print(seg_0s_1s(A))
print(seg_0s_1s(B))

Q.8.Given an array arr[0 … n-1] containing n positive integers, a subsequence of arr[] is called Bitonic if it is first increasing, then decreasing. Write a function that takes an array as argument and returns the length of the longest bitonic subsequence. 
A sequence, sorted in increasing order is considered Bitonic with the decreasing part as empty. Similarly, decreasing order sequence is considered Bitonic with the increasing part as empty. 
A = [1, 11, 2, 10, 4, 5, 2, 1]
B = [80, 60, 30, 40, 20, 10]

def longest_bitonic_subseq(arr):
    lis = [1]*(len(arr)+1)
    for i in range(1,len(arr)):
        for j in range(0,i):
            if arr[i]>arr[j] and lis[i]<lis[j]+1:
                lis[i] = lis[j]+1
    lds = [1]*(len(arr)+1)
    for i in reversed(range(len(arr)-1)):
        for j in reversed(range(i-1,len(arr))):
            if arr[i]>arr[j] and lds[i]<lds[j]+1:
                lds[i] = lds[j]+1
    maximum = lis[0]+lds[0]-1
    for i in range(1,len(arr)):
        maximum = max((lis[i]+lds[i]-1),maximum)
    return maximum

print(longest_bitonic_subseq(A))
print(longest_bitonic_subseq(B))

Q.9.Given an array that contains both positive and negative integers, find the product of the maximum product subarray. Expected Time complexity is O(n) and only O(1) extra space can be used.
A = [6, -3, -10, 0, 2]
B = [-1, -3, -10, 0, 60]

def max_subarray_product(arr):
    max_ending_here =1
    min_ending_here =1
    max_so_far = 0
    flag = 0
    for i in range(len(arr)):
        if arr[i]>0:
            max_ending_here = max_ending_here*arr[i]
            min_ending_here = min(min_ending_here*arr[i],1)
            flag = 1
        elif arr[i]==0:
            max_ending_here =1
            min_ending_here=1
        else:
            temp = max_ending_here
            max_ending_here = max(min_ending_here*arr[i],1)
            min_ending_here = temp*arr[i]
        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
    if flag ==0 and max_so_far == 0:
        return  0
    return max_so_far

print(max_subarray_product(A))
print(max_subarray_product(B))

Q.10.Given n numbers (both +ve and -ve), arranged in a circle, find the maximum sum of consecutive numbers.
A = [8, -8, 9, -9, 10, -11, 12]
B = [-1, 40, -14, 7, 6, 5, -4, -1]

def maxsum_circular_subarray(arr):
    if len(arr)==1:
        return arr[0]
    arr_sum = 0
    for i in range(len(arr)):
        arr_sum += arr[i]
    curr_max = arr[0]
    max_so_far = arr[0]
    curr_min = arr[0]
    min_so_far = arr[0]
    for i in range(1,len(arr)):
        curr_max = max(curr_max+arr[i],arr[i])
        max_so_far=max(max_so_far,curr_max)
        curr_min = min(curr_min+arr[i],arr[i])
        min_so_far = min(curr_min, min_so_far)
    if min_so_far == arr_sum:
        return max_so_far
    return max(max_so_far, arr_sum-min_so_far)


print(maxsum_circular_subarray(A))
print(maxsum_circular_subarray(B))

Q.11.Given an array of 0s and 1s, find the position of 0 to be replaced with 1 to get longest continuous sequence of 1s. Expected time complexity is O(n) and auxiliary space is O(1). 
A = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
B = [1, 1, 1, 1, 0]

def longest_cont_1s_seq(arr):
    max_count = 0
    max_index = 0
    prev_zero = -1
    prev_prev_zero = -1
    for curr in range(len(arr)):
        if arr[curr]==0:
            if curr - prev_prev_zero > max_count:
                max_count = curr -prev_prev_zero
                max_index = prev_zero
            prev_prev_zero = prev_zero
            prev_zero = curr
        if len(arr)-prev_prev_zero > max_count:
            max_index = prev_zero
    return max_index
print(longest_cont_1s_seq(A))
print(longest_cont_1s_seq(B))

Q.12.There are n-pairs and therefore 2n people. everyone has one unique number ranging from 1 to 2n. All these 2n persons are arranged in random fashion in an Array of size 2n. We are also given who is partner of whom. Find the minimum number of swaps required to arrange these pairs such that all pairs become adjacent to each other
A = [0, 3, 5, 6, 4, 1, 2]
B = [0, 3, 6, 1, 5, 4, 2]

def update_index(idx,a,ai,b,bi):
    idx[a] = ai
    idx[b] = bi
def minSwap_Util(arr,pairs,idx,i,n):
    if i > n:
        return 0
    if pairs[arr[i]] == arr[i+1]:
        return minSwap_Util(arr,pairs,idx,i+2,n)
    one = arr[i+1]
    idxtwo = i+1
    idxone = idx[pairs[arr[i]]]
    two = arr[idx[pairs[arr[i]]]]
    arr[i+1],arr[idxone]=arr[idxone],arr[i+1]
    update_index(idx,one,idxone,two,idxtwo)
    a = minSwap_Util(arr,pairs,idx,i+2,n)
    arr[i+1],arr[idxone]=arr[idxone],arr[i+1]
    update_index(idx,one,idxtwo,two,idxone)
    one = arr[i]
    idxone = idx[pairs[arr[i+1]]]
    two = arr[idx[pairs[arr[i+1]]]]
    idxtwo=i
    arr[i],arr[idxone]=arr[idxone],arr[i]
    update_index(idx,one,idxone,two,idxtwo)
    b=minSwap_Util(arr,pairs,idx,i+2,n)
    arr[i],arr[idxone]=arr[idxone],arr[i]
    update_index(idx,one,idxtwo,two,idxone)
    return 1+min(a,b)

def min_swaps(n,pairs,arr):
    idx =[]
    for i in range(2*n+1+1):
        idx.append(0)
    for i in range(1,2*n+1):
        idx[arr[i]]=i
    return minSwap_Util(arr,pairs,idx,1,2*n)
m=len(B)
n =m//2
print(min_swaps(n,B,A))

Q.13.Given an array, find whether it is possible to obtain an array having distinct neighbouring elements by swapping two neighbouring array elements.
A = [1, 1, 2]
B = [7, 7, 7, 7]

def distinct_adj_ele(arr):
    map = {}
    for i in arr:
        if i in map:
            map[i] += 1
        else:
            map[i] = 1
    mx = 0
    for i in range(len(arr)):
        if mx < map[arr[i]]:
            mx = map[arr[i]]
    if mx > (len(arr)+1)//2:
        return 'NO'
    else:
        return 'YES'

print(distinct_adj_ele(A))
print(distinct_adj_ele(B))

Q.14.Given k sorted arrays of possibly different sizes, merge them and print the sorted output.
A = [[1,3],[2,4,6],[0,9,10,11]]
B = [[1,3,20],[2,4,6]]
from heapq import merge
def merge_ksorted_arr(arr,k):
    l = arr[0]
    for i in range(k-1):
        l = list(merge(l,arr[i+1]))
    return l

print(merge_ksorted_arr(A,3))
print(merge_ksorted_arr(B,2))

Q.15.There are 2 sorted arrays A and B of size n each. Write an algorithm to find the median of the array obtained after merging the above 2 arrays(i.e. array of length 2n). The complexity should be O(log(n)). 
A = [1,12,15,26,38]
B = [2,13,17,30,45]
from heapq import merge
def merge_sorted_arr(arr1,arr2):
    l = list(merge(arr1,arr2))
    n = len(l)
    if n%2==0:
        return (l[n//2]+l[n//2-1])//2
    else:
        return l[n//2]

print(merge_sorted_arr(A,B))

Q.16.Given two sorted arrays, a[] and b[], the task is to find the median of these sorted arrays, in O(log n + log m) time complexity, when n is the number of elements in the first array, and m is the number of elements in the second array. 
This is an extension of median of two sorted arrays of equal size problem. Here we handle arrays of unequal size also.
A = [-5, 3, 6, 12, 15]
B = [-12, -10, -6, -3, 4, 10]
from heapq import merge
def merge_sorted_arr(arr1,arr2):
    l = list(merge(arr1,arr2))
    n = len(l)
    if n%2==0:
        return (l[n//2]+l[n//2-1])//2
    else:
        return l[n//2]

print(merge_sorted_arr(A,B))
