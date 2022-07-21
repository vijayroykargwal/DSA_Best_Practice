Q.3.Given two arrays: arr1[0..m-1] and arr2[0..n-1]. Find whether arr2[] is a subset of arr1[] or not. Both the arrays are not in sorted order. It may be assumed that elements in both array are distinct.
A = [11, 1, 13, 21, 3, 7]
B = [11, 3, 7, 1]
from sys import maxsize
def is_subset(arr1,arr2):
    map = set()
    for i in arr1:
        map.add(i)
    for ele in arr2:
        if ele not in map:
            return False
    return True

print(is_subset(A,B))
A=[10, 5, 2, 23, 19]
B=[19, 5, 3]
print(is_subset(A,B))

Q.4.Given two sorted arrays arr1 and arr2 of size m and n respectively. We need to find relative complement of two array i.e, arr1 – arr2 which means that we need to find all those elements which are present in arr1 but not in arr2.
A = [3, 6, 10, 12, 15]
B = [1, 3, 5, 10, 16]
from sys import maxsize
def relative_Compliment(arr1,arr2):
    i = 0
    j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            print(arr1[i])
            i+=1
        elif arr1[i] > arr2[j]:
            j += 1
        elif arr1[i] == arr2[j]:
            i +=1
            j +=1
    while i < len(arr1):
        print(arr1[i])


print(relative_Compliment(A,B))
A=[10, 20, 36, 59]
B=[5, 10, 15, 59]
print(relative_Compliment(A,B))

Q.5.You are given an array of n-elements, you have to find the number of operations needed to make all elements of array equal. Where a single operation can increment an element by k. If it is not possible to make all elements equal print -1.
A = [4, 7, 19, 16]
B = [4, 2, 6, 8]
from sys import maxsize
def make_equal(arr,k):
    max_arr = max(arr)
    ans =0
    for i in range(len(arr)):
        if (max_arr-arr[i])%k != 0:
            return -1
        else:
            ans += (max_arr - arr[i])//k
    return ans

print(make_equal(A,3))
print(make_equal(B,3))

Q.6.Given three sorted arrays A, B, and C of not necessarily same sizes. Calculate the minimum absolute difference between the maximum and minimum number of any triplet A[i], B[j], C[k] such that they belong to arrays A, B and C respectively, i.e., minimize (max(A[i], B[j], C[k]) – min(A[i], B[j], C[k]))
A = [ 1, 4, 5, 8, 10]
B = [6, 9, 15]
C = [2, 3, 6, 6]
from sys import maxsize
def min_diff(arr1,arr2,arr3):
    i = len(arr1)-1
    j = len(arr2)-1
    k = len(arr3)-1
    min_diff = abs(max(arr1[i],arr2[j],arr3[k]) - min(arr1[i],arr2[j],arr3[k]))
    while i != -1 and j != -1 and k != -1:
        curr_diff = abs(max(arr1[i],arr2[j],arr3[k]) - min(arr1[i],arr2[j],arr3[k]))
        if curr_diff < min_diff:
            min_diff = curr_diff
        max_term = max(arr1[i],arr2[j],arr3[k])
        if arr1[i]==max_term:
            i -= 1
        elif arr2[j]==max_term:
            j -= 1
        else:
            k-=1
    return min_diff


print(min_diff(A,B,C))

Array Sorting Problems:

Q.1.There are two sorted arrays. First one is of size m+n containing only m elements. Another one is of size n and contains n elements. Merge these two arrays into the first array of size m+n such that the output is sorted. 
Input: array with m+n elements (mPlusN[]).
NA = -1
A = [ 2, 8, NA, NA, NA, 13, NA, 15, 20]
B = [5, 7, 9, 25]
from sys import maxsize
def moveTo_End(arr):
    i = 0
    j = len(arr)-1
    for i in range(len(arr)-1,-1,-1):
        if arr[i] != NA:
            arr[j] = arr[i]
            j-=1
def merge(arr1,arr2):
    moveTo_End(arr1)
    i = len(arr2)
    j = 0
    k = 0
    while k < len(arr1):
        if j == len(arr2) or (i < len(arr1) and arr1[i] <= arr2[j]):
            arr1[k] = arr1[i]
            k += 1
            i += 1
        else:
            arr1[k] = arr2[j]
            k += 1
            j += 1
    for ele in arr1:
        print(ele)

merge(A,B)


Q.2.Given an array of 0s and 1s in random order. Segregate 0s on left side and 1s on right side of the array. Traverse array only once.
A = [ 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
B = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
from sys import maxsize
def Sort_Type(arr):
    type0_idx = 0
    type1_idx = len(arr)-1
    while type0_idx < type1_idx:
        if arr[type0_idx] == 1:
            arr[type0_idx],arr[type1_idx] = arr[type1_idx], arr[type0_idx]
            type1_idx -= 1
        else:
            type0_idx += 1
    return arr

print(Sort_Type(A))
print(Sort_Type(B))

Q.3.Print the elements of an array in the decreasing frequency if 2 numbers have same frequency then print the one which came first.
A = [ 2, 5, 2, 8, 5, 6, 8, 8]
B = [2, 5, 2, 6, -1, 9999999, 5, 8, 8, 8]
from collections import defaultdict
def Sort_freq(arr):
    map = defaultdict(lambda :0)
    for i in arr:
        map[i] += 1
    arr.sort(key=lambda x:(-map[x],x))
    return arr

print(Sort_freq(A))
print(Sort_freq(B))

Q.4.Inversion Count for an array indicates – how far (or close) the array is from being sorted. If the array is already sorted, then the inversion count is 0, but if the array is sorted in the reverse order, the inversion count is the maximum. 
Formally speaking, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j
A = [ 8, 4, 2, 1]
B = [3, 1, 2]
def merge(arr, temp_arr,left, mid, right):
    i = left
    j = mid+1
    k = left
    inv_count = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            k += 1
            j += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
    for ele in range(left, right+1):
        arr[ele] = temp_arr[ele]
    return inv_count
def merge_sort(arr,temp_arr,left,right):
    inv_count = 0
    if left < right:
        mid = (left + right)//2
        inv_count += merge_sort(arr,temp_arr,left,mid)
        inv_count += merge_sort(arr,temp_arr,mid+1, right)
        inv_count += merge(arr,temp_arr,left,mid,right)
    return inv_count
def count_inversion(arr):
    temp_arr = [0]*len(arr)
    return merge_sort(arr,temp_arr,0,len(arr)-1)

print(count_inversion(A))
print(count_inversion(B))

Q.5.Given an array of n distinct elements, find the minimum number of swaps required to sort the array.
A = [ 4, 3, 2, 1]
B = [1, 5, 4, 3, 2]
def min_swap(arr):
    ans = 0
    temp = arr.copy()
    map = {}
    temp.sort()
    for i in range(len(arr)):
        map[arr[i]] = i
    swap_ele = 0
    for i in range(len(arr)):
        if arr[i] != temp[i]:
            ans += 1
            swap_ele = arr[i]
            arr[i], arr[map[temp[i]]] = arr[map[temp[i]]], arr[i]
            map[swap_ele] = map[temp[i]]
            map[temp[i]] = i
    return ans

print(min_swap(A))
print(min_swap(B))

Q.6.Given two sorted arrays, find their union and intersection.
A = [ 1, 3, 4, 5, 7]
B = [2, 3, 5, 6]
def arr_union(arr1,arr2):
    i = 0
    j = 0
    prev = None
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            if arr1[i] != prev:
                print(arr1[i], end = ' ')
                prev = arr1[i]
            i += 1
        elif arr1[i] > arr2[j]:
            if arr2[j] != prev:
                print(arr2[j], end= ' ')
                prev = arr2[j]
            j += 1
        else:
            if arr1[i] != prev:
                print(arr1[i], end=' ')
                prev = arr1[i]
            i+=1
            j+=1
    while i < len(arr1):
        if arr1[i] != prev:
            print(arr1[i], end = ' ')
            prev = arr1[i]
        i += 1
    while j < len(arr2):
        if arr2[j] != prev:
            print(arr2[j], end = ' ')
            prev = arr2[j]
        j += 1

def arr_intersection(arr1,arr2):
    inter = []
    i,j = 0,0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            if len(inter) > 0 and inter[-1] == arr1[i]:
                i += 1
                j += 1
            else:
                inter.append(arr1[i])
                i += 1
                j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    if len(inter):
        return inter
    else:
        return -1


print(arr_union(A,B))
print(arr_intersection(A,B))

we can use set() to solve unsorted arrays as well.

Q.7.Given an array A[] consisting only 0s, 1s and 2s. The task is to write a function that sorts the given array. The functions should put all 0s first, then all 1s and all 2s in last.
A = [ 0, 1, 2, 0, 1, 2]
B = [0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1]
def sort_012(arr):
    low = 0
    high = len(arr)-1
    mid =0
    while mid <= high:
        if arr[mid] == 0:
            arr[low],arr[mid] = arr[mid],arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid],arr[high] = arr[high], arr[mid]
            high -= 1
    return arr
print(sort_012(A))
print(sort_012(B))

Q.8.Given an unsorted array of positive integers, find the number of triangles that can be formed with three different array elements as three sides of triangles. For a triangle to be possible from 3 values, the sum of any of the two values (or sides) must be greater than the third value (or third side). 
A = [ 4, 6, 3, 7]
B = [10, 21, 22, 100, 101, 200, 300]
def count_triangles(arr):
    arr.sort()
    count = 0
    for i in range(len(arr)-1,0,-1):
        l = 0
        r = i - 1
        while l < r:
            if arr[l]+arr[r]>arr[i]:
                count += r-l
                r -= 1
            else:
                l += 1
    return count
print(count_triangles(A))
print(count_triangles(B))

Q.9.Given an integer array and a positive integer k, count all distinct pairs with differences equal to k.
A = [1, 5, 3, 4, 2]
B = [8, 12, 16, 4, 0, 20]
def count_pair(arr,k):
    map = {}
    count = 0
    for i in arr:
        if i  in map:
            map[i] += 1
        else:
            map[i] = 1
    for i in range(len(arr)):
        if arr[i]+k in map:
            count += 1
        if arr[i]-k >= 0 and arr[i]-k in map:
            count += 1
        del map[arr[i]]
    return count

print(count_pair(A,3))
print(count_pair(B,4))
it  can be solved using aorting and two pointors.

Q.10.Given two arrays of same size, we need to convert the first array into another with minimum operations. In an operation, we can either increment or decrement an element by one. Note that orders of appearance of elements do not need to be same.
Here to convert one number into another we can add or subtract 1 from it
A = [ 3, 1, 1 ]
B = [1, 2, 2 ]
def make_arr_equal(arr1,arr2):
    arr1.sort()
    arr2.sort()
    ans = 0
    for i in range(len(arr1)):
        if arr1[i] > arr2[i]:
            ans += abs(arr1[i]-arr2[i])
        elif arr1[i] < arr2[i]:
            ans += abs(arr1[i]-arr2[i])
    return ans

print(make_arr_equal(A,B))

Q.11.Given a binary array, task is to sort this binary array using minimum swaps. We are allowed to swap only adjacent elements
A = [0, 0, 1, 0, 1, 0, 1, 1]
B = [0, 1, 0, 1, 0]
def min_swap(arr):
    count = 0
    zero_unplaced = 0
    for i in range(len(arr)-1,-1,-1):
        if arr[i] == 0:
            zero_unplaced += 1
        else:
            count += zero_unplaced
    return count

print(min_swap(A))
print(min_swap(B))

Q.12.Given an array of an integer of size, N. Array contains N ropes of length Ropes[i]. You have to perform a cut operation on ropes such that all of them are reduced by the length of the smallest rope. Display the number of ropes left after every cut. Perform operations till the length of each rope becomes zero. 
Note: IF no ropes left after a single operation, in this case, we print 0.
A = [5, 1, 1, 2, 3, 5]
B = [5, 1, 6, 9, 8, 11, 2, 2, 6, 5]
def rope_left(arr):
    arr.sort()
    cutting_rope = arr[0]
    operation = 0
    for i in range(1,len(arr)):
        if arr[i] - cutting_rope > 0:
            print(len(arr)-i, end = ' ')
            cutting_rope = arr[i]
            operation += 1
    if operation == 0:
        print(0)

print(rope_left(A))
print(rope_left(B))

Q.13.Given an array of n integers, We need to find all pairs with a difference less than k
A = [1, 10, 4, 2]
B = [1, 8, 7]
from bisect import bisect_left
def pair_diff(arr,k):
    arr.sort()
    ans = 0
    for i in range(len(arr)):
        val = arr[i]+k
        y = bisect_left(arr,val)
        ans += y-i-1
    return ans

print(pair_diff(A,3))
print(pair_diff(B,7))

Q.14.Given an array of n distinct integers. The problem is to find the sum of minimum absolute difference of each array element. For an element x present at index i in the array its minimum absolute difference is calculated as: 
Min absolute difference (x) = min(abs(x – arr[j])), where 1 <= j <= n and j != i and abs is the absolute value. 
A = [5, 10, 1, 4, 8, 7]
B = [12, 10, 15, 22, 21, 20, 1, 8, 9]

def min_absDiff_sum(arr):
    arr.sort()
    sum = 0
    sum += abs(arr[1]-arr[0])
    sum += abs(arr[len(arr)-1]-arr[len(arr)-2])
    for i in range(1,len(arr)-1):
        sum += min(abs(arr[i]-arr[i-1]), abs(arr[i+1]-arr[i]))
    return sum

print(min_absDiff_sum(A))
print(min_absDiff_sum(B))

Q.15.Given two arrays that have the same values but in a different order and having no duplicate elements in it, we need to make a second array the same as a first array using the minimum number of swaps.
A = [3, 6, 4, 8]
B = [4, 6, 8, 3]
def min_swap_sort(arr):
    ans = 0
    temp = arr.copy()
    map = {}
    temp.sort()
    for i in range(len(arr)):
        map[arr[i]] = i
    swap_ele = 0
    for i in range(len(arr)):
        if arr[i] != temp[i]:
            ans += 1
            swap_ele = arr[i]
            arr[i], arr[map[temp[i]]] = arr[map[temp[i]]], arr[i]
            map[swap_ele] = map[temp[i]]
            map[temp[i]] = i
    return ans

def min_swap_equal(arr1,arr2):
    map = {}
    for i in range(len(arr2)):
        map[arr2[i]] = i
    for i in range(len(arr1)):
        arr2[i] = map[arr1[i]]
    return min_swap_sort(arr2)

print(min_swap_equal(A,B))

