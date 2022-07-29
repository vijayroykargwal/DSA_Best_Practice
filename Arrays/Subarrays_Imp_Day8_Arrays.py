General Things:
Array of n elements has total:
Sub-Array = n*(n+1)/2
subsequences = 2^n-1 (order of element matters)
subsets = 2^n-1(non-empty)

Problems on Subarrays:
General : Print All Subarray
A = [1,2,3]
B = [1,2,3,4]

def all_subarray(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr) + 1):
            print(*arr[i:j])

print(all_subarray(A))
print(all_subarray(B))

Q1.Given an array of both positive and negative integers and a number K. The task is to check if any subarray with product K is present in the array or not.
A = [-2, -1, 3, -4, 5]
B = [3, -1, -1, -1, 5]

def subarray_given_product(arr,k):
    min_val = arr[0]
    max_val = arr[0]
    max_prod = arr[0]
    for i in range(1,len(arr)):
        if arr[i]<0:
            max_val,min_val=min_val,max_val
        max_val = max(arr[i],max_val*arr[i])
        min_val = min(arr[i],min_val*arr[i])
        if min_val == k or max_val == k:
            return True
        max_prod = max(max_prod,max_val)
    return False

print(subarray_given_product(A,2))
print(subarray_given_product(A,3))

Q2.Given an array arr[], an integer K and a Sum. The task is to check if there exists any subarray with K elements whose sum is equal to the given sum. If any of the subarray with size K has the sum equal to the given sum then print YES otherwise print NO.
A = [1, 4, 2, 10, 2, 3, 1, 0, 20]
B = [1, 4, 2, 10, 2, 3, 1, 0, 20]

def subarray_given_sum(arr,k,x):
    curr_sum = 0
    for i in range(k):
        curr_sum += arr[i]
    if curr_sum == x:
        return True
    for j in range(k,len(arr)):
        curr_sum += arr[j] - arr[j-k]
        if curr_sum == x:
            return True
    return False

print(subarray_given_sum(A,4,18))
print(subarray_given_sum(B,3,6))

Q3.Given an array of N integers and a number K, the task is to find the number of subarrays such that all elements are greater than K in it. 
A = [3, 4, 5, 6, 7, 2, 10, 11]
B = [8, 25, 10, 19, 19, 18, 20, 11, 18]

def subarray_greater_K(arr,k):
    count = 0
    number = 0
    for i in range(len(arr)):
        if arr[i]>k:
            count += 1
        else:
            number += count*(count+1)//2
            count = 0
    if count:
        number += count*(count+1)//2
    return number

print(subarray_greater_K(A,5))
print(subarray_greater_K(B,13))

Q4.Given a character array arr[] containing only lowercase English alphabets, the task is to print the maximum length of the subarray such that the first and the last element of the sub-array are same.
A = ['g', 'e', 'e', 'k', 's']
B = ['a', 'b', 'c', 'd', 'a']
class Element:
    def __init__(self):
        self.first_occ = -1
        self.last_occ = -1
    def update_occ(self,idx):
        if self.first_occ == -1:
            self.first_occ = idx
        self.last_occ = idx

def maxLen_Subarray(arr):
    ele = [None]*26
    for i in range(len(arr)):
        ch = ord(arr[i])-ord('a')
        if ele[ch] == None:
            ele[ch] = Element()
        ele[ch].update_occ(i)
    maxLen = 0
    for i in range(0,26):
        if ele[i]!=None:
            length = (ele[i].last_occ - ele[i].first_occ+1)
            maxLen = max(maxLen, length)
    return maxLen

print(maxLen_Subarray(A))
print(maxLen_Subarray(B))

Q5.Given two arrays A[] and B[] consisting of n and m integers. The task is to check whether the array B[] is a subarray of the array A[] or not.
A = [2, 3, 0, 5, 1, 1, 2]
B = [3, 0, 5, 1]

def check_Subarray(arr1,arr2):
    i = 0
    j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i]==arr2[j]:
            i += 1
            j += 1
            if j == len(arr2):
                return True
        else:
            i = i-j+1
            j = 0
    return False

print(check_Subarray(A,B))

Q6.Given an array arr[] of N integers and another integer K. The task is to find the maximum sum of a subsequence such that the difference of the indices of all consecutive elements in the subsequence in the original array is exactly K. For example, if arr[i] is the first element of the subsequence then the next element has to be arr[i + k] then arr[i + 2k] and so on.
A = [2, -3, -1, -1, 2]
B = [2, 3, -1, -1, 2]
from sys import maxsize
def maxSub_arr_sum(arr,k,i):
    max_so_far = -maxsize-1
    curr_sum = 0
    while i < len(arr):
        curr_sum += arr[i]
        if curr_sum > max_so_far:
            max_so_far = curr_sum
        if curr_sum < 0:
            curr_sum = 0
        i += k
    return max_so_far
def maxSub_Seq_Sum(arr,k):
    maxsum = 0
    for i in range(0, min(len(arr),k)+1):
        maxsum = max(maxsum,maxSub_arr_sum(arr,k,i))
    return maxsum

print(maxSub_Seq_Sum(A,2))
print(maxSub_Seq_Sum(B,3))

Q7.Given an array arr[] of length N, the task is the find the length of the longest sub-array with the maximum possible GCD value.
A = [1, 2, 2]
B = [3, 3, 3, 3]
from sys import maxsize
def max_Len_Subarray(arr):
    x = 0
    for i in range(len(arr)):
        x = max(x, arr[i])
    ans = 0
    for i in range(len(arr)):
        if arr[i] != x:
            continue
        j = i
        while arr[j] == x:
            j += 1
            if j >= len(arr):
                break
        ans = max(ans, j-i)
    return ans

print(max_Len_Subarray(A))
print(max_Len_Subarray(B))

Q8.Given an array arr[] of size N and an integer K > 0. The task is to find the number of subarrays with sum at least K.
Input: A = [6, 1, 2, 7], K = 10 
Output: 2 
{6, 1, 2, 7} and {1, 2, 7} are the only valid subarrays.
Input: B = [3, 3, 3], K = 5 
Output: 3 

A = [6, 1, 2, 7]
B = [3, 3, 3]
from sys import maxsize
def subarray_K_Sum(arr,k):
    r,sum = 0,0
    ans = 0
    for l in range(len(arr)):
        while sum < k:
            if r == len(arr):
                break
            else:
                sum += arr[r]
                r += 1
        if sum < k:
            break
        ans += len(arr)-r+1
        sum -= arr[l]
    return ans

print(subarray_K_Sum(A,10))
print(subarray_K_Sum(B,5))


Q.9.Given an array arr[] and an integer K, the task is to calculate the sum of all subarrays of size K.
Input: arr[] = {1, 2, 3, 4, 5, 6}, K = 3 
Output: 6 9 12 15 
Explanation: 
All subarrays of size k and their sum: 
Subarray 1: {1, 2, 3} = 1 + 2 + 3 = 6 
Subarray 2: {2, 3, 4} = 2 + 3 + 4 = 9 
Subarray 3: {3, 4, 5} = 3 + 4 + 5 = 12 
Subarray 4: {4, 5, 6} = 4 + 5 + 6 = 15

A = [1, 2, 3, 4, 5, 6]
B = [1, -2, 3, -4, 5, 6]
from sys import maxsize
def subarray_Sum(arr,k):
    sum = 0
    for i in range(k):
        sum += arr[i]
    print(sum, end = ' ')
    for i in range(k,len(arr)):
        sum += arr[i] - arr[i-k]
        print(sum, end=' ')

print(subarray_Sum(A,3))
print(subarray_Sum(B,2))


Q.10.Given an array arr[] and integer K, the task is to find the minimum bitwise XOR sum of any subarray of size K in the given array.
A = [3, 7, 90, 20, 10, 50, 40]
B = [3, 7, 5, 20, -10, 0, 12]
from sys import maxsize
def min_XOR_sum(arr, k):
    if len(arr) < k:
        return -1
    curr_xor = 0
    for i in range(k):
        curr_xor = curr_xor ^ arr[i]
    min_xor = curr_xor
    for i in range(k,len(arr)):
        curr_xor ^= (arr[i]^arr[i-k])
        if curr_xor < min_xor:
            min_xor = curr_xor
            res_idx = i-k+1
    print(min_xor)

print(min_XOR_sum(A,3))
print(min_XOR_sum(B,2))


Q.11.Given an array a[] of N integers, the task is to find the length of the longest Alternating Even Odd subarray present in the array. 
A = [1, 2, 3, 4, 5, 7, 9]
B = [1, 3, 5]
from sys import maxsize
def longest_even_odd(arr):
    longest = 1
    count = 1
    for i in range(len(arr)-1):
        if (arr[i]+arr[i+1])%2  == 1:
            count += 1
        else:
            longest = max(longest,count)
            count = 1
    if longest == 1:
        return 0
    return max(count,longest)


print(longest_even_odd(A))
print(longest_even_odd(B))

Q.12.Given an array A of size N where the array elements contain values from 1 to N with duplicates, the task is to find the total number of subarrays that start and end with the same element.
A = [1, 2, 1, 5, 2]
B = [1, 5, 6, 1, 9, 5, 8, 10, 8, 9]
from sys import maxsize
def start_end_same(arr):
    ans = 0
    freq = [0]*(len(arr)+1)
    for i in range(len(arr)):
        freq[arr[i]] += 1
    for i in range(1,len(arr)+1):
        freq_i = freq[i]
        ans += freq_i*(freq_i+1)//2
    print(ans)
print(start_end_same(A))
print(start_end_same(B))


Q.13.Given a unsorted integer array arr[] and an integer K. The task is to count the number of subarray with exactly K Perfect Square Numbers. 
A = [2, 4, 9, 3]
B = [4, 2, 5]
from collections import defaultdict
import math
def isperfect_square(x):
    sr = math.sqrt(x)
    return sr - math.floor(sr) == 0
def subarray_sum(arr,k):
    prev_sum = defaultdict(int)
    ans = 0
    curr_sum = 0
    for i in range(len(arr)):
        curr_sum += arr[i]
        if curr_sum == k:
            ans += 1
        if (curr_sum - k) in prev_sum:
            ans += prev_sum[curr_sum-k]
        prev_sum[curr_sum] += 1
    return ans
def k_perfect_square(arr,k):
    for i in range(len(arr)):
        if isperfect_square(arr[i]):
            arr[i] = 1
        else:
            arr[i] = 0
    return subarray_sum(arr,k)

print(k_perfect_square(A,2))
print(k_perfect_square(B,3))

Q.14.Given an array arr[] consisting of integers, the task is to split the given array into two sub-arrays such that the difference between their maximum elements is minimum. 
using sorting:
A = [7, 9, 5, 10]
B = [6, 6, 6]

def minDiff_max(arr):
    arr.sort()
    return arr[len(arr)-1]-arr[len(arr)-2]
print(minDiff_max(A))
print(minDiff_max(B))


using efficient largest and second_largest:
A = [7, 9, 5, 10]
B = [6, 6, 6]
from sys import maxsize
def minDiff_max(arr):
    first = -maxsize-1
    second = -maxsize-1
    for i in range(len(arr)):
        if arr[i] > first:
            second = first
            first = arr[i]
        elif arr[i] > second:
            second = arr[i]
    return first - second
print(minDiff_max(A))
print(minDiff_max(B))


Q.15.Given a very large number N in the form of a string and a number K, the task is to print all the K-digit repeating numbers whose frequency is greater than 1. 
A = '123412345123456'
B = '1432543214325432'

def repeating_kdigit(arr,k):
    map = {}
    for i in range(len(arr)):
        ele = arr[i:i+k]
        map[ele] = 0
    for i in range(len(arr)):
        map[arr[i:i+k]] += 1
    for key,value in map.items():
        if value > 1:
            print(key,'-',value)
print(repeating_kdigit(A,4))
print(repeating_kdigit(B,5))


Q.16.Given an array arr[] of size N and an integer k, our task is to find the length of longest subarray whose sum of elements is not divisible by k. If no such subarray exists then return -1.
A = [8, 4, 3, 1, 5, 9, 2]
B = [6, 3, 12, 15]

def repeating_kdigit(arr,k):
    left = -1
    sum = 0
    for i in range(len(arr)):
        if arr[i]%k != 0:
            if left == -1:
                left = i
            right = i
        sum += arr[i]
    if sum %k !=0:
        return len(arr)
    elif left == -1:
        return -1
    else:
        pre_len = left+1
        suffix_len = len(arr) - right
        return len(arr)-min(pre_len,suffix_len)

print(repeating_kdigit(A,2))
print(repeating_kdigit(B,3))


Q.17.Given an array arr[] of size N and integer Y, the task is to find a minimum of all the differences between the maximum and minimum elements in all the sub-arrays of size Y.
A = [3, 2, 4, 5, 6, 1, 9 ]
B = [ 1, 2, 3, 3, 2, 2]
from sys import maxsize
def minDiff_btw_minmax(arr,y):
    submin = arr.copy()
    submax = arr.copy()
    stack = []
    minn = maxsize
    maxx = -maxsize-1
    for i in range(y):
        stack.append(arr[i])
        maxx = max(arr[i],maxx)
        minn = min(arr[i],minn)
    diff = maxx-minn
    for i in range(y,len(arr)):
        stack.remove(arr[i-y])
        stack.append(arr[i])
        maxx = max(stack)
        minn = min(stack)
        diff = min(diff,maxx-minn)
    return diff

print(minDiff_btw_minmax(A,3))
print(minDiff_btw_minmax(B,4))

efficient :
A = [3, 2, 4, 5, 6, 1, 9 ]
B = [ 1, 2, 3, 3, 2, 2]
from sys import maxsize
def get_submaxarr(arr, n, y):
    j = 0
    stk = []
    maxarr = [0] * n
    stk.append(0)
    for i in range(1, n):
        while (len(stk) > 0 and
               arr[i] > arr[stk[-1]]):
            maxarr[stk[-1]] = i - 1
            stk.pop()
        stk.append(i)
    while (stk):
        maxarr[stk[-1]] = n - 1
        stk.pop()
    submax = []
    for i in range(n - y + 1):
        while (maxarr[j] < i + y - 1 or
               j < i):
            j += 1
        submax.append(arr[j])
    return submax
def get_subminarr(arr, n, y):
    j = 0
    stk = []
    minarr = [0] * n
    stk.append(0)
    for i in range(1, n):
        while (stk and arr[i] < arr[stk[-1]]):
            minarr[stk[-1]] = i
            stk.pop()
        stk.append(i)
    while (stk):
        minarr[stk[-1]] = n
        stk.pop()

    submin = []

    for i in range(n - y + 1):
        while (minarr[j] <= i + y - 1 or
               j < i):
            j += 1

        submin.append(arr[j])
    return submin
def getMinDifference(Arr, N, Y):
    submin = get_subminarr(Arr, N, Y)
    submax = get_submaxarr(Arr, N, Y)
    minn = submax[0] - submin[0]
    b = len(submax)

    for i in range(1, b):
        diff = submax[i] - submin[i]
        minn = min(minn, diff)
    print(minn)

print(getMinDifference(A,len(A),3))
print(getMinDifference(B,len(B),4))


Q.18.Given an array arr[] of N integers, the task is to find the size of the largest subarray with frequency of all elements the same.
A = [1, 2, 2, 5, 6, 5, 6]
B = [ 1, 1, 1, 1, 1]

def largest_subarray_freq(arr):
    ans = 0
    for i in range(len(arr)):
        map1 = {}
        map2 = {}
        for j in range(i,len(arr)):
            if arr[j] not in map1:
                ele_count = 0
            else:
                ele_count = map1[arr[j]]
            if arr[j] in map1:
                map1[arr[j]] += 1
            else:
                map1[arr[j]] = 1
            if ele_count in map2:
                if map2[ele_count]==1:
                    del map2[ele_count]
                else:
                    map2[ele_count] -= 1
            if ele_count+1 in map2:
                map2[ele_count+1] += 1
            else:
                map2[ele_count+1] = 1
            if len(map2) == 1:
                ans = max(ans, j-i+1)
    return ans



print(largest_subarray_freq(A))
print(largest_subarray_freq(B))

Q.19.Given an array arr[] of integers, the task is to find the total count of subarrays such that the sum of elements at even position and sum of elements at the odd positions are equal.
A = [1, 2, 3, 4, 1]
B = [2, 4, 6, 4, 2]

def count_even_odd_sum(arr):
    count = 0
    for i in range(len(arr)):
        sum = 0
        for j in range(i,len(arr)):
            if (j-1)%2 == 0:
                sum += arr[j]
            else:
                sum -= arr[j]
            if sum==0:
                count += 1
    return count

print(count_even_odd_sum(A))
print(count_even_odd_sum(B))

Q.20.Given an array arr[] consisting of N integers, the task is to find the largest subarray consisting of unique elements only.
A = [1, 2, 3, 4, 5, 1, 2, 3]
B = [1, 2, 4, 4, 5, 6, 7, 8, 3, 4, 5, 3, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
from collections import defaultdict
def largest_subarray_unique(arr):
    idx = defaultdict(lambda :0)
    ans =0
    j = 0
    for i in range(len(arr)):
        j = max(idx[arr[i]],j)
        ans = max(ans,i-j+1)
        idx[arr[i]]=i+1
        i += 1
    return ans

print(largest_subarray_unique(A))
print(largest_subarray_unique(B))

Q.21.Given an array arr[] consisting of N non-negative integers, the task is to find the minimum number of subarrays that needs to be reduced by 1 such that all the array elements are equal to 0.
A = [1, 2, 3, 2, 1]
B = [5, 4, 3, 4, 4]

def min_decrements(arr):
    if len(arr)==0:
        return 0
    ans = arr[0]
    for i in range(1,len(arr)):
        if arr[i] > arr[i-1]:
            ans += arr[i]-arr[i-1]
    return ans

print(min_decrements(A))
print(min_decrements(B))


Q.22.Given an integer array arr[], the task is to split the given array into two subarrays such that the difference between their sum is minimum.
A = [7, 9, 5, 10]
B = [6, 6, 6]
from sys import maxsize
def minDiff_Sum(arr):
    total_sum = 0
    min_diff = maxsize
    prefix_sum = 0
    for i in range(len(arr)):
        total_sum += arr[i]
    for i in range(len(arr)-1):
        prefix_sum += arr[i]
        diff = abs((total_sum-prefix_sum)-prefix_sum)
        if diff < min_diff:
            min_diff = diff
    return min_diff

print(minDiff_Sum(A))
print(minDiff_Sum(B))

Q.23.Given an array arr[] and an integer K, the task is to print the maximum number of non-overlapping subarrays with a sum equal to K.
A = [-2, 6, 6, 3, 5, 4, 1, 2, 8]
B = [1, 1, 1]
from sys import maxsize
def max_non_over(arr,k):
    st = set()
    prefix_sum = 0
    st.add(prefix_sum)
    ans = 0
    for i in range(len(arr)):
        prefix_sum += arr[i]
        if (prefix_sum - k) in st:
            ans += 1
            prefix_sum = 0
            st.clear()
            st.add(0)
        st.add(prefix_sum)
    return ans

print(max_non_over(A,10))
print(max_non_over(B,2))

Q.24.Given an array, arr[] of size N, the task is to split the array into the maximum number of subarrays such that the first and the last occurrence of all distinct array element lies in a single subarray.
A = [1, 1, 2, 2]
B = [1, 2, 4, 1, 4, 7, 7, 8]
from sys import maxsize
def max_count_firstLast(arr):
    hash = [0]*(len(arr)+1)
    for i in range(len(arr)):
        hash[arr[i]] = i
    max_idx = -1
    ans = 0
    for i in range(len(arr)):
        max_idx = max(max_idx,hash[arr[i]])
        if max_idx == i:
            ans+=1
    return ans

print(max_count_firstLast(A))
print(max_count_firstLast(B))


Q.25.Given an array arr[] consisting of N positive integers, the task is to find the maximum product of subarray sum with the minimum element of that subarray.
A = [3, 1, 6, 4, 5, 2]
B = [4, 1, 2, 9, 3]
from sys import maxsize
def maxProduct_subarrSum(arr):
    pre_sum = [0]*len(arr)
    pre_sum[0]=arr[0]
    for i in range(1,len(arr)):
        pre_sum[i] = pre_sum[i-1]+arr[i]
    l = [0]*len(arr)
    r = [0]*len(arr)
    st = []
    for i in range(1,len(arr)):
        while len(st) and arr[st[len(st)-1]] >= arr[i]:
            st.remove(st[len(st)-1])
        if len(st):
            l[i] = st[len(st)-1] + 1
        else:
            l[i] = 0
        st.append(i)
    while len(st):
        st.remove(st[len(st)-1])
    i = len(arr)-1
    while i >= 0:
        while len(st) and arr[st[len(st)-1]] >= arr[i]:
            st.remove(st[len(st)-1])
            if len(st):
                r[i]=st[len(st)-1]-1
            else:
                r[i]=len(arr)-1
        st.append(i)
        i -= 1
    max_prod = 0
    for i in range(len(arr)):
        if l[i]==0:
            temp_prod = arr[i]*pre_sum[r[i]]
        else:
            temp_prod = arr[i]*(pre_sum[r[i]]-pre_sum[l[i]-1])
        max_prod = max(max_prod,temp_prod)
    return max_prod


print(maxProduct_subarrSum(A))
print(maxProduct_subarrSum(B))

Q.26.Given an array arr[] of N positive integers, the task is to find the sum of the product of elements of all the possible subarrays.
A = [1, 2, 3]
B = [1, 2, 3, 4]
from sys import maxsize
def sumof_subarr_prod(arr):
    ans = 0
    res = 0
    i = len(arr)-1
    while i >= 0:
        incr = arr[i]*(1+res)
        ans += incr
        res = incr
        i -=1
    return ans

print(sumof_subarr_prod(A))
print(sumof_subarr_prod(B))


Q.27.Given an array arr[] consisting of N integers, the task is to print the length of the smallest subarray to be removed from arr[] such that the remaining array is sorted
A = [1, 2, 3, 10, 4, 2, 3, 5]
B = [5, 4, 3, 2, 1]
from sys import maxsize
def remove_subarr_sorted(arr):
    min_len = maxsize
    left = 0
    right = len(arr)-1
    while left < right and arr[left+1] >= arr[left]:
        left += 1
    if left == len(arr)-1:
        return 0
    while right > left and arr[right-1] <= arr[right]:
        right -= 1
    min_len = min(len(arr)-left-1, right)
    j = right
    for i in range(left+1):
        if arr[i]<=arr[j]:
            min_len = min(min_len,j-i-1)
        elif j < len(arr) - 1:
            j+=1
        else:
            break
    return min_len

print(remove_subarr_sorted(A))
print(remove_subarr_sorted(B))

Q.28.Given an array arr[] consisting of N integers and an integer K, the task is to find the length of the longest subarray such that each element occurs K times.
A = [3, 5, 2, 2, 4, 6, 4, 6, 5]
B = [5, 5, 5, 5]
from collections import defaultdict
def longest_subarr_freqk(arr,k):
    ans = 0
    for i in range(len(arr)):
        map1 = defaultdict(int)
        map2 = defaultdict(int)
        for j in range(i,len(arr)):
            if arr[j] not in map1:
                prev_freq = 0
            else:
                prev_freq = map1[arr[j]]
            map1[arr[j]] += 1
            if prev_freq in map2:
                if map2[prev_freq] == 1:
                    del map2[prev_freq]
                else:
                    map2[prev_freq] -= 1
            new_freq = prev_freq + 1
            map2[new_freq] += 1
            if len(map2) == 1 and new_freq == k:
                ans = max(ans,j-i+1)
    return ans

print(longest_subarr_freqk(A,2))
print(longest_subarr_freqk(B,3))

Q.29.Given an array arr[] of integers and an integer K, the task is to find the length of the smallest subarray that needs to be removed such that the sum of remaining array elements is divisible by K. Removal of the entire array is not allowed. If it is impossible, then print “-1”.
A = [3, 1, 4, 2]
B = [3, 6, 7, 1]
from sys import maxsize
def len_smallest_subarr(arr,k):
    mod_arr = [0]*len(arr)
    total_sum = 0
    for i in range(len(arr)):
        mod_arr[i] = (arr[i]+k)%k
        total_sum += arr[i]
    target_rem = total_sum % k
    if target_rem == 0:
        return 0
    map = {}
    map[0] = -1
    curr_rem = 0
    res = maxsize
    for i in range(len(arr)):
        curr_rem = (curr_rem + arr[i]+k)%k
        map[curr_rem] = i
        mod = (curr_rem - target_rem + k)%k
        if mod in map.keys():
            res = min(res, i-map[mod])
    if res == maxsize or res == len(arr) :
        res = -1
    return res

print(len_smallest_subarr(A,6))
print(len_smallest_subarr(B,9))

Q.30.Given an array arr[] consisting of N integers, the task is to count ways to split array into two subarrays of equal sum by changing the sign of any one array element.
A = [2, 2, -3, 3]
B = [2, 2, 1, -3, 3]
from sys import maxsize
def count_2subarr_equalsum(arr):
    prefix_count = {}
    suffix_count={}
    total = 0
    for i in range(len(arr)-1,-1,-1):
        total += arr[i]
        suffix_count[arr[i]] = suffix_count.get(arr[i],0)+1
    prefix_sum = 0
    suffix_sum = 0
    count = 0
    for i in range(len(arr)-1):
        prefix_sum += arr[i]
        prefix_count[arr[i]] = prefix_count.get(arr[i],0)+1
        suffix_sum = total-prefix_sum
        suffix_count[arr[i]] -= 1
        diff = prefix_sum - suffix_sum
        if diff % 2 == 0:
            y,z = 0,0
            if -diff//2 in suffix_count:
                y = suffix_count[-diff//2]
            if diff//2 in prefix_count:
                z = prefix_count[diff//2]
            x = z+y
            count += x
    return count


print(count_2subarr_equalsum(A))
print(count_2subarr_equalsum(B))

Q.31.Given an array arr[] consisting of N integers and an integer K, the task is to find the length of the longest subarray in which all the elements are smaller than K.
Constraints:
0 <= arr[i]  <= 10^5
A = [1, 8, 3, 5, 2, 2, 1, 13]
B = [8, 12, 15, 1, 3, 9, 2, 10]
from sys import maxsize
def len_longest_subarr(arr,k):
    count = 0
    length = 0
    for i in range(len(arr)):
        if arr[i] < k:
            count += 1
        else:
            length = max(length,count)
            count = 0
    if count:
        length = max(length,count)
    return length
print(len_longest_subarr(A,6))
print(len_longest_subarr(B,10))

Q.32.Given an array arr[] of size N, the task is to find the maximum product from any subarray consisting of elements in strictly increasing or decreasing order.
A = [1, 2, 10, 8, 1, 100, 101 ]
B = [ 1, 5, 7, 2, 10, 12 ]
from sys import maxsize
def max_subarr_prod(arr):
    max_ending_here = 1
    min_ending_here = 1
    max_so_far = 0
    flag = 0
    for i in range(len(arr)):
        if arr[i]>0:
            max_ending_here = max_ending_here * arr[i]
            min_ending_here = min(min_ending_here*arr[i],1)
            flag = 1
        elif arr[i]==0:
            max_ending_here = 1
            min_ending_here = 1
        else:
            temp = max_ending_here
            max_ending_here = max(max_ending_here*arr[i],1)
            min_ending_here = temp*arr[i]
        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
    if flag == 0 and max_so_far == 0:
        return 0
    return max_so_far

def find_max_prod(arr):
    i = 0
    max_prod = -maxsize-1
    while i < len(arr):
        v = []
        v.append(arr[i])
        if i < len(arr)-1 and arr[i] < arr[i+1]:
            while i < len(arr)-1 and arr[i] < arr[i+1]:
                v.append(arr[i+1])
                i += 1
        elif i < len(arr)-1 and arr[i]>arr[i+1]:
            while i < len(arr)-1 and arr[i]>arr[i+1]:
                v.append(arr[i+1])
                i += 1
        prod = max_subarr_prod(v)
        max_prod = max(max_prod,prod)
        i+=1
    return max_prod



print(find_max_prod(A))
print(find_max_prod(B))

