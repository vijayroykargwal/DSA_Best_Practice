Optimization Problems:
Q.1.In daily share trading, a buyer buys shares in the morning and sells them on the same day. If the trader is allowed to make at most 2 transactions in a day, whereas the second transaction can only start after the first one is complete (Buy->sell->Buy->sell). Given stock prices throughout the day, find out the maximum profit that a share trader could have made.
A = [10, 22, 5, 75, 65, 80]
B = [2, 30, 15, 10, 8, 25, 80]
from sys import maxsize
def max_profit(arr):
    profit = [0]*len(arr)
    max_price = arr[len(arr)-1]
    for i in range(len(arr)-2,-1,-1):
        if arr[i] > max_price:
            max_price = arr[i]
        profit[i] = max(profit[i+1], max_price-arr[i])
    min_price = arr[0]
    for j in range(1,len(arr)):
        if arr[j] < min_price:
            min_price = arr[j]
        profit[j] = max(profit[j-1], profit[j]+arr[j]-min_price)
    return profit[len(arr)-1]

print(max_profit(A))
print(max_profit(B))

Q.2.Given an array arr[] of size n and integer k such that k <= n.Find the subarray with least average
A = [3, 7, 90, 20, 10, 50, 40]
B = [3, 7, 5, 20, -10, 0, 12]
from sys import maxsize
def least_avg_subarray(arr,k):
    if len(arr) < k:
        return 0
    curr_sum = 0
    for i in range(k):
        curr_sum += arr[i]
    min_sum = curr_sum
    for j in range(k,len(arr)):
        curr_sum += arr[j] - arr[j-k]
        if curr_sum < min_sum:
            min_sum = curr_sum
            res_idx = j-k+1
    return arr[res_idx:res_idx+k]

print(least_avg_subarray(A,3))
print(least_avg_subarray(B,2))

Q.3.Given an unsorted array arr[] and two numbers x and y, find the minimum distance between x and y in arr[]. The array might also contain duplicates. You may assume that both x and y are different and present in arr[].
A = [3, 5, 4, 2, 6, 5, 6, 6, 5, 4, 8, 3]
B = [2, 5, 3, 5, 4, 4, 2, 3]
from sys import maxsize
def min_dist(arr,x,y):
    x_idx, y_idx = -1, -1
    min_dist = maxsize
    for i in range(len(arr)):
        if arr[i]==x:
            x_idx = i
        elif arr[i]==y:
            y_idx = i
        if x_idx != -1 and y_idx != -1:
            min_dist = min(min_dist, abs(x_idx-y_idx))
    if x_idx ==-1 or y_idx == -1:
        return -1
    else:
        return min_dist

print(min_dist(A,3,6))
print(min_dist(B,3,2))

Q.4.Given heights of n towers and a value k. We need to either increase or decrease the height of every tower by k (only once) where k > 0. The task is to minimize the difference between the heights of the longest and the shortest tower after modifications and output this difference.
A = [1, 5, 15, 10]
B = [1, 10, 14, 14, 14, 15]
from sys import maxsize
def min_maxHeightDiff(arr,k):
    arr.sort()
    ans = arr[len(arr)-1] - arr[0]
    tempmin = arr[0]
    tempmax= arr[len(arr)-1]
    for i in range(1,len(arr)):
        tempmin = min(arr[0]+k,arr[i]-k)
        tempmax = max(arr[i-1]+k, arr[len(arr)-1]-k)
        ans = min(ans, tempmax-tempmin)
    return ans

print(min_maxHeightDiff(A,3))
print(min_maxHeightDiff(B,6))

Q.5.Given an array of integers where each element represents the max number of steps that can be made forward from that element. Write a function to return the minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, they cannot move through that element. If the end isn’t reachable, return -1.
A = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]
B = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
from sys import maxsize
def min_jumps(arr):
    maxR = arr[0]
    steps = arr[0]
    jump = 1
    if len(arr) <= 1:
        return 0
    if arr[0]==0:
        return -1
    for i in range(1,len(arr)):
        if i == len(arr)-1:
            return jump
        maxR = max(maxR, i+arr[i])
        steps -= 1
        if steps == 0:
            jump += 1
            if i >= maxR:
                return -1
            steps = maxR - i
    return -1

print(min_jumps(A))
print(min_jumps(B))

Q.6.Given an array of n positive integers. Write a program to find the sum of maximum sum subsequence of the given array such that the integers in the subsequence are sorted in increasing order. For example, if input is {1, 101, 2, 3, 100, 4, 5}, then output should be 106 (1 + 2 + 3 + 100), if the input array is {3, 4, 5, 10}, then output should be 22 (3 + 4 + 5 + 10) and if the input array is {10, 5, 4, 3}, then output should be 10
A = [1, 101, 2, 3, 100, 4, 5]
B = [10, 5, 4, 3]
from sys import maxsize
def max_incSubSum(arr):
    max_sum = [0]*(len(arr))
    for i in range(len(arr)):
        max_sum[i]=arr[i]
    for i in range(1,len(arr)):
        for j in range(i):
            if arr[i]>arr[j] and max_sum[i]<max_sum[j]+arr[i]:
                max_sum[i] = max_sum[j]+arr[i]

    return max(max_sum)

print(max_incSubSum(A))
print(max_incSubSum(B))

Q.7.Given an array of integers and a number x, find the smallest subarray with sum greater than the given value.
A = [1, 4, 45, 6, 0, 19]
B = [1, 10, 5, 2, 7]
from sys import maxsize
def smallest_subwithsum(arr,x):
    min_len = len(arr)+1
    curr_sum = 0
    i,j =0,0
    while j < len(arr):
        while curr_sum <= x and j < len(arr):
            curr_sum += arr[j]
            j += 1
        while curr_sum > x and i < len(arr):
            min_len = min(min_len, j-i)
            curr_sum -= arr[i]
            i += 1
    return min_len

print(smallest_subwithsum(A,51))
print(smallest_subwithsum(B,9))

Q.8.Given an array with positive and negative numbers, find the maximum average subarray of the given length.
A = [1, 12, -5, -6, 50, 3]
B = [1, 10, 5, 2, 7]
from sys import maxsize
def maxAvg_Subarray(arr,k):
    if k > len(arr):
        return -1
    curr_sum = sum(arr)
    max_sum = curr_sum
    max_kidx = k-1
    for i in range(k,len(arr)):
        curr_sum += arr[i] - arr[i-k]
        if curr_sum > max_sum:
            max_sum = curr_sum
            max_kidx = i
    max_avg = sum(arr[max_kidx-k+1:max_kidx+1]) // k
    res_idx = max_kidx - k+1
    return max_avg
print(maxAvg_Subarray(A,4))
print(maxAvg_Subarray(B,3))

Q.9.Consider an array with n elements and value of all the elements is zero. We can perform following operations on the array. 
1.Incremental operations:Choose 1 element from the array and increment its value by 1.
2.Doubling operation: Double the values of all the elements of array.
We are given desired array target[] containing n elements. Compute and return the smallest possible number of the operations needed to change the array from all zeros to desired array. 
A = [2, 3]
B = [16, 16, 16]
from sys import maxsize
def count_minStep(arr):
    res = 0
    while True:
        zero_count = 0
        i = 0
        while i < len(arr):
            if arr[i]&1 > 0:
                break
            elif arr[i] == 0:
                zero_count += 1
            i+=1
        if zero_count == len(arr):
            return res
        if i == len(arr):
            for j in range(len(arr)):
                arr[j] = arr[j]//2
            res += 1
        for j in range(i,len(arr)):
            if arr[j] & 1:
                arr[j] -= 1
                res += 1

print(count_minStep(A))
print(count_minStep(B))

Q.10.You are given an array of n-elements, you have to find the number of subsets whose product of elements is less than or equal to a given integer k.
A = [2, 4, 5, 3]
B = [12, 32, 21]
import bisect
def create_subset(arr,sub,k):
    total = 2**(len(arr))
    for i in range(0,total):
        product = 1
        for j in range(0,len(arr)):
            if i & (1<<j) != 0:
                product *= arr[j]
        if product <= k:
            sub.append(product)
    return
def subset_product(arr,k):
    arr1,arr2,sub1,sub2=[],[],[],[]
    for i in range(len(arr)):
        if arr[i] > k:
            continue
        if i < len(arr)//2:
            arr1.append(arr[i])
        else:
            arr2.append(arr[i])

    create_subset(arr1,sub1,k)
    create_subset(arr2,sub2,k)
    sub2.sort()
    ans = 0
    for ele in sub1:
        ans += bisect.bisect(sub2, k//ele)
    return ans-1

print(subset_product(A,12))
print(subset_product(B,1))

Q.11.Given an array of positive integers. We need to make the given array a ‘Palindrome’. The only allowed operation is”merging” (of two adjacent elements). Merging two adjacent elements means replacing them with their sum. The task is to find the minimum number of merge operations required to make the given array a ‘Palindrome’.
To make any array a palindrome, we can simply apply merge operation n-1 times where n is the size of the array (because a single-element array is always palindromic, similar to single-character string). In that case, the size of array will be reduced to 1. But in this problem, we are asked to do it in the minimum number of operations.
A = [1, 4, 5, 1]
B = [11, 14, 15, 99]
def minOp_Palind(arr):
    ans = 0
    i,j = 0, len(arr)-1
    while i <=j:
        if arr[i]==arr[j]:
            i+=1
            j-=1
        elif arr[i] > arr[j]:
            j -= 1
            arr[j] += arr[j+1]
            ans += 1
        elif arr[j] > arr[i]:
            i += 1
            arr[i] += arr[i-1]
            ans += 1
    return ans

print(minOp_Palind(A))
print(minOp_Palind(B))

Q.12.Given an array of positive numbers, find the smallest positive integer value that cannot be represented as the sum of elements of any subset of a given set. 
The expected time complexity is O(nlogn).
A = [1, 10, 3, 11, 6, 15]
B = [1, 2, 5, 10, 20, 40]
C = [1, 1, 4, 3]
def smallest_value(arr):
    arr.sort()
    ans = 1
    for i in range(0,len(arr)):
        if arr[i] <= ans:
            ans += arr[i]
        else:
            break
    return ans

print(smallest_value(A))
print(smallest_value(B))
print(smallest_value(C))

Q.13.An array is given, find length of the subarray having maximum sum.
A = [1, -2, 1, 1, -2, 1]
B = [-2, -3, 4, -1, -2, 1, 5, -3]
from sys import maxsize
def maxSum_Subarray(arr):
    max_so_far = -maxsize-1
    max_ending_here = 0
    start,end = 0,0
    s = 0
    for i in range(0,len(arr)):
        max_ending_here += arr[i]
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = s
            end = i
        if max_ending_here < 0:
            max_ending_here = 0
            s = i+1
    return end - start + 1

print(maxSum_Subarray(A))
print(maxSum_Subarray(B))

Q.14.Given an unsorted array, find the minimum difference between any pair in given array.
A = [1, 5, 3, 19, 18, 25]
B = [1, 19, -4, 31, 38, 25, 100]
from sys import maxsize
def minDiff_Pair(arr):
    arr.sort()
    diff = maxsize
    for i in range(len(arr)-1):
        if arr[i+1]-arr[i] < diff:
            diff = arr[i+1]-arr[i]
    return diff

print(minDiff_Pair(A))
print(minDiff_Pair(B))

Q.15.Given two binary arrays, arr1[] and arr2[] of the same size n. Find the length of the longest common span (i, j) where j >= i such that arr1[i] + arr1[i+1] + …. + arr1[j] = arr2[i] + arr2[i+1] + …. + arr2[j].The expected time complexity is Θ(n).
A = [0, 1, 0, 0, 0, 0]
B = [1, 0, 1, 0, 0, 1]
from sys import maxsize
def longest_common_sum(arr1,arr2):
    maxlen = 0
    presum1,presum2 = 0,0
    diff = {}
    for i in range(len(arr1)):
        presum1 += arr1[i]
        presum2 += arr2[i]
        curr_diff = presum1 - presum2
        if curr_diff ==0:
            maxlen = i+1
        elif curr_diff not in diff:
            diff[curr_diff]=i
        else:
            length = i - diff[curr_diff]
            maxlen = max(maxlen, length)
    return maxlen

print(longest_common_sum(A,B))
A = [0, 1, 0, 1, 1, 1, 1]
B = [1, 1, 1, 1, 1, 0, 1]
print(longest_common_sum(A,B))

Misc Problems :
Q.1.Given an array of positive numbers and a number k, find the number of subarrays having product exactly equal to k. We may assume that there is no overflow.
A = [2, 1, 1, 1, 4, 5]
B = [1, 2, 3, 4, 1]
from sys import maxsize
def countOne(arr):
    i,length,ans = 0,0,0
    while i < len(arr):
        if arr[i] == 1:
            length = 0
            while i < len(arr):
                i+=1
                length +=1
            ans += (length*(length+1))//2
        i += 1
    return ans

def nof_sub_product(arr,k):
    start,end,p,countOnes,res = 0,0,1,0,0
    while end < len(arr):
        p = p*arr[end]
        if p > k:
            while start <= end and p > k:
                p = p//arr[start]
                start += 1
        if p == k:
            countOnes = 0
            while end + 1 < len(arr) and arr[end+1]==1:
                countOnes += 1
                end += 1
            res += countOnes +1
            while start <= end and arr[start] == 1 and k!=1:
                res+= countOnes +1
                start +=1
            p = p//arr[start]
            start += 1
        end += 1
    return res


#print(countOne(A))
print(nof_sub_product(A,4))
print(nof_sub_product(B,24))

Q.2.Given an unsorted array of numbers, write a function that returns true if the array consists of consecutive numbers.
using sorting:
A = [5, 2, 3, 1, 4]
B = [34, 23, 52, 12, 3]
from sys import maxsize
def is_Consecutive(arr):
    arr.sort()
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]+1:
            return False
        return True
print(is_Consecutive(A))
print(is_Consecutive(B))

using XOR:
A = [5, 2, 3, 1, 4]
B = [34, 23, 52, 12, 3]
from sys import maxsize
def is_Consecutive(arr):
    n = len(arr)
    min_ele = arr.index(min(arr))
    num = 0
    for i in range(0, n):
        num ^= arr[min_ele] ^ arr[i]
    if num == 0:
        return True
    return False

print(is_Consecutive(A))
print(is_Consecutive(B))



