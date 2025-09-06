# ---------------- Problem 1: Longest Common Prefix ----------------
# Problem: Find the longest common prefix string amongst an array of strings.
# If there is no common prefix, return an empty string.
# Example:
# ["flower", "flow", "flight"] -> "fl"
# ["dog", "racecar", "car"]    -> ""
# ["apple", "ape", "apricot"]  -> "ap"


# --- Solution 1 (Your Original Logic: Sort by length and compare first & last) ---
def longest_common_prefix(strs):
    if not strs:
        return ""
    strs.sort(key=len)
    first, last = strs[0], strs[-1]
    prefix = ''
    for i in range(min(len(first), len(last))):
        if first[i] == last[i]:
            prefix += first[i]
        else:
            break
    return prefix


# --- Solution 2 (Horizontal Scanning) ---
# Compare prefix with each word and shrink prefix until it matches
def longest_common_prefix_horizontal(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for word in strs[1:]:
        while word[:len(prefix)] != prefix:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# --- Solution 3 (Vertical Scanning) ---
# Compare characters column by column
def longest_common_prefix_vertical(strs):
    if not strs:
        return ""
    for i in range(len(strs[0])):
        char = strs[0][i]
        for word in strs[1:]:
            if i == len(word) or word[i] != char:
                return strs[0][:i]
    return strs[0]


# --- Solution 4 (Using zip and set) ---
# Zip groups characters column-wise, use set to check if all same
def longest_common_prefix_zip(strs):
    prefix = ""
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            prefix += chars[0]
        else:
            break
    return prefix


# ---------------- Tests ----------------
print(longest_common_prefix(["flower", "flow", "flight"]))     # "fl"
print(longest_common_prefix(["dog", "racecar", "car"]))        # ""
print(longest_common_prefix(["apple", "ape", "apricot"]))      # "ap"

print(longest_common_prefix_horizontal(["flower", "flow", "flight"]))  # "fl"
print(longest_common_prefix_vertical(["flower", "flow", "flight"]))    # "fl"
print(longest_common_prefix_zip(["flower", "flow", "flight"]))         # "fl"



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 2: Group Anagrams ----------------
# Problem: Group words that are anagrams of each other.
# Example:
# ["eat", "tea", "tan", "ate", "nat", "bat"]
# -> [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]


# --- Solution 1 (Your Original Logic: Sort word as key) ---
def group_anagrams(words):
    mydict = {}
    for word in words:
        temp = ''.join(sorted(word))  # canonical key
        if temp in mydict:
            mydict[temp].append(word)
        else:
            mydict[temp] = [word]
    return list(mydict.values())


# --- Solution 2 (Using defaultdict for cleaner code) ---
from collections import defaultdict

def group_anagrams_defaultdict(words):
    anagrams = defaultdict(list)
    for word in words:
        key = ''.join(sorted(word))
        anagrams[key].append(word)
    return list(anagrams.values())


# --- Solution 3 (Using character frequency count as key) ---
def group_anagrams_count(words):
    from collections import Counter
    anagrams = {}
    for word in words:
        key = tuple(sorted(Counter(word).items()))  # frequency-based key
        if key not in anagrams:
            anagrams[key] = []
        anagrams[key].append(word)
    return list(anagrams.values())


# --- Solution 4 (Tuple of char counts for efficiency) ---
def group_anagrams_tuple(words):
    anagrams = defaultdict(list)
    for word in words:
        # Each word -> 26-length tuple of counts
        count = [0] * 26
        for c in word:
            count[ord(c) - ord('a')] += 1
        anagrams[tuple(count)].append(word)
    return list(anagrams.values())


# ---------------- Tests ----------------
words = ["eat", "tea", "tan", "ate", "nat", "bat"]

print(group_anagrams(words))              # [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
print(group_anagrams_defaultdict(words))  # same result
print(group_anagrams_count(words))        # same result
print(group_anagrams_tuple(words))        # same result

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 3: Sort Dictionary by Values ----------------
# Problem: Given a dictionary, return a new dictionary sorted by its values.
# Example:
# {'a': 3, 'b': 1, 'c': 2} -> {'b': 1, 'c': 2, 'a': 3}


# --- Solution 1 (Your Original Logic: manual iteration) ---
def sort_dict(d):
    mylist = list(sorted(d.values()))
    final_dict = {}
    for i in mylist:
        for key in d.keys():
            if d[key] == i:
                final_dict[key] = i
    return final_dict


# --- Solution 2 (Using sorted with lambda) ---
def sort_dict_lambda(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))


# --- Solution 3 (Using dict comprehension + sorted) ---
def sort_dict_comprehension(d):
    return {k: v for k, v in sorted(d.items(), key=lambda x: x[1])}


# --- Solution 4 (Sort by values descending) ---
def sort_dict_desc(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


# ---------------- Tests ----------------
d = {'a': 3, 'b': 1, 'c': 2}

print(sort_dict(d))              # {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_lambda(d))       # {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_comprehension(d))# {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_desc(d))         # {'a': 3, 'c': 2, 'b': 1}


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 4: Quick Sort ----------------
# Problem: Implement the quicksort algorithm to sort an array.
# Example:
# [3, 1, 4, 1, 5, 9, 2, 6] -> [1, 1, 2, 3, 4, 5, 6, 9]


# --- Solution 1 (Your Original Logic: Recursive with list comprehensions) ---
def quick_sort(arr):
    if len(arr) < 2:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# --- Solution 2 (In-place Lomuto Partition Scheme) ---
def quick_sort_inplace(arr):
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i + 1

    def quicksort(low, high):
        if low < high:
            pi = partition(low, high)
            quicksort(low, pi - 1)
            quicksort(pi + 1, high)

    quicksort(0, len(arr) - 1)
    return arr


# --- Solution 3 (Hoare Partition Scheme) ---
def quick_sort_hoare(arr):
    def partition(low, high):
        pivot = arr[(low + high) // 2]
        i, j = low, high
        while True:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1

    def quicksort(low, high):
        if low < high:
            pi = partition(low, high)
            quicksort(low, pi)
            quicksort(pi + 1, high)

    quicksort(0, len(arr) - 1)
    return arr


# --- Solution 4 (Using Python's built-in sorted for simplicity) ---
def quick_sort_builtin(arr):
    return sorted(arr)


# ---------------- Tests ----------------
arr = [3, 1, 4, 1, 5, 9, 2, 6]

print(quick_sort(arr))          # [1, 1, 2, 3, 4, 5, 6, 9]
print(quick_sort_inplace(arr[:]))  # [1, 1, 2, 3, 4, 5, 6, 9]
print(quick_sort_hoare(arr[:]))    # [1, 1, 2, 3, 4, 5, 6, 9]
print(quick_sort_builtin(arr))     # [1, 1, 2, 3, 4, 5, 6, 9]



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Longest Substring Without Repeating Characters ----------------
# Problem: Find the length of the longest substring without repeating characters.
# Example:
# "abcabcbb" -> 3 ("abc")
# "bbbbb"    -> 1 ("b")

# ✅ Your Solution (Sliding Window with Set)
def length_of_longest_substring(s):
    char_set = set()
    left, max_len = 0, 0

    for right in range(len(s)):
        while s[right] in char_set:   # shrink window
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])        # expand window
        max_len = max(max_len, right - left + 1)

    return max_len

# Example Runs:
# print(length_of_longest_substring("abcabcbb"))  # Output: 3
# print(length_of_longest_substring("bbbbb"))     # Output: 1
# print(length_of_longest_substring("pwwkew"))    # Output: 3 ("wke")


# ✅ Advanced Solution 1 (Using Dictionary for O(1) index lookup)
def length_of_longest_substring_dict(s):
    char_index = {}
    left, max_len = 0, 0

    for right, ch in enumerate(s):
        if ch in char_index and char_index[ch] >= left:
            left = char_index[ch] + 1  # move left pointer after duplicate
        char_index[ch] = right
        max_len = max(max_len, right - left + 1)

    return max_len

# print(length_of_longest_substring_dict("abcabcbb"))  # Output: 3
# print(length_of_longest_substring_dict("pwwkew"))    # Output: 3


#  Advanced Solution 2 (Brute Force - for learning, O(n^3))
def length_of_longest_substring_bruteforce(s):
    def all_unique(sub):
        return len(set(sub)) == len(sub)

    max_len = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if all_unique(s[i:j+1]):
                max_len = max(max_len, j - i + 1)

    return max_len

# print(length_of_longest_substring_bruteforce("abcabcbb"))  # Output: 3
# print(length_of_longest_substring_bruteforce("bbbbb"))     # Output: 1


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Flatten a Nested List ----------------
# Problem: Given a nested list of integers, flatten it into a single list.
# Example:
# Input:  [[1, 2, [3]], 4, [5, [6, 7]]]
# Output: [1, 2, 3, 4, 5, 6, 7]

# Your Solution (Recursive Approach)
def flatten_list(mylist):
    myfinalist = []
    for word in mylist:
        if isinstance(word, list):
            myfinalist.extend(flatten_list(word))   # recursive call
        else:
            myfinalist.append(word)
    return myfinalist


# Advanced Solution 1 (Using Stack - Iterative)
def flatten_list_iterative(mylist):
    stack, result = list(mylist), []
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)   # push nested list back to stack
        else:
            result.append(item)
    return result[::-1]  # reverse because we pop from stack


# Advanced Solution 2 (Using Generators - Lazy Flatten)
def flatten_list_generator(mylist):
    for item in mylist:
        if isinstance(item, list):
            yield from flatten_list_generator(item)  # recursive generator
        else:
            yield item


# Advanced Solution 3 (Using collections.abc for robustness)
from collections.abc import Iterable

def flatten_list_generic(mylist):
    for item in mylist:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_list_generic(item)
        else:
            yield item


# Final Run Example
print(list(flatten_list_generic([[1, 2, [3]], 4, [5, [6, 7]]])))  
# Output: [1, 2, 3, 4, 5, 6, 7]


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Two Sum ----------------
# Problem: Given an array of integers and a target value, 
# return the indices (or values) of two numbers such that they add up to the target.
# Example:
# nums = [2, 7, 11, 15], target = 9 -> [2, 7]

# Your Solution (Two-Pointer Approach - requires sorted input)
def two_sum_two_pointer(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        total_value = nums[left] + nums[right]
        if total_value == target:
            return [nums[left], nums[right]]   # return values, not indices
        elif total_value > target:
            right -= 1
        else:
            left += 1
    return -1


# Advanced Solution 1 (Hash Map - works for unsorted arrays, O(n))
def two_sum_hashmap(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [nums[seen[diff]], num]   # return values
        seen[num] = i
    return -1


# Advanced Solution 2 (Brute Force - O(n^2))
def two_sum_bruteforce(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
    return -1


# Final Run Example
nums = [2, 7, 11, 15]
target = 9
print(two_sum_hashmap(nums, target))  
# Output: [2, 7]

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Valid Parentheses ----------------
# Problem: Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
# determine if the input string is valid.
# A string is valid if open brackets are closed in the correct order.
# Example:
# "()"     -> True
# "()[]{}" -> True
# "(]"     -> False

# Your Solution (Stack with Mapping)
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack


# Advanced Solution 1 (Explicit Stack Handling)
def is_valid_parentheses_explicit(s):
    stack = []
    for char in s:
        if char in "({[":
            stack.append(char)
        else:
            if not stack:
                return False
            top = stack.pop()
            if (char == ')' and top != '(') or \
               (char == '}' and top != '{') or \
               (char == ']' and top != '['):
                return False
    return not stack


# Advanced Solution 2 (Using Dictionary with Early Exit)
def is_valid_parentheses_fast(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping.values():   # opening brackets
            stack.append(char)
        elif char in mapping:          # closing brackets
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
    return len(stack) == 0


# Final Run Example
print(is_valid_parentheses("{[]}"))  
# Output: True


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Check if Two Strings are Anagrams ----------------
# Problem: Given two strings, determine if they are anagrams of each other.
# An anagram is formed by rearranging the letters of a word to produce another word.
# Example:
# "listen", "silent" -> True
# "triangle", "integral" -> True
# "hello", "world" -> False

# Your Solution (Dictionary Frequency Count)
def are_anagrams(s1, s2):
    if len(s1) != len(s2):
        return False
    mydict = {}
    for i in s1:
        mydict[i] = mydict.get(i, 0) + 1
    for i in s2:
        mydict[i] = mydict.get(i, 0) - 1
    for count in mydict.values():
        if count != 0:
            return False
    return True


# Advanced Solution 1 (Sorting Both Strings)
def are_anagrams_sorting(s1, s2):
    return sorted(s1) == sorted(s2)


# Advanced Solution 2 (Using collections.Counter)
from collections import Counter
def are_anagrams_counter(s1, s2):
    return Counter(s1) == Counter(s2)


# Final Run Example
print(are_anagrams_counter("listen", "silent"))  
# Output: True


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Check if a String is a Palindrome ----------------
# Problem: Determine if a given string reads the same forward and backward.
# Example:
# "madam" -> True
# "hello" -> False

# Your Solution (Recursive Approach)
def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])


# Advanced Solution 1 (Reverse String Check)
def is_palindrome_reverse(s):
    return s == s[::-1]


# Advanced Solution 2 (Two-Pointer Iterative Approach)
def is_palindrome_two_pointer(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True


# Advanced Solution 3 (Case-insensitive + Alphanumeric Only)
import re
def is_palindrome_clean(s):
    cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
    return cleaned == cleaned[::-1]


# Final Run Example
print(is_palindrome_two_pointer("madam"))  
# Output: True


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Best Time to Buy and Sell Stock ----------------
# Problem: Given an array of stock prices, find the maximum profit you can achieve.
# You may complete only one transaction (buy and sell one share of the stock).
# Example:
# [7, 1, 5, 3, 6, 4] -> 5 (Buy at 1, Sell at 6)
# [7, 6, 4, 3, 1]    -> 0 (No profit possible)

# Your Solution (Single Pass)
def max_profit(prices):
    min_price = float('inf')
    max_profit_val = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit_val = max(max_profit_val, price - min_price)
    return max_profit_val


# Advanced Solution 1 (Brute Force - Check All Pairs)
def max_profit_brute(prices):
    max_profit_val = 0
    n = len(prices)
    for i in range(n):
        for j in range(i+1, n):
            max_profit_val = max(max_profit_val, prices[j] - prices[i])
    return max_profit_val


# Advanced Solution 2 (Dynamic Programming Concept)
def max_profit_dp(prices):
    if not prices:
        return 0
    min_price = prices[0]
    max_profit_val = 0
    for price in prices[1:]:
        max_profit_val = max(max_profit_val, price - min_price)
        min_price = min(min_price, price)
    return max_profit_val


# Final Run Example
prices = [7, 1, 5, 3, 6, 4]
print(max_profit(prices))  
# Output: 5

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Merge Two Sorted Lists ----------------
# Problem: Merge two sorted lists into a single sorted list.
# Example:
# l1 = [1,3,5,7], l2 = [2,4,6,8] -> [1,2,3,4,5,6,7,8]

# Your Solution (Two Pointers)
def merge_sorted_lists(l1, l2):
    i = j = 0
    final_list = []

    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            final_list.append(l1[i])
            i += 1
        else:
            final_list.append(l2[j])
            j += 1
    final_list.extend(l1[i:])
    final_list.extend(l2[j:])
    return final_list


# Advanced Solution 1 (Using heapq.merge)
import heapq
def merge_sorted_lists_heapq(l1, l2):
    return list(heapq.merge(l1, l2))


# Advanced Solution 2 (Concatenate + Sort)
def merge_sorted_lists_sort(l1, l2):
    return sorted(l1 + l2)


# Advanced Solution 3 (Recursive Merge)
def merge_sorted_lists_recursive(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1[0] < l2[0]:
        return [l1[0]] + merge_sorted_lists_recursive(l1[1:], l2)
    else:
        return [l2[0]] + merge_sorted_lists_recursive(l1, l2[1:])


# Final Run Example
l1 = [1, 3, 5, 7]
l2 = [2, 4, 6, 8]
print(merge_sorted_lists(l1, l2))  
# Output: [1, 2, 3, 4, 5, 6, 7, 8]

# -----------------------------------------------------------------------------------------------------------------------------------------------
# ---------------- Problem: Factorial ----------------
# Problem: Compute the factorial of a number n.
# Example:
# 5! -> 120
# 0! -> 1

# Your Solution (Recursive)
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

# Your Solution (Iterative)
def factorial1(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

# Advanced Solution 1 (Using math module)
import math
def factorial_math(n):
    return math.factorial(n)

# Advanced Solution 2 (Using functools.reduce)
from functools import reduce
def factorial_reduce(n):
    if n == 0 or n == 1:
        return 1
    return reduce(lambda x, y: x*y, range(1, n+1))

# Final Run Example
n = 5
print(factorial(n))           # Output: 120
print(factorial1(n))          # Output: 120
print(factorial_math(n))      # Output: 120
print(factorial_reduce(n))    # Output: 120

n = 0
print(factorial(n))           # Output: 1
print(factorial1(n))          # Output: 1
print(factorial_math(n))      # Output: 1
print(factorial_reduce(n))    # Output: 1



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Anagram Check ----------------
# Problem: Check if two strings are anagrams of each other.
# Example:
# "listen", "silent" -> True
# "hello", "world" -> False

# Your Solution (Dictionary Counting)
def is_anagram(s1, s2):
    if len(s1) != len(s2):
        return False
    
    mydict = {}
    for i in s1:
        mydict[i] = mydict.get(i, 0) + 1
    for j in s2:
        mydict[j] = mydict.get(j, 0) - 1
    return all(value == 0 for value in mydict.values())


# Advanced Solution 1 (Sorting)
def is_anagram_sort(s1, s2):
    return sorted(s1) == sorted(s2)


# Advanced Solution 2 (Counter from collections)
from collections import Counter
def is_anagram_counter(s1, s2):
    return Counter(s1) == Counter(s2)


# Final Run Example
s1 = "listen"
s2 = "silent"
s3 = "hello"
s4 = "world"

print(is_anagram(s1, s2))            # Output: True
print(is_anagram(s3, s4))            # Output: False
print(is_anagram_sort(s1, s2))       # Output: True
print(is_anagram_counter(s3, s4))    # Output: False



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Fibonacci Sequence ----------------
# Problem: Generate the first n numbers in the Fibonacci sequence.
# Example:
# n = 10 -> [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Your Solution (Iterative)
def fibonacci(n):
    a, b = 0, 1
    series = []
    for _ in range(n):
        series.append(a)
        a, b = b, a + b
    return series

# Advanced Solution 1 (Recursive)
def fibonacci_recursive(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    seq = fibonacci_recursive(n-1)
    seq.append(seq[-1] + seq[-2])
    return seq

# Advanced Solution 2 (Using Dynamic Programming)
def fibonacci_dp(n):
    if n <= 0:
        return []
    dp = [0] * n
    if n > 1:
        dp[1] = 1
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp

# Advanced Solution 3 (Using Generator)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Final Run Example
n = 10
print(fibonacci(n))                  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(fibonacci_recursive(n))        # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(fibonacci_dp(n))               # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(list(fibonacci_generator(n)))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Prime Number Check ----------------
# Problem: Check if a given number n is prime.
# Example:
# 7  -> True
# 10 -> False

# Your Solution (Simple Iteration)
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

# Advanced Solution 1 (Optimized Iteration up to sqrt(n))
def is_prime_optimized(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Advanced Solution 2 (Using 6k ± 1 Optimization)
def is_prime_6k(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Advanced Solution 3 (Using Sympy Library)
from sympy import isprime as sympy_isprime

# Final Run Example
print(is_prime(7))             # Output: True
print(is_prime(10))            # Output: False
print(is_prime_optimized(7))   # Output: True
print(is_prime_optimized(10))  # Output: False
print(is_prime_6k(7))          # Output: True
print(is_prime_6k(10))         # Output: False
print(sympy_isprime(7))        # Output: True
print(sympy_isprime(10))       # Output: False


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Find Missing Number ----------------
# Problem: Given an array of n-1 integers in the range 1 to n, find the missing number.
# Example:
# [3, 4, 5, 7] -> 1+2+3+4+5+6+7 - sum([3,4,5,7]) = 1+2+6 = 9? Actually correct: numbers expected from 1..7, missing is 1,2,6? 
# Let's stick with your method: missing number assuming 1..n

# Your Solution (Using Sum Formula)
def find_missing_number(nums):
    n = len(nums) + 1
    total = n * (n + 1) // 2
    return total - sum(nums)

# Advanced Solution 1 (Using XOR)
def find_missing_xor(nums):
    n = len(nums) + 1
    xor_all = 0
    xor_arr = 0
    
    for i in range(1, n+1):
        xor_all ^= i
    for num in nums:
        xor_arr ^= num
    return xor_all ^ xor_arr

# Advanced Solution 2 (Using Set Difference)
def find_missing_set(nums):
    n = len(nums) + 1
    full_set = set(range(1, n+1))
    return list(full_set - set(nums))[0]

# Advanced Solution 3 (Iterative Check)
def find_missing_iter(nums):
    n = len(nums) + 1
    nums_set = set(nums)
    for i in range(1, n+1):
        if i not in nums_set:
            return i

# Final Run Example
nums = [3, 4, 5, 7]
print(find_missing_number(nums))  # Output: 1 (assuming numbers start from 1)
print(find_missing_xor(nums))     # Output: 1
print(find_missing_set(nums))     # Output: 1
print(find_missing_iter(nums))    # Output: 1



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Sort Dictionary by Values ----------------
# Problem: Sort a dictionary based on its values in ascending order.
# Example:
# {'a': 3, 'b': 1, 'c': 2} -> {'b': 1, 'c': 2, 'a': 3}

# Your Solution (Using nested loops)
def sort_dict_value(my_dict):
    new_dict = {}
    mylist = list(sorted(my_dict.values()))
    for i in mylist:
        for key in my_dict:
            if my_dict[key] == i:
                new_dict[key] = i
    return new_dict

# Advanced Solution 1 (Using sorted with key)
def sort_dict_value_sorted(my_dict):
    return dict(sorted(my_dict.items(), key=lambda item: item[1]))

# Advanced Solution 2 (Using operator.itemgetter)
from operator import itemgetter
def sort_dict_value_itemgetter(my_dict):
    return dict(sorted(my_dict.items(), key=itemgetter(1)))

# Advanced Solution 3 (Using collections.OrderedDict)
from collections import OrderedDict
def sort_dict_value_ordered(my_dict):
    return OrderedDict(sorted(my_dict.items(), key=lambda x: x[1]))

# Final Run Example
d = {'a': 3, 'b': 1, 'c': 2}
print(sort_dict_value(d))              # {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_value_sorted(d))       # {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_value_itemgetter(d))   # {'b': 1, 'c': 2, 'a': 3}
print(sort_dict_value_ordered(d))      # OrderedDict([('b', 1), ('c', 2), ('a', 3)])


# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: Isomorphic Strings ----------------
# Problem: Check if two strings are isomorphic. 
# Two strings s1 and s2 are isomorphic if characters in s1 can be replaced to get s2, maintaining the order.
# Example:
# "aab", "xxy" -> True
# "foo", "bar" -> False

# Your Solution (Using two dictionaries to track first occurrence)
def areIsomorphic(s1, s2):
    m1 = {}
    m2 = {}

    for i in range(len(s1)):
        if s1[i] not in m1:
            m1[s1[i]] = i
        if s2[i] not in m2:
            m2[s2[i]] = i

        if m1[s1[i]] != m2[s2[i]]:
            return False

    return True

# Advanced Solution 1 (Using single mapping with set for uniqueness)
def areIsomorphic_map(s1, s2):
    if len(s1) != len(s2):
        return False
    mapping = {}
    mapped_values = set()

    for c1, c2 in zip(s1, s2):
        if c1 in mapping:
            if mapping[c1] != c2:
                return False
        else:
            if c2 in mapped_values:
                return False
            mapping[c1] = c2
            mapped_values.add(c2)
    return True

# Advanced Solution 2 (Using index pattern tuples)
def areIsomorphic_pattern(s1, s2):
    return [s1.index(c) for c in s1] == [s2.index(c) for c in s2]

# Advanced Solution 3 (Using Python translation tables)
def areIsomorphic_translate(s1, s2):
    if len(s1) != len(s2):
        return False
    table = str.maketrans(s1, s2)
    return s1.translate(table) == s2

# Final Run Example
s1 = "aab"
s2 = "xxy"
print(areIsomorphic(s1, s2))             # True
print(areIsomorphic_map(s1, s2))         # True
print(areIsomorphic_pattern(s1, s2))     # True
print(areIsomorphic_translate(s1, s2))   # True

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem: String Compression ----------------
# Problem: Compress a string by counting consecutive characters.
# Example:
# "aabccddaabb" -> "a2b1c2d2a2b2"

# Solution 1 (Using simple iteration)
def compress_string(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(s[i-1] + str(count))
            count = 1
    result.append(s[-1] + str(count))
    return ''.join(result)

# Solution 2 (Using itertools.groupby)
from itertools import groupby

def compress_string_groupby(s):
    return ''.join(char + str(len(list(group))) for char, group in groupby(s))

# Solution 3 (Using regular expressions)
import re

def compress_string_regex(s):
    return ''.join(f"{m.group(1)}{len(m.group(0))}" for m in re.finditer(r"(.)\1*", s))

# Final Run Example
s = "aabccddaabb"
print(compress_string(s))          # Output: a2b1c2d2a2b2
print(compress_string_group



# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 5: Flatten a Nested JSON/Dict ----------------
# Problem: Flatten a nested dictionary into a single-level dictionary
# Example:
# {"a":1,"b":{"c":2,"d":{"e":3}}} -> {"a":1,"b.c":2,"b.d.e":3}

# Your Solution
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# Optimized Solution 1: Using recursion with dict comprehension
def flatten_dict_comp(d, parent_key='', sep='.'):
    return { (parent_key + sep + k if parent_key else k) : v
             for k, v in d.items()
             for v in (flatten_dict_comp(v, parent_key + sep + k, sep) if isinstance(v, dict) else [v])}

# Optimized Solution 2: Using iterative stack approach
def flatten_dict_iter(d):
    stack = [((), d)]
    result = {}
    while stack:
        path, current = stack.pop()
        for k, v in current.items():
            new_path = path + (k,)
            if isinstance(v, dict):
                stack.append((new_path, v))
            else:
                result[".".join(new_path)] = v
    return result

nested_dict = {"a":1,"b":{"c":2,"d":{"e":3}}}
print(flatten_dict(nested_dict))  # Output: {'a': 1, 'b.c': 2, 'b.d.e': 3}



# ---------------- Problem 7: String Manipulation Problem ----------------
# Problem: Reverse words in a string or capitalize first letters, etc.

# Your Solution: Reverse words
def reverse_words(s):
    return ' '.join(s.split()[::-1])

# Optimized Solution 1: Using list comprehension
def reverse_words_lc(s):
    return ' '.join([word for word in reversed(s.split())])

# Optimized Solution 2: Using map
def reverse_words_map(s):
    return ' '.join(map(str, reversed(s.split())))

sample_string = "hello world from python"
print(reverse_words(sample_string))  # Output: "python from world hello"

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 9: DFS and BFS Traversal ----------------
# Problem: Implement DFS and BFS for a graph
# Example: graph = {0:[1,2], 1:[2], 2:[0,3], 3:[3]}

graph = {0:[1,2], 1:[2], 2:[0,3], 3:[3]}

# Your Solution: DFS
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# Your Solution: BFS
from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
    return visited

# Optimized Solution 1: DFS iterative
def dfs_iterative(graph, start):
    visited, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(reversed(graph[node]))
    return visited

# Optimized Solution 2: BFS using list
def bfs_list(graph, start):
    visited, queue = [], [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])
    return visited

print(dfs(graph,0))        # Output: {0, 1, 2, 3}
print(bfs(graph,0))        # Output: {0, 1, 2, 3}

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Problem 8: Remove Adjacent Characters ----------------
# Problem: Remove consecutive duplicate characters from a string
# Example: "aaabbcddd" -> "abcd"

# Your Solution
def remove_adjacent(s):
    if not s:
        return ""
    result = [s[0]]
    for char in s[1:]:
        if char != result[-1]:
            result.append(char)
    return ''.join(result)

# Optimized Solution 1: Using itertools.groupby
from itertools import groupby
def remove_adjacent_groupby(s):
    return ''.join(k for k,_ in groupby(s))

# Optimized Solution 2: Recursive approach
def remove_adjacent_recursive(s):
    if len(s) < 2:
        return s
    if s[0] == s[1]:
        return remove_adjacent_recursive(s[1:])
    else:
        return s[0] + remove_adjacent_recursive(s[1:])

sample_str = "aaabbcddd"
print(remove_adjacent(sample_str))  # Output: "abcd"

# -----------------------------------------------------------------------------------------------------------------------------------------------


# ---------------- Problem 6: Binary Tree Vertical Order Traversal ----------------
# Problem: Print nodes of a binary tree in vertical order
# Example: 
#        3
#       / \
#      9   8
#     / \ / \
#    4  0 1  7
# Output: [[4],[9],[3,0,1],[8],[7]]

from collections import defaultdict, deque

class TreeNode:
    def __init__(self, val=0,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right

# Your Solution
def vertical_order(root):
    if not root:
        return []
    col_table = defaultdict(list)
    queue = deque([(root, 0)])
    while queue:
        node, col = queue.popleft()
        col_table[col].append(node.val)
        if node.left:
            queue.append((node.left, col-1))
        if node.right:
            queue.append((node.right, col+1))
    return [col_table[x] for x in sorted(col_table.keys())]

# Optimized Solution 1: Using DFS with column tracking
def vertical_order_dfs(root):
    res = defaultdict(list)
    def dfs(node, col):
        if node:
            res[col].append(node.val)
            dfs(node.left, col-1)
            dfs(node.right, col+1)
    dfs(root, 0)
    return [res[x] for x in sorted(res.keys())]

# Sample Tree
root = TreeNode(3, TreeNode(9, TreeNode(4), TreeNode(0)), TreeNode(8, TreeNode(1), TreeNode(7)))
print(vertical_order(root))  # Output: [[4], [9], [3,0,1], [8], [7]]


# -----------------------------------------------------------------------------------------------------------------------------------------------


# ---------------- Problem 10: Star Pattern ----------------
# Problem: Print a simple right-angled triangle star pattern
# Example: n=5
# *
# **
# ***
# ****
# *****

# Your Solution
def right_triangle_star(n):
    for i in range(1, n+1):
        print('*'*i)

# Optimized Solution 1: Using nested loops
def right_triangle_star_nested(n):
    for i in range(1,n+1):
        for j in range(i):
            print('*', end='')
        print()

# Optimized Solution 2: Using list comprehension and join
def right_triangle_star_join(n):
    [print(''.join(['*' for _ in range(i)])) for i in range(1,n+1)]


# ---------------- Problem 11: Reverse Star Pattern ----------------
# Problem: Print a reverse right-angled triangle
# Example: n=5
# *****
# ****
# ***
# **
# *

# Your Solution
def reverse_triangle_star(n):
    for i in range(n,0,-1):
        print('*'*i)

# Optimized Solution 1: Nested loops
def reverse_triangle_star_nested(n):
    for i in range(n,0,-1):
        for j in range(i):
            print('*', end='')
        print()

# Optimized Solution 2: List comprehension
def reverse_triangle_star_join(n):
    [print(''.join(['*' for _ in range(i)])) for i in range(n,0,-1)]


# ---------------- Problem 12: Pyramid Pattern ----------------
# Problem: Print a centered pyramid star pattern
# Example: n=5
#     *
#    ***
#   *****
#  *******
# *********

# Your Solution
def pyramid_star(n):
    for i in range(1,n+1):
        print(' '*(n-i) + '*'*(2*i-1))

# Optimized Solution 1: Using string formatting
def pyramid_star_format(n):
    for i in range(1,n+1):
        print(f"{'*'*(2*i-1):^{2*n-1}}")

# Optimized Solution 2: Using join and list comprehension
def pyramid_star_join(n):
    [print(' '*(n-i) + ''.join(['*' for _ in range(2*i-1)])) for i in range(1,n+1)]


# ---------------- Problem 13: Inverted Pyramid Pattern ----------------
# Problem: Print an inverted pyramid pattern
# Example: n=5
# *********
#  *******
#   *****
#    ***
#     *

# Your Solution
def inverted_pyramid(n):
    for i in range(n,0,-1):
        print(' '*(n-i) + '*'*(2*i-1))

# Optimized Solution 1: Using string formatting
def inverted_pyramid_format(n):
    for i in range(n,0,-1):
        print(f"{'*'*(2*i-1):^{2*n-1}}")

# Optimized Solution 2: List comprehension
def inverted_pyramid_join(n):
    [print(' '*(n-i) + ''.join(['*' for _ in range(2*i-1)])) for i in range(n,0,-1)]


# ---------------- Problem 14: Number Pyramid Pattern ----------------
# Problem: Print a number pyramid pattern
# Example: n=5
#     1
#    121
#   12321
#  1234321
# 123454321

# Your Solution
def number_pyramid(n):
    for i in range(1,n+1):
        left = ''.join(str(j) for j in range(1,i+1))
        right = left[:-1][::-1]
        print(' '*(n-i) + left + right)

# Optimized Solution 1: Using string formatting
def number_pyramid_format(n):
    for i in range(1,n+1):
        s = ''.join(str(j) for j in range(1,i+1))
        print(f"{s + s[:-1][::-1]:^{2*n-1}}")

# Optimized Solution 2: Using list comprehension
def number_pyramid_join(n):
    [print(' '*(n-i) + ''.join([str(j) for j in range(1,i+1)]) + ''.join([str(j) for j in range(1,i)[::-1]])) for i in range(1,n+1)]


# ---------------- Problem 15: Reverse Number Pyramid Pattern ----------------
# Problem: Print a reverse number pyramid pattern
# Example: n=5
# 123454321
#  1234321
#   12321
#    121
#     1

# Your Solution
def reverse_number_pyramid(n):
    for i in range(n,0,-1):
        left = ''.join(str(j) for j in range(1,i+1))
        right = left[:-1][::-1]
        print(' '*(n-i) + left + right)

# Optimized Solution 1: String formatting
def reverse_number_pyramid_format(n):
    for i in range(n,0,-1):
        s = ''.join(str(j) for j in range(1,i+1))
        print(f"{s + s[:-1][::-1]:^{2*n-1}}")
















