def longest_common_prefix(strs):
    if not strs:
        return ""
    strs.sort(key=len)
    first,last=strs[0],strs[-1]

    prefix = ''
    for i in range(min(len(first),len(last))):
        if first[i] == last[i]:
            prefix += first[i]
        else:
            pass

    return prefix


# print(longest_common_prefix(["flower", "flow", "flight"]))  # Output: "fl" 
# print(longest_common_prefix(["dog", "racecar", "car"]))  # Output: "" 
# print(longest_common_prefix(["apple", "ape", "apricot"]))  # Output: "ap" 

def group_anagrams(words):
    mydict = {}
    for word in words:
        temp = ''.join(sorted(word))
        if temp in mydict:
            mydict[temp].append(word)
        else:
            mydict[temp] = [word]
    return list(mydict.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"] 
print(group_anagrams(words)) 

def sort_dict(d):
    mylist = list(sorted(d.values()))
    final_dict = {}

    for i in mylist:
        for key in d.keys():
            if d[key] == i:
                final_dict[key] = i

    return final_dict
d = {'a': 3, 'b': 1, 'c': 2} 
# print(sort_dict(d)) 

def quick_sort(arr):
    if len(arr) < 2:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

arr = [3, 1, 4, 1, 5, 9, 2, 6] 
# print(quick_sort(arr)) 


def length_of_longest_substring(str1):
    char_set = set()
    left,max_len = 0,0

    for right in range(len(str1)):
        while str1[right] in char_set:
            char_set.remove(str1[left])
            left += 1
        char_set.add(str1[right])
        max_len = max(max_len, right - left+1)
    return max_len

# print(length_of_longest_substring("abcabcbb"))  # Output: 3 
# print(length_of_longest_substring("bbbbb"))    # Output: 1 

def flatten_list(mylist):
    myfinalist = []
    for word in mylist:
        if isinstance(word,list):
            myfinalist.extend(flatten_list(word))
        else:
            myfinalist.append(word)
    return myfinalist
 

# print(flatten_list([[1, 2, [3]], 4, [5, [6, 7]]]))  


def two_sum_two_pointer(nums,target):
    left,right = 0,len(nums)-1
    while left < right:
        total_value = nums[left] + nums[right]
        if total_value == target:
            return [left,right]
        elif total_value > target:
            right -= 1
        else:
            left += 1
    return -1

nums = [2, 7, 11, 15] 
target = 9 
# print(two_sum_two_pointer(nums, target))  # Output: [2, 7] 

def two_sum(nums,target):
    mydict = {}
    for index,value in enumerate(nums):
        diff = target - value
        if diff in mydict:
            return [mydict[diff],index]
        mydict[value] = index
    return []


nums = [2, 7, 11, 15]
target = 9
# print(two_sum(nums,target))


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

# print(is_valid_parentheses("()"))       # Output: True 
# print(is_valid_parentheses("()[]{}"))   # Output: True 
# print(is_valid_parentheses("(]"))       # Output: False 
# print(is_valid_parentheses("([)]"))     # Output: False 
# print(is_valid_parentheses("{[]}"))     # Output: True 

def are_anagrams(s1,s2):
    if len(s1) != len(s2):
        return False
    mydict = {}

    for i in s1:
        mydict[i] = mydict.get(i,0)+1
    
    for i in s2:
        mydict[i] = mydict.get(i,0)-1
    
    for count in mydict.values():
        if count != 0:
            return False

    return True

# print(are_anagrams("\", "silent"))  # Output: True 
# print(are_anagrams("triangle", "integral"))  # Output: True 
# print(are_anagrams("hello", "world"))  # Output: False 
# print(are_anagrams("anagram", "nagaram"))  # Output: True 

def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])

string = "madam" 
# print(f"Is '{string}' a palindrome? {is_palindrome(string)}") 
 
string = "hello" 
# print(f"Is '{string}' a palindrome? {is_palindrome(string)}")

def max_profit(prices):
    min_profit = float('inf')
    maxs_profit = 0

    for price in prices:
        min_profit = min(min_profit,price)
        maxs_profit = max(maxs_profit,price-min_profit)
    return maxs_profit

prices = [7, 1, 5, 3, 6, 4] 
# print(max_profit(prices))  # Output: 5 

def merge_sorted_lists(l1,l2):
    i=j=0
    final_list = []

    while i<len(l1) and j<len(l2):
        if l1[i] < l2[j]:
            final_list.append(l1[i])
            i += 1
        else:
            final_list.append(l2[j])
            j += 1
    final_list.extend(l1[i:])
    final_list.extend(l2[j:])
    return final_list

l1 = [1, 3, 5, 7] 
l2 = [2, 4, 6, 8] 
# print(merge_sorted_lists(l1, l2)) 

def is_valid(s):
    stack,m=[], {'}':'{',')':'(',']':'['}

    for c in s:
        if c in m:
            if not stack or stack.pop() != m[c]:
                return False
        else:
            stack.append(c)
    return not stack


# print(is_valid("()[]{}"))  # True 
# print(is_valid("([)]"))    # False 
# print(is_valid("{[]}"))   #True 


def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])

# print(is_palindrome("madam"))  # Output: True 
# print(is_palindrome("hello"))  # Output: False 


def is_anagram(s1,s2):
    if len(s1) != len(s2):
        return False
    
    mydict = {}

    for i in s1:
        mydict[i] = mydict.get(i,0)+1
    for j in s2:
        mydict[j] = mydict.get(j,0)-1
    return all(value == 0 for value in mydict.values())

# print(is_anagram("listen", "silent"))  # Output: True 
# print(is_anagram("hello", "world"))    # Output: False 

def isAnagram(s1,s2):
    sorted_s1 = sorted(s1)
    sorted_s2 = sorted(s2)
    return sorted_s1 == sorted_s2

# print(is_anagram('listen','silent'))
# print(is_anagram('hello','world'))


def groupAnagrams(str1):
    mydict = {}

    for word in str1:
        temp = ''.join(sorted(word))
        if temp in mydict:
            mydict[temp].append(word)
        else:
            mydict[temp] = [word]
    return list(mydict.values())

strs = ["eat", "tea", "tan", "ate", "nat", "bat"] 
# print(groupAnagrams(strs)) 
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']] 


def is_palindrome1(s):
    return s == s[::-1]

# print(is_palindrome1("madam"))  # Output: True 
# print(is_palindrome1("hello"))  # Output: False 

def is_palindrome2(s):
    left,right = 0,len(s)-1
    while left<right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True



# print(is_palindrome2("madam"))  # Output: True 
# print(is_palindrome2("hello"))  # Output: False 


def factorial(n):
    if n==0 or n==1:
        return 1
    return n * factorial(n-1)

# print(factorial(5))  # Output: 120

def factorial1(n):
    result = 1
    for i in range(2,n+1):
        result *= i
    return result


# print(factorial1(5))  # Output: 120 
# print(factorial1(0))  # Output: 1 (by definition) 

def fibonacci(n):
    a,b=0,1
    series = []
    for _ in range(n):
        series.append(a)
        a,b = b, a+b
    return series

# print(fibonacci(10))  
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 

def is_prime(n):
    if n<2:
        return False
    for i in range(2,n):
        if n%i == 0:
            return False
    return True



# print(is_prime(7))   # Output: True 
# print(is_prime(10))  # Output: False 

def merge_sorted_lists(s1,s2):
    result,i,j = [],0,0
    while i<len(s1) and j<len(s2):
        if s1[i] < s2[j]:
            result.append(s1[i])
            i += 1
        else:
            result.append(s2[j])
            j += 1
    result.extend(s1[i:])
    result.extend(s2[j:])

    return result


# print(merge_sorted_lists([1, 3, 5], [2, 4, 6]))  
# Output: [1, 2, 3, 4, 5, 6] 


def find_missing_number(nums):
    n = len(nums) +1
    total = n * (n+1)//2
    return total - sum(nums)

# print(find_missing_number([3,4,5,7]))



def sort_dict_value(my_dict):
    new_dict = {}
    mylist = list(sorted(my_dict.values()))

    for i in mylist:
        for key in my_dict:
            if my_dict[key] == i:
                new_dict[key] = i
    return new_dict
    # print(mylist)

d = {'a': 3, 'b': 1, 'c': 2} 
# print(sort_dict_value(d)) 


def length_of_longest_substring(s):
    char_set = set()
    left,max_len = 0,0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left+1)
    return max_len

# print(length_of_longest_substring("abcabcbb"))  # Output: 3 
# print(length_of_longest_substring("bbbbb"))    # Output: 1 

def areIsomorphic(s1, s2):
    m1 = {}
    m2 = {}

    for i in range(len(s1)):

        if s1[i] not in m1:
            m1[s1[i]] = i
        if s2[i] not in m2:
            m2[s2[i]] = i

        # Check if the first occurrence indices match
        if m1[s1[i]] != m2[s2[i]]:
            return False

    return True

s1 = "aab"
s2 = "xxy"
print(areIsomorphic(s1,s2))