---
layout: post
title: An idea to test programming solution
date: 2020-01-31 16:41 +0100
categories: [programming]
tags: [python, data-structure, algorithm, competitive-programming, complexity-analysis, leetcode]
image:
  path: thumbnail.jpeg
  alt: An idea to test programming solution
math: true
---

## New Edit

I have moved from emacs to VSCode + Vim key bindings IDE wise, but the idea is still the same and useful.

## (Migrated from my old blog)

Alright! Today I just had a cool idea while trying to solve something :) To explain it, We will be solving this problem [https://leetcode.com/problems/first-missing-positive](https://leetcode.com/problems/first-missing-positive) i.e First Missing Positive Problem, plus we will see discuss its solutions and a way to test the solution before it fails on the competitive platform.

Okay, let talk algorithm so first try will be brute forcing, we will take this as the test input $ [3,4,-1,1] $.

## 1. Brute Force Solution

So one solution will be that you take the maximum element of the array and create a cache array initialize it with zero, where if the positive element exists will have a value of 1, so in this case, your max becomes 4 and your cache array is filled like : $ [1, 0, 1, 1] $ since 2 is not present the index of 2 is zero and thus that is your first missing positive integer. The code for the same will be:

```python
class Solution:
    def firstMissingPositive(self, arr):
        if not arr:
            return 1
        m = max(arr)
        if m < 0:
            return 1
        cache = [0 for _ in range(m + 2)]
        for a in arr:
            if a >= 0:
                cache[a] = 1
        for i, v in enumerate(cache):
            if i == 0:
                continue
            if v == 0:
                return i

# Driver code to make your code run like LeetCode
if __name__ == "__main__":
    arr = list(map(int, input().split()))
    sol = Solution()
    print(sol.firstMissingPositive(arr))
```

This might look like a linear solution, and it is but the space complexity will be $ O(M) $ where M is the maximum number present in the array, So if your test case looks like $ [29000000] $ for one element it will create a cache of 29000000 space and maybe if you have more element, because of Memory Thrashing it will throw TLE and it did check my TLE solution here: [https://leetcode.com/submissions/detail/298567793/](https://leetcode.com/submissions/detail/298567793/)

## 2. Little Better Solution

So the next thought that came into my mind was still $ O(N \log{N}) $ but the solution had no memory thrashing and it might not work if the testing element size i.e $ N $ is very very large. $ O(N \log{N}) $ is still considered a good algorithm.

The idea was that I will sort it and remove positive elements and voila the first missing element will be my first missing positive number. The code looks like

```python
class Solution:

    def firstMissingPositive(self, arr):
        arr = [a for a in sorted(set(arr)) if a > 0]
        for i in range(1, len(arr)+1):
           # print(arr[i-1], i)
            if arr[i-1] != i:
                return i
        return len(arr) + 1



# Driver code to run like LeetCode
if __name__ == "__main__":
    arr = list(map(int, input().split()))
    sol = Solution()
    print(sol.firstMissingPositive(arr))
```

And it did get accepted [https://leetcode.com/submissions/detail/298885591/](https://leetcode.com/submissions/detail/298885591/) and with no extra memory. But it could be improved more.

## 3. O(N) Solution

So the idea behind is something similar to a problem where you have to find the first positive number or first non-reappearing number. We will first separate the negative elements to positive elements and then use the element and put a negative sign on the element present on that index i.e in this case where our elements are $ [3,4,-1,1] $ our segregated array will be $ [4,3,1] $.

Now we will iterate this array.

- 4 $ \rightarrow $ We first encounter 4 since if we will do $ a[4 - 1] $ it will be index out of bound it will be skipped
- 3 $ \rightarrow $ We will negate the value of $ a[3-1] = a[2] $ so our array becoms $ [4, 3, -1] $
- 1 $ \rightarrow $ We will negate the value of $ a[1-1] = a[0] $ so our array becomes $ [-4, 3, -1] $

Now we will iterate the array again and the first non negative index will be our answer i.e $ 2 $

Code looks like

```python
class Solution:
    def segregate_positives(self, a):
        j = 0
        for i in range(len(a)):
            if a[i] <= 0:
                a[i], a[j] = a[j], a[i]
                j += 1
        return a[j:]


    def firstMissingPositive(self, arr):
        arr = self.segregate_positives(arr)
        for i in range(len(arr)):
            if (abs(arr[i]) - 1) < len(arr):
                arr[abs(arr[i]) - 1] = - abs(arr[abs(arr[i]) - 1])

        for i in range(len(arr)):
            if arr[i] > 0:
                return i + 1

        return len(arr) + 1

# Driver code to run like LeetCode
if __name__ == "__main__":
    arr = list(map(int, input().split()))
    sol = Solution()
    print(sol.firstMissingPositive(arr))
```

This was the fastest and with lowest memory: [https://leetcode.com/submissions/detail/298578539/](https://leetcode.com/submissions/detail/298578539/).

# Testing

For difficult problems and when there is no time-bound. it is a good idea to write lot of tests and check it.

So we first create a generating program

```python
#!/usr/bin/env python3
import random
import sys

seed = int(sys.argv[1])

a = set()

for i in range(random.randint(1, 1000)):
    a.add(random.randint(-1000, 100))

a = list(a)
random.shuffle(a)
print(' '.join(map(str, a)))
```

Next we put our solution and a working brute force solution in files, I used these two codes and these two file names

```python

class Solution:
    def segregate_positives(self, a):
        j = 0
        for i in range(len(a)):
            if a[i] <= 0:
                a[i], a[j] = a[j], a[i]
                j += 1
        return a[j:]


    def firstMissingPositive(self, arr):
        arr = self.segregate_positives(arr)
        for i in range(len(arr)):
            if (abs(arr[i]) - 1) < len(arr):
                arr[abs(arr[i]) - 1] = - abs(arr[abs(arr[i]) - 1])

        for i in range(len(arr)):
            if arr[i] > 0:
                return i + 1

        return len(arr) + 1

# Driver code to run like LeetCode
if __name__ == "__main__":
    arr = list(map(int, input().split()))
    sol = Solution()
    print(sol.firstMissingPositive(arr))
```

```python
class Solution:

    def firstMissingPositive(self, arr):
        arr = [a for a in sorted(set(arr)) if a > 0]
        for i in range(1, len(arr)+1):
           # print(arr[i-1], i)
            if arr[i-1] != i:
                return i
        return len(arr) + 1



# Driver code to run like LeetCode
if __name__ == "__main__":
    arr = list(map(int, input().split()))
    sol = Solution()
    print(sol.firstMissingPositive(arr))
```

Now my testing module which will take input from the test generator feed it into a file inp, and feed this file to the stdin of both programs and check for similar outputs's if it will be different it will tell the different outputs and show the invalid test case.

```python
import sys
import subprocess
i = 0


filenames = {
    'generator': 'test_generator.py',
    'brute_force': 'firstMissing_bf.py',
    'solution' : 'first_missing_positive.py'
    }


while True:
    print('Test Case: {}'.format(i))
    test_case = subprocess.check_output("python {} {} > inp".format(filenames['generator'], i), shell=True).decode('ascii').strip()
    out1 = subprocess.check_output("python {} < inp".format(filenames['brute_force']), shell=True).decode('ascii').strip()
    out2 = subprocess.check_output("python {} < inp".format(filenames['solution']), shell=True).decode('ascii').strip()
    if out1 != out2:
        print('Output1: {}\nOutput2: {}'.format(out1, out2))
        print('Test Case: {}'.format(test_case))
        print('check file named inp')
        print(subprocess.check_output("cat inp", shell=True).decode('ascii').strip())
        break
    i += 1
```

Running this file you can check where your code fails, in this case, it worked flawlessly but I put a bug intentionally to check it.

![Testing Solution](testing_solution.png)

Testing Solution

and a huge shout out to EMACS for letting me do all of these together like this

![Multibuffer view of EMACS](thumbnail.jpeg)

Multi buffer view of EMACS
