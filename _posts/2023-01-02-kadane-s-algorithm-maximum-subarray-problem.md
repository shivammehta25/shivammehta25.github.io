---
layout: post
title: Kadane's algorithm - Maximum subarray problem
date: 2019-12-09 16:19 +0100
categories: [programming]
tags: [python, data-structure, algorithm, competitive-programming, complexity-analysis]
image:
  path: thumbnail.jpeg
  alt: Kadane's algorithm - Maximum subarray problem
math: true
---

## (Migrated from my old blog)

Okay so, I was brushing up my Algorithm skills at CodeSignal and I found this maxSubArrayProblem: [https://app.codesignal.com/challenge/LrAwpTnYZR6NMCbfs](https://app.codesignal.com/challenge/LrAwpTnYZR6NMCbfs)

So we will solve this naive and Dynamic Programming Approach also known as Kadane's Algorithm.

Please try this problem once yourself then let's talk about the solution.

First, Lets understand the task.

Task: Given an array of integers `inputArray`, find the contiguous subarray which has the maximum sum. Return that sum.

Here Contiguous subarray means that the subarray should be continuous and have no break in between.

Lets assume the array that we have to find is $$inputArray = [-1, 7, -2, 3, 4, 0, 1, -1] $$ the answer that we have to find is 13 from the subarray $ [7, -2, 3, 4, 0, 1] $

Lets start like we always do finding a BruteForce solution to this:

A bruteforce way will be to take the first element and then start from backward and try to find the sum of all possible option:

```text
max_sum := 0
Loop i:0 .. n:
    First_element := a[i]
    Loop j: n -1 .. i + 1:
         max_sum = Maximum( max_sum, sum(FirstElement + a[j.. n])
    end Loop j
End Loop i
```

This is a highly ineffective solution to this problem $ O(n^3) $ (Never do it please) First i'th loop, second j'th loop and the third loop to get the sum. The code for this in Python can be like this.

```python
def maxSubarray(inputArray):
    N = len(inputArray)
    max_sum = 0
    for i in range(N):
        for j in range(N-1, i, -1):
            max_sum = max(max_sum, sum([inputArray[i]] + inputArray[i+1:j]))

    return max_sum
```

So a Better an $ O(n) $ solution to this problem will be **Kadane's Algorithm**. How the algorithm work is that we will find the maximum positive contiguous subarrays. The moment our sum will touch 0 we will dismiss all and start with the number that is positive and keep the value saved until we again find the maximum value.

Let's try writing an algorithm for it

```text
current_sum := 0
max_sum := 0
Loop i: 0 .. n:
    current_sum := current_sum + a[i]
    IF current_sum < 0 THEN
         current_sum = 0
    ELSE
         IF max_sum < current_sum THEN
              current_sum := max_sum
```

The python implementation for this can be written like this: I have written two extra variables start pointer and end pointer to get the value of start and end pointer if needed.

```python
def maxSubarray(inputArray):
    max_current = 0
    max_ = 0
    start_pointer, end_pointer = 0, 0
    for index, i in enumerate(inputArray):
        max_current += i
        if max_current < 0:
            max_current = 0
            start_pointer = index + 1
        else:
            if max_ > max_current:
                continue
            else:
                max_ = max_current
                end_pointer = index


    return max_
```

A more Pythonic way it can be written as

```python
def maxSubarray(inputArray):
    max_sum = 0
    current_sum = 0
    for i in range(len(inputArray)):
        current_sum = max(0, current_sum+ inputArray[i])
        max_sum = max(current_sum, max_sum)

    return max_sum
```

Here the Dynamic Programming's sub problem is that the maximum subarray ending at this position.

Reference and Header image from: [https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/](https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/)
