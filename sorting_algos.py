#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: sorting_algos.py
#  Created: 07/05/2018, 17:14
#   Author: Bernie Roesler
#
"""
  Description: Sorting algorithms
"""
#==============================================================================

import numpy as np

#------------------------------------------------------------------------------
#        0. Bubble
#------------------------------------------------------------------------------
def bubble_sort(A):
    """Swap elements so that lesser elements 'float' to the top."""
    while True:
        no_swaps = True
        for i in range(1, len(A)):
            if A[i-1] > A[i]:
                A[i-1], A[i] = A[i], A[i-1] # swap!
                no_swaps = False
        if no_swaps:
            break
    return A    # sorted in place

#------------------------------------------------------------------------------
#        1. Selection
#------------------------------------------------------------------------------
def selection_sort(A):
    """Find minimum element of unsorted subset and append to sorted array."""
    sorted_A = []
    while A != []:
        # Find minimum element in A
        a_min = A[0]
        for a in A:
            if a < a_min:
                a_min = a
        sorted_A.append(a_min)
        A.remove(a_min)
    return sorted_A

#------------------------------------------------------------------------------
#        2. Insertion
#------------------------------------------------------------------------------
def insertion_sort(A):
    """do something."""
    hi = 1 # upper bound
    while hi < len(A):
        j = hi
        while j > 0 and A[j-1] > A[j]:
            A[j-1], A[j] = A[j], A[j-1] # swap!
            j -= 1
        hi += 1
    return A

#------------------------------------------------------------------------------
#        3. Merge
#------------------------------------------------------------------------------
def merge_sort(A):
    """Recursively merge-sort halves of the list."""
    # Trivial sort
    if len(A) < 2:
        return A
    split = int(len(A)/2)
    bot = merge_sort(A[split:])
    top = merge_sort(A[:split])
    return merge(bot, top)

def merge(A, B):
    """Merge two sorted lists."""
    C = []
    # Take lesser of two elements
    while A and B:
        if A[0] < B[0]:
            C.append(A.pop(0))
        else:
            C.append(B.pop(0))
    # Grab the rest
    if A:
        C.extend(A)
    elif B:
        C.extend(B)
    return C

#------------------------------------------------------------------------------
#        4. Quick
#------------------------------------------------------------------------------
def quick_sort(A):
    """Wrapper to fit in with paradigm."""
    return quick_sort_(A, 0, len(A)-1)

def quick_sort_(A, lo, hi):
    """Recursively most elements less than pivot to the left."""
    if lo < hi:
        p = partition(A, lo, hi)
        quick_sort_(A, lo, p-1)
        quick_sort_(A, p+1, hi)
    return A

def partition(A, lo, hi):
    piv = A[hi] # pivot value
    i = lo - 1
    for j in range(lo, hi):
        if A[j] < piv:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[hi] = A[hi], A[i+1]
    return i+1

#==============================================================================
#==============================================================================
