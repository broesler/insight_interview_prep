#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: sorting_tests.py
#  Created: 07/05/2018, 17:32
#   Author: Bernie Roesler
#
"""
  Description: Test various sorting algorithms
"""
#==============================================================================

import numpy as np

from sorting_algos import bubble_sort, selection_sort, insertion_sort,\
                          merge_sort, quick_sort

def should_be(x):
    """Test a condition."""
    global tests, fails
    tests += 1
    if not x:
        fails += 1

# Define test cases
A = [8, 4, 3, 2, 1, 7, 6, 0, 5, 9]
sorted_A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

sort_funs = [bubble_sort, selection_sort, insertion_sort,\
             merge_sort, quick_sort]

#------------------------------------------------------------------------------
#        Run general sorting algorithm tests
#------------------------------------------------------------------------------
tests = 0
fails = 0
for sort in sort_funs:
    should_be(sort([]) == [])                   # empty list
    should_be(sort([0]) == [0])                 # single element list
    should_be(sort([1, 1, 1]) == [1, 1, 1])     # all equal
    # Pass a copy so we don't sort the original
    should_be(sort(list(sorted_A)) == sorted_A) # sorted list
    should_be(sort(list(A[::-1])) == sorted_A)  # reverse sorted list
    should_be(sort(list(A)) == sorted_A)        # randomized A

if fails > 0:
    print("{}/{} tests failed!".format(fails, tests))
else:
    print("All {} test passed!".format(tests))

#==============================================================================
#==============================================================================
