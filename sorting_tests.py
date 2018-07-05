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

from sorting_algos import bubble_sort, selection_sort

N = 10
A = [8, 4, 3, 2, 1, 7, 6, 0, 5, 9]
sorted_A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

sort_funs = [bubble_sort, selection_sort]

#------------------------------------------------------------------------------ 
#        Run general sorting algorithm tests
#------------------------------------------------------------------------------
tests = 0
fails = 0
for sort in sort_funs:
    # Test empty list
    tests += 1
    if sort([]) != []:
        fails += 1

    # Test single element list
    tests += 1
    if sort([0]) != [0]:
        fails += 1

    # Test on sorted list
    tests += 1
    if sort(sorted_A) != sorted_A:
        fails += 1

    # Test on reverse sorted list
    tests += 1
    if sort(A[::-1]) != sorted_A:
        fails += 1

    # Test on randomized A
    tests += 1
    if sort(A) != sorted_A:
        fails += 1

if fails > 0:
    print("{}/{} tests failed!".format(fails, tests))
else: 
    print("All {} test passed!".format(tests))

#==============================================================================
#==============================================================================
