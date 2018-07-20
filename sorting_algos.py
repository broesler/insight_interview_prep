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

#------------------------------------------------------------------------------
#        5. Heap
#------------------------------------------------------------------------------
def heap_sort(A):
    """Heap sort!

    1. 'heapify' array as max-heap
    2. Move the root to the end of the array (in its sorted place)
    3. 'sift_down' to fix the remaining elements into a new heap

    Tests
    -----
    >>> A = [5, 8, 6, 0, 7, 9, 1, 2, 4, 3]
    [5, 8, 6, 0, 7, 9, 1, 2, 4, 3]
    >>> heap_sort(A)
    heapify:  [9, 8, 6, 4, 7, 5, 1, 2, 0, 3]
    sift_down:  [8, 7, 6, 4, 3, 5, 1, 2, 0, 9]
    sift_down:  [7, 4, 6, 2, 3, 5, 1, 0, 8, 9]
    sift_down:  [6, 4, 5, 2, 3, 0, 1, 7, 8, 9]
    sift_down:  [5, 4, 1, 2, 3, 0, 6, 7, 8, 9]
    sift_down:  [4, 3, 1, 2, 0, 5, 6, 7, 8, 9]
    sift_down:  [3, 2, 1, 0, 4, 5, 6, 7, 8, 9]
    sift_down:  [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]
    sift_down:  [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
    sift_down:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    See Also
    --------
    A.sort(), sorted(A)
    """
    count = len(A)

    heapify(A, count) # max heap, A[0] is the root

    # A[:end] is a heap, A[end:count] is in sorted order
    end = count - 1
    while end > 0:
        # Move A[0] to its sorted place at the end
        A[end], A[0] = A[0], A[end]
        # reduce the heap size by one
        end -= 1
        # repair the heap property
        sift_down(A, 0, end)
    return A

# Indexing functions for a max heap
p_idx = lambda i: (i-1) // 2
l_idx = lambda i: 2*i + 1
r_idx = lambda i: 2*i + 2

def heapify(A, count):
    """Put elements of A in max heap order, in-place."""
    # start at lowest parent node on the heap
    start = p_idx(count-1)

    while start >= 0:
        # sift down the node at index 'start' to the proper place such that all
        # nodes below the start index are in heap order
        sift_down(A, start, count-1)
        # go to the next parent node
        start -= 1
    return A

def sift_down(A, start, end):
    """Repair the heap whose root element is at index 'start', assuming the
    heaps rooted at its children are valid.
    """
    root = start

    # While the root has at least one child
    while l_idx(root) <= end:
        child = l_idx(root)   # Left child of root
        swap = root           # Keep track of child to swap with

        if A[swap] < A[child]:
            swap = child

        # If there is a right child and that child is greater
        if (child + 1 <= end) and (A[swap] < A[child + 1]):
            swap = child + 1  # by definition of [lr]_idx above

        if swap == root:
            # The root holds the largest element. Since we assume the heaps
            # rooted at the children are valid, this means that we are done.
            return
        else:
            # Move the root down the heap
            A[root], A[swap] = A[swap], A[root]
            root = swap  # keep track of the root node!


#------------------------------------------------------------------------------ 
#        6. Tim sort
#------------------------------------------------------------------------------

#==============================================================================
#==============================================================================
