#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: stack.py
#  Created: 08/03/2018, 14:33
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

class Stack:
    """Define a stack data structure using a list."""
    def __init__(self, items=list()):
        self.items = items
        self.len = len(self.items)

    def push(self, item):
        """Push item onto stack."""
        self.len += 1
        return self.items.append(item)

    def pop(self):
        """Remove item from top of stack."""
        self.len -= 1
        return self.items.pop()

    def peek(self):
        """Look at the top of the stack but don't pop it."""
        return self.items[-1]

    def __str__(self):
        return str(self.items)

    def __repr__(self):
        return str(self.items)

def isValid(s):
    """
    :type s: str
    :rtype: bool
    """
    open_pairs = dict({')':'(', ']':'[', '}':'{'})
    # This object appears to persist across function calls?!
    stack = Stack(list())
    for c in s:
        # No problems until we encounter a close
        if c in '([{':
            stack.push(c)
            continue

        if c in ')]}':
            if stack.peek() == open_pairs[c]:
                stack.pop()
            else:
                return False, stack.items

    # If we've matched all parens, none left on stack
    if stack.len == 0:
        return True, stack.items
    else:
        return False, stack.items

# Test case
def test_stack():
    stack = Stack([7, 8, 9])
    stack.push(1)
    stack.push(2)
    stack.push(3)
    stack.push(4)
    print(stack)
    popped = stack.pop()
    print(stack, popped)

# inputs = dict({'[':False,
#                '(':False,
#                '{':False,
#                '()':True,
#                '([])':True,
#                '[{]':False,
#                '()[]{}':True})
# inputs = dict({'()[]{}':True})
# fails = []
# for s, v in inputs.items():
#     out, stack = isValid(s)
#     if out != v:
#         fails.append((s, out, v))
#
# if fails:
#     print('These tests failed: {}'.format(fails))
# else:
#     print('All tests passed!')

test1 = isValid('[(]') # test = (False, ['[', '('])
test2 = isValid('()')  # test = (False, ['[', '(']) # STACK SHOULD BE EMPTY!!

#==============================================================================
#==============================================================================
