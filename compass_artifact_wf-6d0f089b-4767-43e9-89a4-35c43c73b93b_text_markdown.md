# Python Debugging and Troubleshooting Sample Questions for TestGorilla Assessments

TestGorilla's Python assessment tests evaluate candidates' practical programming abilities with a strong focus on debugging and troubleshooting skills. This report presents sample questions likely to appear across TestGorilla's Python tests, organized by test category.

## Python (Coding): Debugging Test

This test specifically evaluates a candidate's ability to identify and fix bugs in Python code. TestGorilla presents candidates with partially working scripts and clear descriptions of expected functionality.

### Question 1: List Filtering Function (Beginner)

**Problem Statement:**
A junior developer has written a function to filter a list of numbers, keeping only those divisible by either 2 or 3. Debug the code to make it function as intended.

**Sample Code with Bugs:**
```python
def filter_numbers(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0 or num % 3 = 0:  # Syntax error
            results.append(num)  # Variable name error
    return result

# Test case
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered = filter_numbers(numbers)
print(filtered)  # Should print [2, 3, 4, 6, 8, 9, 10]
```

**Correct Solution:**
```python
def filter_numbers(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0 or num % 3 == 0:  # Fixed syntax error
            result.append(num)  # Fixed variable name
    return result
```

**Explanation:**
This question contains two common errors:
1. A syntax error using assignment operator `=` instead of comparison operator `==`
2. A variable name inconsistency where `results` is used but `result` was defined

**Common Errors:**
- Fixing only one of the two errors
- Not testing with various inputs
- Misunderstanding operator precedence in conditionals

### Question 2: User Authentication System (Intermediate)

**Problem Statement:**
Debug a user authentication function that should validate credentials based on specific requirements: username between 4-12 characters, password at least 8 characters long containing at least one digit.

**Sample Code with Bugs:**
```python
def validate_credentials(username, password):
    # Check username length
    if len(username) >= 4 or len(username) <= 12:  # Logical error: should use 'and' not 'or'
        username_valid = True
    else:
        username_valid = False
    
    # Check password
    password_valid = False
    if len(password) >= 8:
        has_digit = False
        for char in password:
            if char.isdigit():
                has_digit = True
                break
        
        if has_digit == True:
            password_valid = True
    
    # Return result
    return username_valid and password_valid
```

**Correct Solution:**
```python
def validate_credentials(username, password):
    # Check username length
    if len(username) >= 4 and len(username) <= 12:  # Fixed logic: use 'and' not 'or'
        username_valid = True
    else:
        username_valid = False
    
    # Check password
    password_valid = False
    if len(password) >= 8:
        has_digit = False
        for char in password:
            if char.isdigit():
                has_digit = True
                break
        
        if has_digit == True:
            password_valid = True
    
    # Return result
    return username_valid and password_valid
```

**Explanation:**
The function has a logical error in the username validation. It uses `or` instead of `and`, making the condition always true for most usernames (they're either ≥4 OR they're ≤12, which covers almost all cases incorrectly).

**Common Errors:**
- Missing the logical error in the boolean expression
- Not testing with boundary cases (usernames of exactly 4 or 12 characters)
- Overlooking the inclusive nature of the length requirements

### Question 3: Data Processing Function (Intermediate)

**Problem Statement:**
Debug a function that should process customer data, calculating total spending for each customer and returning high-value customers (those who spent over $1000).

**Sample Code with Bugs:**
```python
def get_high_value_customers(customer_data):
    """
    Process customer data to find high-value customers who spent over $1000.
    
    Args:
        customer_data: List of dictionaries with 'customer_id', 'name', and 'purchases'.
                      'purchases' is a list of amounts spent.
    
    Returns:
        List of dictionaries with 'customer_id', 'name', and 'total_spent'
        for customers who spent over $1000.
    """
    high_value_customers = []
    
    for customer in customer_data:
        # Calculate total spent by the customer
        total_spent = 0
        for purchase in customer['purchase']:  # KeyError: 'purchase' should be 'purchases'
            total_spent += purchase
        
        # Check if high-value customer
        if total_spent > 1000:
            high_value_customers.append({
                'customer_id': customer['customer_id'],
                'name': customer['name'],
                'total_spent': round(total_spent, 2)
            })
        
    return sorted(high_value_customers, key=lambda x: x['total_spent'])  # Logical error: should sort in descending order
```

**Correct Solution:**
```python
def get_high_value_customers(customer_data):
    high_value_customers = []
    
    for customer in customer_data:
        # Calculate total spent by the customer
        total_spent = 0
        for purchase in customer['purchases']:  # Fixed KeyError
            total_spent += purchase
        
        # Check if high-value customer
        if total_spent > 1000:
            high_value_customers.append({
                'customer_id': customer['customer_id'],
                'name': customer['name'],
                'total_spent': round(total_spent, 2)
            })
        
    return sorted(high_value_customers, key=lambda x: x['total_spent'], reverse=True)  # Fixed: sort in descending order
```

**Explanation:**
This question contains two key bugs:
1. A KeyError due to using 'purchase' instead of 'purchases' when accessing the dictionary
2. A logical error in sorting - high-value customers should typically be sorted with highest spending first

**Common Errors:**
- Focusing on just one bug and missing the other
- Not understanding the implications of key errors in dictionaries
- Forgetting to round the total_spent values as specified

## Python (Coding): Entry-Level Algorithms Test

This test evaluates a candidate's ability to implement and debug basic algorithms. Here are examples focused on the debugging aspects:

### Question 1: Selection Sort Debugging (Intermediate)

**Problem Statement:**
Implement a function to sort an array of integers in ascending order using the selection sort algorithm. The provided implementation has bugs that need fixing.

**Sample Code that Needs Debugging:**
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # There's a bug in the swapping logic below
        arr[i] = arr[min_idx]
        arr[min_idx] = arr[i]
    return arr
```

**Correct Solution:**
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):  # Start from i+1, not i
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Fix the swapping logic
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

**Explanation:**
The original code had two bugs:
- The inner loop started from index `i` instead of `i+1`, which is inefficient
- The swapping logic was incorrect: after setting `arr[i] = arr[min_idx]`, the original value of `arr[i]` is lost

**Common Errors:**
- Incorrect swapping logic
- Inefficient looping (not starting the inner loop at the correct position)
- Not handling edge cases (empty arrays, single-element arrays)

### Question 2: Duplicates Removal Bug (Beginner)

**Problem Statement:**
The following function is supposed to remove duplicates from a list while preserving the original order of elements. Find and fix the bug.

**Sample Code that Needs Debugging:**
```python
def remove_duplicates(items):
    result = []
    for item in items:
        if item not in result:
            result.remove(item)
    return result
```

**Correct Solution:**
```python
def remove_duplicates(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)  # Changed remove to append
    return result
```

**Explanation:**
The bug is in the conditional block: when an item is NOT in the result list, we incorrectly tried to remove it (which would raise an error). The fixed solution appends the item to the result list when it's not already there.

**Common Errors:**
- Not carefully reading the code to identify logical errors
- Confusing `remove()` and `append()` methods
- Not testing with various inputs

## Python (Working With Arrays) Test

This test evaluates a candidate's ability to manipulate arrays/lists and debug array-related issues.

### Question 1: Daily Temperature Problem (Intermediate)

**Problem Statement:**
Debug a function that takes a list of daily temperatures and returns a list where each element is the number of days you would have to wait until a warmer temperature.

**Sample Code with Bugs:**
```python
def daily_temperatures(temperatures):
    """
    Calculate how many days you would have to wait for a warmer temperature.
    
    Parameters:
    temperatures (list): A list of daily temperatures.
    
    Returns:
    list: A list where each element is the number of days to wait for a warmer temperature.
    """
    n = len(temperatures)
    result = [0] * n
    
    for i in range(n):
        for j in range(i+1, n):
            if temperatures[j] < temperatures[i]:  # Bug: This comparison is wrong
                result[i] = j - i
                break
    
    return result

# Example usage:
temps = [73, 74, 75, 71, 69, 72, 76, 73]
# Expected output: [1, 1, 4, 2, 1, 1, 0, 0]
```

**Correct Solution:**
```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    
    for i in range(n):
        for j in range(i+1, n):
            if temperatures[j] > temperatures[i]:  # Fixed: Check for warmer temperature
                result[i] = j - i
                break
    
    return result
```

**Explanation:**
The original code had a crucial bug in the comparison logic. It was checking if the future temperature was less than the current temperature, which is the opposite of what we want. The correct solution checks if the future temperature is greater than (warmer than) the current temperature.

**Common Errors:**
- Using the wrong comparison operator
- Not breaking out of the loop after finding the first warmer day
- Inefficient implementation (this solution is O(n²), while an O(n) solution exists using a stack)

### Question 2: Merge Sorted Lists Efficiency (Intermediate)

**Problem Statement:**
Debug and optimize the function that merges two sorted lists into a single sorted list.

**Sample Code to Debug:**
```python
def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists into a single sorted list.
    
    Parameters:
    list1 (list): First sorted list of integers.
    list2 (list): Second sorted list of integers.
    
    Returns:
    list: Merged sorted list.
    """
    # Inefficient implementation
    merged = list1 + list2
    return sorted(merged)  # Using built-in sort - O((m+n)log(m+n)) complexity
```

**Correct Solution:**
```python
def merge_sorted_lists(list1, list2):
    merged = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged
```

**Explanation:**
The original implementation is inefficient - it concatenates the lists and then sorts the result, which has O((m+n)log(m+n)) time complexity. The optimized solution takes advantage of the fact that both input lists are already sorted, achieving O(m+n) time complexity.

**Common Errors:**
- Not leveraging the property that input lists are already sorted
- Not handling cases where one list is exhausted before the other
- Off-by-one errors in the loop conditions

## Python (Coding): Data Structures and Objects Test

This test evaluates understanding of data structures, object-oriented programming, and related debugging skills.

### Question 1: Stack Implementation Debugging (Intermediate)

**Problem Statement:**
Debug the implementation of a Stack class that has errors in its core methods.

**Sample Code with Bugs:**
```python
class Stack:
    def __init__(self):
        self.items = []
        
    def push(self, item):
        self.items.append(item)
        
    def pop(self):
        return self.items.pop()
        
    def peek(self):
        return self.items[-1]
        
    def is_empty(self):
        return self.items == []
        
    def size(self):
        return len(self.items)
```

**Correct Solution:**
```python
class Stack:
    def __init__(self):
        self.items = []
        
    def push(self, item):
        self.items.append(item)
        
    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()
        
    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.items[-1]
        
    def is_empty(self):
        return len(self.items) == 0  # More efficient than comparing to []
        
    def size(self):
        return len(self.items)
```

**Explanation:**
The original implementation has two key issues:
1. No error handling for `pop()` and `peek()` when the stack is empty
2. Inefficient implementation of `is_empty()` using equality comparison instead of checking length

**Common Errors:**
- Forgetting to handle edge cases like empty stacks
- Using inefficient equality comparisons for empty list checking
- Not raising appropriate exceptions for error conditions

### Question 2: Binary Search Tree Implementation (Advanced)

**Problem Statement:**
Debug this Binary Search Tree implementation that has issues with its insertion method.

**Sample Code with Bugs:**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
            
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                # Bug: incorrect recursive call
                self._insert_recursive(node.left, value)  # Should be node.right
```

**Correct Solution:**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
            
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)  # Fixed recursive call
```

**Explanation:**
The bug is in the recursive insertion for the right subtree. The original code incorrectly recursively calls `_insert_recursive(node.left, value)` instead of `_insert_recursive(node.right, value)`, which would cause values to be inserted in the wrong place.

**Common Errors:**
- Confusing left and right subtrees in recursive calls
- Not maintaining BST property (left < node < right)
- Missing base cases for recursion

## Common Python Debugging Scenarios from Similar Platforms

These examples cover debugging patterns frequently seen across multiple assessment platforms:

### 1. Mutable Default Arguments Bug

**Problem Statement:**
Debug the function that uses a default parameter value. It doesn't work correctly when called multiple times.

**Buggy Code:**
```python
def print_from_stream(n, stream=EvenStream()):
    for _ in range(n):
        print(stream.get_next())
```

**Correct Solution:**
```python
def print_from_stream(n, stream=None):
    if stream is None:
        stream = EvenStream()
    for _ in range(n):
        print(stream.get_next())
```

**Explanation:**
Default arguments are evaluated only once when the function is defined, not each time the function is called. The fix uses `None` as the default value and creates a new instance inside the function when needed.

### 2. List Mutation During Iteration

**Problem Statement:**
Debug the function that tries to remove all even numbers from a list.

**Buggy Code:**
```python
def remove_even_numbers(numbers):
    for num in numbers:
        if num % 2 == 0:  # If number is even
            numbers.remove(num)
    return numbers
```

**Correct Solution:**
```python
def remove_even_numbers(numbers):
    # Create a new list with odd numbers only
    return [num for num in numbers if num % 2 != 0]
```

**Explanation:**
The bug occurs because the function is modifying the list while iterating over it. When an element is removed, the indices shift, causing the iteration to skip elements. The fix creates a new list instead of modifying the original during iteration.

### 3. Off-by-One Loop Error

**Problem Statement:**
Debug the function that calculates the sum of integers from 1 to n (inclusive).

**Buggy Code:**
```python
def sum_to_n(n):
    total = 0
    for i in range(n):
        total += i
    return total
```

**Correct Solution:**
```python
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
```

**Explanation:**
In Python, `range(n)` generates numbers from 0 to n-1, not 1 to n. This results in an "off-by-one" error. The fix uses `range(1, n + 1)` which generates numbers from 1 to n inclusive.

## Debugging Patterns and Testing Tips

Throughout these examples, several common debugging patterns emerge:

1. **Syntax Errors**:
   - Incorrect operators (= vs ==)
   - Missing colons, parentheses, or indentation
   - Mismatched quotes or brackets

2. **Runtime Errors**:
   - KeyErrors when accessing dictionaries
   - IndexErrors when accessing lists
   - TypeError when mixing incompatible types
   - NameErrors from undefined variables
   - Infinite recursion without base cases

3. **Logical Errors**:
   - Incorrect boolean operators (and/or confusion)
   - Off-by-one errors in loops and indices
   - Boundary condition issues
   - Incorrect sorting order
   - Modifying collections during iteration

**Testing Tips for Debugging Questions:**

1. **Check Boundary Conditions**: Test with empty inputs, single items, and edge cases
2. **Verify Logic**: Trace through the code with a simple example by hand
3. **Inspect Variables**: Look for inconsistent variable names or scope issues
4. **Consider Efficiency**: Look for unnecessary operations or poor algorithm choices
5. **Find Multiple Bugs**: Don't stop after finding the first issue - there may be multiple bugs

By practicing with these examples and understanding common debugging patterns, candidates will be better prepared for TestGorilla's Python assessment tests and their debugging-focused questions.