# Python Programming

### 1. Python Questions

#### What are the key features of Python?
 - Python is one of the most popular programming languages used by data scientists and AIML professionals. This popularity is due to the following key features of Python:
 - Python is easy to learn due to its clear syntax and readability
 - Python is easy to interpret, making debugging easy
 - Python is free and Open-source
 - It can be used across different languages
 - It is an object-oriented language which supports concepts of classes
 - It can be easily integrated with other languages like C++, Java and more

#### What Native Data Structures Can You Name in Python?
Common native data structures in Python are as follows:
- Dictionaries
- Lists
- Sets
- Strings
- Tuples

#### What is an interface?
At a high level, an interface acts as a blueprint for designing classes. Like classes, interfaces define methods. Unlike classes, these methods are abstract. An abstract method is one that the interface simply defines. It doesn’t implement the methods. This is done by classes, which then implement the interface and give concrete meaning to the interface’s abstract methods.

##### Of These, Which Are Mutable, and Which Are Immutable?
- Lists, dictionaries, and sets are mutable. This means that you can change their content without changing their identity.
- Strings and tuples are immutable, as their contents can’t be altered once they’re created.


##### What is the difference between a list and a tuple?
Where they are common:
- They are both sequence data types that store a collection of items
- They can store items of any data type
- And any item is accessible via its index.

Where they differ:
 - lists are mutable (they can be modified).
  - Lists are ordered by one central index
  - Implication of iterations is Time-consuming
  - Lists consume more memory
  - The list is better for performing operations, such as insertion and deletion.
  - `Ex: list_data = ['an', 'example', 'of', 'a', 'list']`
  - list(list_name) requires copying of all data to a new list.


 - tuples are immutable (they cannot be modified).
  - a tuple may hold multiple data types together in an index-like-form.
  - The implication of iterations is comparatively Faster
  - Tuple data type is appropriate for accessing the elements
  - Tuple consume less memory as compared to the list
  - Tuples cannot be copied. The reason is that tuples are immutable. If you run tuple(tuple_name), it will immediately return itself.
  - `Ex: tuple_data = ('this', 'is', 'an', 'example', 'of', 'tuple')`

##### What is the difference between lists and sets?
Sets are like lists as well in terms of being a sequence of objects, but they too also have a key difference: in this case, sets only take unique values. So, if you have a list that has duplicate values within it and you try to convert it into a set, the resulting set will eliminate all duplicates and leave only the sequence of unique values in your original list.


##### Are dictionaries or lists faster for lookups?
- `Lists` are slower for lookups: it’ll take `O(N)` time since filtering through for a value will require as a worst-case scenario filtering through every value in the list.
- `Dictionaries` are well-set up with key-value pairs, similar to a hash table. Thus the time to search will be `O(1)` as long as you have the correct key.


#### When Would You Use a List vs. a Tuple vs. a Set in Python?
- A `list` is a common data type that is highly flexible. It can store a sequence of objects that are mutable, so it’s ideal for projects that demand the storage of objects that can be changed later.

- A `tuple` is similar to a list in Python, but the key difference between them is that tuples are immutable. They also use less space than lists and can only be used as a key in a dictionary. Tuples are a perfect choice when you want a list of constants.

- `Sets` are a collection of unique elements that are used in Python. Sets are a good option when you want to avoid duplicate elements in your list. This means that whenever you have two lists with common elements between them, you can leverage sets to eliminate them.

#### Can You Explain What a List or Dict Comprehension Is?
List Comprehension is a handy and faster way to create lists in Python in just a single line of code. It helps us write easy to read for loops in a single line.

The idea of comprehension is not just unique to lists in Python. Dictionaries, one of the commonly used data structures in data science, can also do comprehension. With dict comprehension or dictionary comprehension, one can easily create dictionaries.

List Comprehension Examples:
```python
> my_list = [0,1,2,3,4,5]
> [str(x) for x in my_list]
['0', '1', '2', '3', '4', '5']
#
> ['even' if i%2==0 else 'odd' for i in my_list]
['even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd']

```

Dict Comprehension Examples:
```python
# dict comprehension to create dict with numbers as values
>{str(i):i for i in [1,2,3,4,5]}
{'1': 1, '3': 3, '2': 2, '5': 5, '4': 4}
#
# create list of fruits
>fruits = ['apple', 'mango', 'banana','cherry']
# dict comprehension to create dict with fruit name as keys
>{f:len(f) for f in fruits}
{'cherry': 6, 'mango': 5, 'apple': 5, 'banana': 6}
#
>{f:f.capitalize() for f in fruits}
{'cherry': 'Cherry', 'mango': 'Mango', 'apple': 'Apple', 'banana': 'Banana'}
#
# dict comprehension example using enumerate function
>{f:i for i,f in enumerate(fruits)}
{'cherry': 3, 'mango': 1, 'apple': 0, 'banana': 2}
```

### 2. Python Coding Exercises


#### Python Coding 1 - Using list comprehension, print the odd numbers between 0 and 100.
List comprehensions are a feature in Python that allows us to work with algorithms within the default list data structure in Python. Here, we’re looking for odd numbers.
```
[x for x in range(100) if x%2 !=0]
```

#### Python Coding 2 - Show me three different ways of fetching every third item in the list.
```
cnt=0
for i in thelist:
 if cnt % 3 == 0:
   print(i)
 cnt+=1
```
```
cnt=0
for i in range(len(thelist)):
 if cnt % 3 == 0:
   print(i)
 cnt+=1    
```
```
[x for i, x in enumerate(thelist) if i%3 == 0]
```


#### Python Coding 3 - Write a regular expression that confirms an email id using the python reg expression module “re”?
Example of email: john.smith@gmail.com
```python
import re
print(re.search(r"[0-9a-zA-Z.]+@[a-zA-Z]+\.(com|co\.in)$", john.smith@gmail.com))
```


#### Python Coding 4 - Percent Change (Arithmetic Algorithm)
Write a function that calculates the % change between two values `x1` and `x2`.

```python
def percent(x1, x2):
    '''returns the percent change between x1 and x2
    '''
    return (x2-x1)/x1 * 100

print(percent(10,2))
# returns -80.0
```

#### Python Coding 5 - Insertion Sort (Sorting Algorithm)
We want to take an input (a list of numbers) and return them sorted from the least amount to the most amount.

Notes: `sort()` is inplace, meaning it modifies the original list it is called upon and returns None (which is why you are getting the error 'NoneType' object is not iterable).

You will need to use something that leaves the original list intact and returns a new, sorted list, like `sorted()``:


```python
def sort_list(mylist):
    '''returns a sorted list
    '''
    return sorted(mylist)

print(sort_list([1,5,3,9,5,2]))
# returns [1, 2, 3, 5, 5, 9]
```

Another approach which is a bit more costly
```python
unsorted_list = [3, 1, 0, 9, 4]
sorted_list = []

while unsorted_list:
    minimum = unsorted_list[0]
    for item in unsorted_list:
        if item < minimum:
            minimum = item
    sorted_list.append(minimum)
    unsorted_list.remove(minimum)

print(sorted_list)
# returns [0, 1, 3, 4, 9]
```


#### Python Coding 6 - Binary Search (Search Algorithm) and recursion
Starting from a sorted list (as we have seen in previous exercise), Implement a binary search algorithm.
This checks to see if the midpoint is your search term, then bisects the list to search for an item within the subset. If the item is on the list, it’ll return True—if not it’ll return False.

Notes: you can use `//` that is a floor division - it rounds the result down to the nearest whole number

```python
def binary_search(x, search_item):

    if len(x) == 0:
        return False
    mid = len(x) // 2
    if x[mid] == search_item:
        return True
    if search_item < x[mid]:
        return binary_search(x[:mid], search_item)
    else:
        return binary_search(x[mid + 1 :], search_item)

a_list = [2,3,4,5,8]
print(a_list)
print(binary_search(a_list, 2))
print(binary_search(a_list, 7))
# returns
# [2, 3, 4, 5, 8]
# True
# False
```

#### Python Coding 7 - Palindrome
Create a function that returns true if the input string is a palindrome and false otherwise.

1st option, Using list slicing to check if the original string is the original string sampled backwards one char at a time.
But this is costly. Slicing essentially creates a copy of the original string. If the original string is large, its copy will double its space use.

```python
def is_palyndrome(mystring):
    '''function that returns true if the input string is a palindrome
    '''
    return mystring == mystring[::-1]

assert is_palindrome('a')== True
assert is_palindrome('ab')== False
assert is_palindrome('aba') == True
assert is_palindrome('abba') == True

```

2nd option: better with space, using 2 pointers:
```python
def is_palindrome(mystring):
    start = 0
    end = len(mystring)-1
    for i in range(len(mystring) // 2):
        if mystring[i] == mystring[end]:
            end -= 1
            continue
        else:
            return False
    return True

def is_palindrome2(word):
    for i in range(len(word) // 2):
        if word[i] != word[-1 -i]:
          return False
    return True

assert is_palindrome('a')== True
assert is_palindrome('ab')== False
assert is_palindrome('aba') == True
assert is_palindrome('abba') == True
```

Similar option 2
```python
def is_palindrome(arr):
 start = 0
 end = len(arr)-1
 while start < end:
    if arr[start] != arr[end]:
      return False
    start +=1
    end -=1
  return True
assert is_palindrome('a')== True
assert is_palindrome('aba') == True
assert is_palindrome('abba') == True
```
