# Programming


#### 1. Do you have experience with Spark or big data tools for machine learning?
 - You’ll want to get familiar with the meaning of big data for different companies and the different tools they’ll want. Spark is the big data tool most in demand now, able to handle immense datasets with speed. Be honest if you don’t have experience with the tools demanded, but also take a look at job descriptions and see what tools pop up: you’ll want to invest in familiarizing yourself with them.

#### 2. What are some differences between a linked list and an array?
 - An **array** is an ordered collection of objects.
 - A **linked list** is a series of objects with pointers that direct how to process them sequentially.
 - An array assumes that every element has the same size, unlike the linked list.
 - A linked list can more easily grow organically: an array has to be pre-defined or re-defined for organic growth. Shuffling a linked list involves changing which points direct where—meanwhile, shuffling an array is more complex and takes more memory.

##### 2b. What is the difference between a list and a tuple [in Python]?
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

##### 2c. What is the difference between lists and sets?
Sets are like lists as well in terms of being a sequence of objects, but they too also have a key difference: in this case, sets only take unique values. So, if you have a list that has duplicate values within it and you try to convert it into a set, the resulting set will eliminate all duplicates and leave only the sequence of unique values in your original list.


##### 2d. Are dictionaries or lists faster for lookups?
- `Lists` are slower for lookups: it’ll take `O(N)` time since filtering through for a value will require as a worst-case scenario filtering through every value in the list.
- `Dictionaries` are well-set up with key-value pairs, similar to a hash table. Thus the time to search will be `O(1)` as long as you have the correct key.


#### 3. Describe a hash table.
 - A hash table is a data structure that produces an associative array. A key is mapped to certain values through the use of a hash function. They are often used for tasks such as database indexing.
 - A hash table is a data structure that is used to store keys/value pairs. It uses a hash function to compute an index into an array in which an element will be inserted or searched. By using a good hash function, hashing can work well. Under reasonable assumptions, the average time required to search for an element in a hash table is O(1).

Let us consider string S. You are required to count the frequency of all the characters in this string.

#### 4. Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?
 - What’s important here is to define your views on how to properly visualize data and your personal preferences when it comes to tools. Popular tools include R’s ggplot, Python’s seaborn and matplotlib, and tools such as Plot.ly and Tableau.

#### 5. How are primary and foreign keys related in SQL?
 - Most machine learning engineers are going to have to be conversant with a lot of different data formats. SQL is still one of the key ones used. Your ability to understand how to manipulate SQL databases will be something you’ll most likely need to demonstrate. In this example, you can talk about how foreign keys allow you to match up and join tables together on the primary key of the corresponding table—but just as useful is to talk through how you would think about setting up SQL tables and querying them.

#### 6. How would you build a data pipeline?
 - Data pipelines are the bread and butter of machine learning engineers, who take data science models and find ways to automate and scale them. Make sure you’re familiar with the tools to build data pipelines (such as Apache Airflow) and the platforms where you can host models and pipelines (such as Google Cloud or AWS or Azure). Explain the steps required in a functioning data pipeline and talk through your actual experience building and scaling them in production.

#### 7. What are the key features of Python?
 - Python is one of the most popular programming languages used by data scientists and AIML professionals. This popularity is due to the following key features of Python:
 - Python is easy to learn due to its clear syntax and readability
 - Python is easy to interpret, making debugging easy
 - Python is free and Open-source
 - It can be used across different languages
 - It is an object-oriented language which supports concepts of classes
 - It can be easily integrated with other languages like C++, Java and more

#### 8. How do you handle missing or corrupted data in a dataset?
You could find missing/corrupted data in a dataset and either drop those rows or columns, or decide to replace them with another value.

In Pandas, there are two very useful methods: `isnull()` and `dropna()` that will help you find columns of data with missing or corrupted data and drop those values. If you want to fill the invalid values with a placeholder value (for example, 0), you could use the `fillna()` method.

#### 9. What is the Big O notation
In computer science, big O notation is used to classify algorithms according to how their run time or space requirements grow as the input size grows
Examples:
- `O(1)` - constant / Ex: Determining if a binary number is even or odd; Calculating (-1)^{n}; Using a constant-size lookup table
- `O(n)` - linear / Ex: Finding an item in an unsorted list or in an unsorted array

![big O](../images/BigO-complexity.png)

##### 9b. What does it mean if an operation is O(log n)?  
 - `O(log n)` means for every element, you're doing something that only needs to look at log N of the elements. This is usually because you know something about the elements that let you make an efficient choice (for example to reduce a search space). Big

![O)log n)](../images/BigO-logn.png)

##### 9c. Why do we use Big O notation to compare algorithms?   
The fact is it's difficult to determine the exact runtime of an algorithm. It depends on the speed of the computer processor. So instead of talking about the runtime directly, we use Big O Notation to talk about **how quickly the runtime grows depending on input size**.

With Big O Notation, we use the size of the input, which we call `n`. So we can say things like the runtime grows ``“on the order of the size of the input” (O(n))`` or ``“on the order of the square of the size of the input” (O(n2))``. Our algorithm may have steps that seem expensive when n is small but are eclipsed eventually by other steps as n gets larger. For Big O Notation analysis, we care more about the stuff that grows fastest as the input grows, because everything else is quickly eclipsed as n gets very large.

#### 10. Show me three different ways of fetching every third item in the list.
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

#### 11. Using list comprehension, print the odd numbers between 0 and 100.
List comprehensions are a feature in Python that allows us to work with algorithms within the default list data structure in Python. Here, we’re looking for odd numbers.
```
[x for x in range(100) if x%2 !=0]
```

#### 12. Write a regular expression that confirms an email id using the python reg expression module “re”?
Example of email: john.smith@gmail.com
```python
import re
print(re.search(r"[0-9a-zA-Z.]+@[a-zA-Z]+\.(com|co\.in)$", john.smith@gmail.com))
```
