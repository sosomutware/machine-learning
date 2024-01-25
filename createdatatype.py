# 1.	Create an integer, float, and string variable
a = 2
b = 11.5
st = 'Hello world'
# 2.	Print these to the screen.
print(a)
print(b)
print(st)
# 3.	Play around using different variable names
m = 10
y = 2.5
msg = "This fun"
print(m, y, msg)
# 4.	Check the type of one of your variables
print(type(st))
# 5.	operations using the arithmetic operators & their results
print("\n", "10+2.5= ", m + y, "\n", "10*2.5= ", m * y, "\n", "10-2.5= ", m - y, "\n", "10/2.5", m / y)
# 6.	Write a couple of operations using  comparison operators
print(m > y, y >= a, msg == st)
# 7.	Create a string variable. 8.Print out the length of the string.
n_st = "Machine learning"
print(len(n_st))
# 9.	Create a string that is 10 characters in length.
st10 = "Sebastienn"
print(st10[1])  # Print the second character
print(st10[-3])  # Print the third to last character
print(st10[4:])  # Print all characters after the fourth character
print(st10[1:8])  # Print characters 2-8
# 10.	With an example created and displayed in python explain:Tuples,Liste,Dictionary,set
# Example of Tuple
myTuple = (1, 2, 3, 'a', 'b')
print(myTuple)

# Example of List
myList = [1, 2, 3, 'a', 'b']
print(myList)

# Example of Dictionary
mydict = {'name': 'Muhire', 'age': 23, 'city': 'Ngoma'}
print(mydict)

# Example of Set
my_set = {1, 2, 3, 'a', 'b'}
print(my_set)
# 11.	Create a list and populate it with  some elements.
my1List = [10, 20, 30, 'Muhire', 'Yvan']
print(my1List)
# Replace the second element in your list with the integer 2
my1List[1] = 2
print(my1List)
# Use insert() to put the integer 3 after the 2 that you just added to your list.
my1List.insert(2, 3)
print(my1List)
# Use append() to add 4 "end" as the last element in your list.
my1List.append(4)
my1List.append("end")
print(my1List)
# Use del to remove the 3 that you added to the list earlier.
del my1List[2]

print(my1List)
# 12.	We have a list of species:
species_list = ['Lion', 'Tiger', 'Leopard', 'Cheetah']
# Use a for loop to print each element of the list to the screen.
for i in species_list:
    print(i)
# 13.	Create an empty list
# Create an empty list
new_list = []
# Use the range() and append() functions to add the integers 1-20 to the empty list.
for i in range(1, 21):
    new_list.append(i)
# Print the list to the screen
print(new_list)


# 14.	Create a function that takes two inputs, multiplies them, and then returns the result
def two_inputs(k, l):
    r = k * l
    return r


re = two_inputs(2, 3)
print(re)
