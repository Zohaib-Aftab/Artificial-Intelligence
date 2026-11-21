"""Task No 1:
Create a list that contains five student names. After the list is created, add two more
student names. Then, remove one student from the list. Sort the final list of names
alphabetically and print the length of the list (i.e., the number of students)."""
mylist = ["Ahmed","Ali","Shoaib","Farhan","Haris"]
print("List of 5 Students:",mylist)
mylist.append("Hamza")
mylist.append("Shahzaib")
print("Two more Student are Added:",mylist)
mylist.remove("Shoaib")
print("List after Remove one Student:",mylist)
mylist.sort()
print("After sorting the list:",mylist)
print("Total number of Students in list:",(len(mylist)))




"""Task No 2:
Two sets of students need to be managed. The first set contains five student names,
while the second set contains three student names, some of which may overlap with the first
set. Perform the following operations:
1. Calculate and print the union of both sets (all unique students).
2. Calculate and print the intersection of both sets (students common to both groups).
Calculate and print the difference between the first and second sets (students in the first set
but not in the second). This task evaluates understanding of set operations, including union,
intersection, and difference."""
first = {"Fahad","Khalid","Hassan","Usama","Asif"}
second = {"Hassan","Jawad","Fahad"}
union = first | second
print("Union of Both Sets:",union)
intersection = first & second
print("Intersection of Both Sets:",intersection)
difference = first - second
print("Names that are in set one and not in second set:",difference)




"""Task No 3:
Create a tuple that contains four integers. Attempt to modify one element of the
tuple to demonstrate the immutability of tuples. After encountering the immutability error,
convert the tuple into a list, modify the list, and convert it back into a tuple. Print the new
tuple."""
mytuple = (20,40,60,80)
print("Elements of Tuple:",mytuple)
try:
    mytuple[2] = 35
except TypeError:
    print("Tuple cannot be modified")
mylist = list(mytuple)
mylist[2] = 35
newtuple = tuple(mylist)
print("After Modifying the Tuple:",newtuple)




"""Task No 4:
Create a list of lists where each inner list contains three elements: a student's name,
age, and GPA. Access and print the details of the second student. Next, sort the list of students
in descending order based on their GPA and print the sorted list. Finally, add a new student’s
record to the list and print the updated list."""
mylist = [["Daniyal",23,3.4],["Subhan",21,3.7],["Farhan",25,2.8]]
print("List of Students:",mylist)
print("Details of Second Student:",mylist[1])
mylist.sort(key = lambda x: x[2], reverse = True)
print("Student sorted by GPA:",mylist)
mylist.append(["Wahab",29,2.7])
print("List after adding another Student:",mylist)




"""Task no 5:
Create a dictionary where the keys are student names and the values are tuples
containing each student's age and GPA. Perform the following operations:
1. Access and print the age and GPA of a specific student.
2. Update the GPA of one student.
3. Add a new student to the dictionary.
4. Remove a student from the dictionary. This task tests skills related to dictionary
manipulation, including accessing, updating, and deleting entries."""
mydictionary = {"Daniyal" : (23,3.2),"Hamza": (20,2.7),"Yasir" : (25,3.5)}
print("Students Data:",mydictionary)
print("Yasir Age:",mydictionary["Yasir"][0])
print("Yasir GPA:",mydictionary["Yasir"][1])
mydictionary["Yasir"] = (25,3.1)
print("Yasir GPA Changed:",mydictionary)
mydictionary["Shoaib"] = (27,3,9)
print("Add one Student:",mydictionary)
del mydictionary["Daniyal"]
print("Delete One Student:",mydictionary)




"""Task no 6:
Create a dictionary where each key is a subject name (e.g., 'Math', 'Science') and the
corresponding value is a set of student names enrolled in that subject. Perform the following
operations:
1. Add a new student to the 'Math' subject.
2. Remove a student from the 'Science' subject.
3. Identify and print the students enrolled in both 'Math' and 'Science' using set
intersection.
4. Identify and print all unique students across all subjects using set union. This task tests
the ability to combine dictionaries with sets and perform operations such as
intersection and union in a multi-dimensional context."""
mydictionary = {"Math": {"Ahmed", "Ali", "Haris"}, "Science": {"Ali", "Haris", "Shoaib"}}
print("Original Dictionary:",mydictionary)
mydictionary["Math"].add("Hamza")
print("After Adding One Student:",mydictionary)
mydictionary["Science"].remove("Shoaib")
print("After Removing One Student:",mydictionary)
intersection = mydictionary["Math"] & mydictionary["Science"]
print("Students in both Math and Science:",intersection)
union = mydictionary["Math"] | mydictionary["Science"]
print("All Unique Students:",union)