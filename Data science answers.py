#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Question 1: Nick's Check

def consecutive(Num, mark_list):
    mark_list.sort()
    for i in range(1, Num):
        if mark_list[i] != mark_list[i - 1] + 1:
            return 0
    return 1

Num = int(input('Number of elements '))

mark_list = []
for i in range(Num):
    mark_list.append(int(input()))
    
print('Output',consecutive(Num, mark_list))


# In[51]:


#Question2:Evaluate a Given Postfix Expression
def evaluate_postfix(expression):
    stack = []
    
    for token in expression.split():
        if token.isdigit():
            stack.append(int(token))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            if token == '+':
                result = operand1 + operand2
            elif token == '-':
                result = operand1 - operand2
            elif token == '*':
                result = operand1 * operand2
            elif token == '/':
                result = operand1 / operand2  # Assuming Python 3 behavior for division
            stack.append(result)
    
    return stack.pop()

expression = "8 4 -"
print("Result:", evaluate_postfix(expression))


# In[50]:


expression = "8 7 9 - 2 * +"
print("Result:", evaluate_postfix(expression))


# In[2]:


#Question3: Annual Day
def count_pairs(N, point):
    count = 0
    for i in range(N):
        for j in range(i,N):
            if point[i] > point[j]:
                count += 1
    return count

N = int(input('Number of elements '))

point = []
for i in range(N):
    point.append(int(input()))
print("Total count of pairs:", count_pairs(N, point))


# In[ ]:





# 

# In[ ]:




