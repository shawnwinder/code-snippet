# 1: normal lamba
'''
g = lambda x : x**4
print(g(2))
'''

# 2: lambda with function
'''
def foo(x, k):
    print("I am in foo")
    print("I may be called by a lambda")
    return x + k

g = lambda x : foo(x, 3)
print(g(2))
'''

# 3: return None
'''
def foo(x, k):
    print("I am in foo")
    print("I may be called by a lambda")

g = lambda x : foo(x, 3)
print(g(2))
'''
