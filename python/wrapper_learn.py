import time
import sys



def timeit(func):
    def wrapper():
        print("Start running {}".format(func.__name__))
        start_time = time.time()
        func()
        end_time = time.time()
        print("Run {} over, total time is {:.6f}s".format(func.__name__,
                                                      end_time - start_time))
    return wrapper

@timeit
def foo():
    print("this is foo() running")
    time.sleep(1)

def bar():
    print("this is bar() running")



if __name__ == '__main__':
    foo()
