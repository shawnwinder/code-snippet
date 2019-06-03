from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import os
import sys
import time



def foo(*args, **kwargs):
    print("="*30)
    print('args = {}'.format(args))
    print('kwargs = {}'.format(kwargs))


def bar(name, *args, **kwargs):
    print("name: {}".format(name))
    new_a = kwargs.get('a', 100) + 1
    print("new_a: {}".format(new_a))

def f1(model, **kwargs):
    print("model is: {}".format(model))

if __name__ == "__main__":
    # test 1
    '''
    foo(1, 2, 3, 4)
    foo(a=1, b=2, c=3)
    foo(1, 2, 3, a=1, b=2, c=3)
    foo('a', 1, None, a=1, b='2', c=np.arange(5))
    '''

    # test 2
    # bar('shawn', b=2)

    # test 3
    f1("ResNet50")


