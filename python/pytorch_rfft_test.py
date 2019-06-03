import numpy as np
import torch

def contiguous_clone(X):
    if X.is_contiguous():
        return X.clone()
    else:
        return X.contiguous()

def backward(grad_output_re, grad_output_im):
    # Clone the array and make contiguous if needed
    grad_output_re = contiguous_clone(grad_output_re)
    grad_output_im = contiguous_clone(grad_output_im)

    if _to_save_input_size & 1:
        grad_output_re[...,1:] /= 2
    else:
        grad_output_re[...,1:-1] /= 2

    if _to_save_input_size & 1:
        grad_output_im[...,1:] /= 2
    else:
        grad_output_im[...,1:-1] /= 2

    # gr = irfft(grad_output_re,grad_output_im,_to_save_input_size, normalize=False)
    return gr
