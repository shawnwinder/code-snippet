import torch


# formular
'''
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
#
# start.record()
# some CPU bound operations, i.e. loading data...
# end.record()
#
# # Waits for everything to finish running
# torch.cuda.synchronize()
#
# print(start.elapsed_time(end))
'''

# example
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = x + y
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))

