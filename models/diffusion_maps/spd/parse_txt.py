import re
import numpy as np
import collections
import matplotlib.pyplot as plt

RESULT_DIR = "Results"

num_procs = 100
rank_end_component_arr = np.zeros(num_procs)
rank_end_iter_arr = np.zeros(num_procs)

def read_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            regex = re.findall(r'\d+(?:\.\d+)?', line)
            if(len(regex)==4):
                regex = [float(d) for d in regex]
                # print(line)
                # print(regex)
                rank_end_component_arr[int(regex[0])] = int(regex[2])
                rank_end_iter_arr[int(regex[0])] = int(regex[1])
            # get rank and iteration and number of components
            # start = iter+1

read_txt('slurm-1812174.out')
read_txt('slurm-1814917.out')

# # rank_end_component_arr : a
np.save("comp_arr",rank_end_component_arr)
# # rank_end_iter_arr : b
np.save("iter_arr",rank_end_iter_arr)


# -------------------------------------------------------------

# def read_error_dict(file, error_dict):

#     with open(file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             regex = re.findall(r'\d+(?:\.\d+)?', line)
#             if(len(regex)==4):
#                 regex = [float(d) for d in regex]
#                 # print(regex)
#                 comp = int(regex[2])
#                 if comp in error_dict:
#                     error_dict[comp].append(float(regex[3]))
#                 else:
#                     error_dict[comp] = [float(regex[3])]

# error_dict = dict()
# read_error_dict('slurm-1781078.out', error_dict)
# read_error_dict('slurm-1783744.out', error_dict)
# read_error_dict('slurm-1785672.out', error_dict)
# read_error_dict('slurm-1790054.out', error_dict)

# error_dict = collections.OrderedDict(sorted(error_dict.items()))

# error_arr = []
# for key, value in error_dict.items():
#     error_arr += value

# N = len(error_arr)
# np.save(RESULT_DIR+"/train_error_v_m",error_arr)

# fig, ax = plt.subplots(1,1,figsize=[15,4])
# plt.plot(range(N), error_arr)
# plt.xlabel("number of dmap components")
# plt.ylabel("reconstruction train mse error")
# plt.yscale("log")
# plt.savefig(RESULT_DIR+"/train_recon_v_m.png")

# ------------------------------------------------------
# def read_txt_2_dict(file, components):

#     with open(file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             regex = re.findall(r'\d+(?:\.\d+)?', line)
#             if(len(regex)==4):
#                 regex = [float(d) for d in regex]
#                 rank = int(regex[0])
#                 if rank in components:
#                     components[rank].append(int(regex[2]))
#                 else:
#                     components[rank] = [int(regex[2])]

# components = dict()
# read_txt_2_dict('slurm-1781078.out', components)
# read_txt_2_dict('slurm-1783744.out', components)
# read_txt_2_dict('slurm-1785672.out', components)
# read_txt_2_dict('slurm-1790054.out', components)

# components = collections.OrderedDict(sorted(components.items()))

# com = []
# for key, value in components.items():
#     com += value

# com.sort()

# def find_missing(lst):
#     max = lst[0]
#     for i in lst :
#         if i > max :
#             max= i

#     min = lst [0]
#     for i in lst :
#         if i < min:
#             min = i
#     missing = max+1
#     list1=[]

#     for _ in lst :
#         max = max -1
#         if max not in lst :
#             list1.append(max)

#     return list1

# # print(find_missing(com))

# for i,j in zip(com, range(10000)):
#     if i!=j:
#         print(i,j)


