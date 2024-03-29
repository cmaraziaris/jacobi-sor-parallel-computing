import os
from os.path import isfile, join
import re
from matplotlib import colors
import matplotlib.pyplot as plt
import sys 

# sort based on input and procs number
def get_details(x):
    details = x.split(".")[0].split("_")[2:]
    procs = int(details[0])
    input_size = int(details[1])
    convergence = "CONV" if details[-1] == "CONV" else "NO-CONV"
    return procs, input_size, convergence

#  get time record from given file path


def get_time_from_output(f):
    with open(f) as file:
        lines = file.readlines()
        return [l for l in lines if "MPI Wall time" in l][0].split(" ")[-1].strip()

# get all the times from the records


def get_all_times(out_files):
    times = {}
    for f in out_files:
        procs, in_size, conv = get_details(f)
        time = get_time_from_output(f)
        times[procs, in_size, conv] = round(float(time), 5)

    return times

#  get speedup


def get_speedup(t1, t0):
    return round(t0/t1, 3)


def get_all_speedup(t0s, t1s):
    speedup = {}
    for t1_k in t1s:
        procs, in_size, conv = t1_k
        s = get_speedup(t1s[t1_k], t0s[(1, in_size, conv)])
        print("Speed up for {} {}: {} / {} = {}".format(in_size,
              procs, t0s[(1, in_size, conv)], t1s[t1_k], s))
        speedup[procs, in_size, conv] = s
    return speedup


def get_efficiency(S, n):
    return round(S/n, 3)


def get_all_efficiency(speedup):
    efficiency = {}
    for procs, in_size, conv in speedup:
        e = get_efficiency(speedup[procs, in_size, conv], procs)
        efficiency[procs, in_size, conv] = e
    return efficiency


def print_metrics(metrics, title, unit_str):
    print("\n====PRINTING {}====".format(title))
    proc_cnts = sorted(list(set([x[0] for x in metrics])))
    sizes = sorted(list(set([x[1] for x in metrics])))

    for conv in ["CONV", "NO-CONV"]:
        print("\n"+conv, end='\n\n')
        print(" & ".join([""] + [str(i)
              for i in proc_cnts]), end=" \\\\ \\hline \\hline\n")
        for size in sizes:
            line = [str(metrics[i])+unit_str for i in metrics if size ==
                        i[1] and i[2] == conv]
            print(" & ".join([str(size)] + [str(i)
                  for i in line]), end=' \\\\ \\hline\n')


def print_graph(d, ax, fig, CONV):
    sizes = sorted(list(set([e[1] for e in d.keys()])))
    if CONV == "CONV":
        sizes = sizes[:-1]

    x = sorted(list(set([e[0] for e in d.keys()])))
    ys = [[d[i] for i in d if i[1] == size and i[2] == CONV]
           for size in sizes]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    for i, y in enumerate(ys):
        ax.plot(x, y, c=colors[i], label=str(sizes[i]))
        ax.set_xticks(x)
        ax.legend()
        

if __name__ == "__main__":
    program_type = sys.argv[1].lower() # challenge, mpi, hybrid 
    print("Program Type: ", program_type)
    if program_type not in ['mpi', 'challenge', 'hybrid']: 
        print("Allowed program types : mpi, challenge, hybrid")
        exit(1)

    # get all seq-runs output files
    out_re = re.compile("\AJ_seq_\w*\.o[0-9]*")
    seq_out_files = sorted(
                            [file for file in os.listdir() 
                                if isfile(file) and out_re.match(file)],
                            key=lambda x : get_details(x)
                        )
    #  get all parallel-runs output files
    out_re = re.compile("\AJ_{}_\w*\.o[0-9]*".format(program_type))
    par_out_files = sorted(
                            [file for file in os.listdir() 
                                if isfile(file) and out_re.match(file)],
                            key=lambda x : get_details(x)
                        )


    #  time benchmarks for both conv-check or not
    init_times = get_all_times(seq_out_files)
    times = get_all_times(par_out_files)

    # speed up in both cases
    speedup = get_all_speedup(init_times, times)

    # efficiency in bath cases
    efficiency = get_all_efficiency(speedup)

    print_metrics(times, "Times", "s")
    print_metrics(speedup, "Speedup", "")
    print_metrics(efficiency, "Efficiency", "")


    print(efficiency)
        
    # print graphs with allreduce use
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    plt.xlabel("Processes")
    ax[0][0].set_title("w\ Allreduce")
    ax[0][1].set_title("w\out Allreduce")
    
    print_graph(times, ax[0][0], fig, "CONV")
    ax[0][0].set_ylabel("Time")
    print_graph(speedup, ax[1][0], fig, "CONV")
    ax[1][0].set_ylabel("Speedup")
    print_graph(efficiency, ax[2][0], fig, "CONV")
    ax[2][0].set_ylabel("Efficiency")

    # print graphs without allreduce use
    print_graph(times, ax[0][1], fig, "NO-CONV")
    ax[0][0].set_ylabel("Time")
    print_graph(speedup, ax[1][1], fig, "NO-CONV")
    ax[1][0].set_ylabel("Speedup")
    print_graph(efficiency, ax[2][1], fig, "NO-CONV")
    ax[2][0].set_ylabel("Efficiency")

    plt.show()



