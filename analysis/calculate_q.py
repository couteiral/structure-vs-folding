import os
import sys
import argparse
import numpy as np
import pandas as pd
import MDAnalysis as MDA
import dill as pickle
import pathos.multiprocessing as mp
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
from MDAnalysis.analysis.distances import self_distance_array

stats = rpackages.importr('stats')
rpy2.robjects.numpy2ri.activate()


def load_stride(stride_file):
    """Load the segments of secondary structure
    defined on a STRIDE file. Considers only segments
    with at least four amino acids."""

    counters = {
        'AlphaHelix': 0,
        '310Helix': 0,
        'Strand': 0
    }
    ss_elements = {}

    found_offset = False

    with open(stride_file, 'r') as f:
        for line in f:
            if line[:3] == 'LOC':
                if line.split()[1] == 'Disulfide': continue
                _, kind, _, i, _, _, j, _, _ = line.split()
                i, j = int(i), int(j)
                if kind in counters.keys() and abs(i-j) >= 4:
                    ss_id = f'{kind}_{letters[counters[kind]]}'
                    counters[kind] += 1
                    ss_elements[ss_id] = (i, j)

            if not found_offset and line[:3] == 'ASG':
                found_offset = True
                offset = int(line.split()[3])
                print(offset)

    ss_elements = {k: (x-offset, y-offset) for k, (x, y) in ss_elements.items()}

    return ss_elements

def intra_segment_contacts(seg, data, protein):
    """Returns the number of intra    segment contacts
    for a particular secondary structure segment."""

    traj = []
    for frame in data.trajectory:
        coor = protein.positions
        contacts = 0
        for i in range(seg[0], seg[1]+1):
            for j in range(i+1, seg[1]+1):
                if abs(j-i) > 3:
                    d = np.linalg.norm(coor[j, :]-coor[i, :])
                    if d <= dist[i, j] * adjust and dist[i, j] != 0:
                            contacts += 1
        traj.append(contacts)
    return np.array(traj)

def isinss(i, ss_elements):
    """Returns True if the ith aminoacid forms
    part of any secondary structure element in
    ss_elements."""

    for _, ss_element in ss_elements.items():
        if _isinss(i, ss_element):
            return True
    return False

def _isinss(i, ss_element):
    """Returns True if the ith aminoacid
    forms part of the secondary structure
    element provided."""
    return ss_element[0] <= i and i <= ss_element[1]

def inter_segment_contacts(seg1, seg2, data, protein, dist):
    """Returns the number of inter    segment contacts
    for a pair of secondary structure elements."""

    traj = []
    for frame in data.trajectory:
        coor = protein.positions
        contacts = 0
        for i in range(seg1[0], seg1[1]+1):
            for j in range(seg2[0], seg2[1]+1):
                if abs(j-i) > 3:
                    d = np.linalg.norm(coor[j, :]-coor[i, :])
                    if d <= dist[i, j] * adjust and dist[i, j] != 0:
                            contacts += 1
        traj.append(contacts)
    return np.array(traj)

def smooth(y):
    x = np.linspace(0, 1, y.shape[0])
    x, y = stats.supsmu(x, y)
    return y

def process_all(packed_input):

    dcd, top, ref, stride, output = packed_input

    # Read data
    data = MDA.Universe(top, dcd)
    protein = data.select_atoms('(resname GLY and name CA) or (not resname GLY and name CB)')
    reference = MDA.Universe(ref, ref)
    reference = reference.select_atoms('(resname GLY and name CA) '
        'or (not resname GLY and name CB)')
    n_residues = reference.positions.shape[0]
    ss_elements = load_stride(stride)

    # Define contacts
    dist_ = self_distance_array(reference.positions)
    dist = np.zeros([n_residues, n_residues])
    mask = np.zeros([n_residues, n_residues])
    k = 0
    for i in range(n_residues):
        for j in range(i+1, n_residues):
            dist[i, j] = dist[j, i] = dist_[k]
            k += 1

            if isinss(i, ss_elements) and isinss(j, ss_elements) and abs(j-i) > 3:
                mask[i, j] = mask[j, i] = 1. if dist[i, j] <= min_dist else 0.

    dist = np.multiply(mask, dist)

    ## Compute intra    segment contacts
    #intra_segment = {}
    #for name, seg in ss_elements.items():
    #    intra_segment[name] = intra_segment_contacts(seg, data, protein) \
    #        / mask[seg[0]:seg[1]+1, seg[0]:seg[1]+1].sum()

    #print(intra_segment)

    # Compute inter    segment contacts
    inter_segment = {}
    names = list(ss_elements.keys())
    for k, name1 in enumerate(names):
        for name2 in names[k+1:]:
            seg1, seg2 = ss_elements[name1], ss_elements[name2]
            n_contacts = mask[seg1[0]:seg1[1]+1, seg2[0]:seg2[1]+1].sum()
            if n_contacts == 0: continue

            inter_segment[f'{name1}-{name2}'] = inter_segment_contacts(seg1, seg2, data, protein, dist) \
                / n_contacts 

    df = pd.DataFrame({k: np.array(smooth(v)).tolist() for k, v in inter_segment.items()})
    df.to_csv(output, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dcd_file')
    parser.add_argument('top_file')
    parser.add_argument('ref_file')
    parser.add_argument('stride_file')
    parser.add_argument('output_file', default='2fnc.csv')
    parser.add_argument('--min_dist', default=8.0)
    parser.add_argument('--adjust', default=1.2)
    parser.add_argument('--n_proc', default=None)
    args = parser.parse_args()

    global min_dist
    global adjust
    min_dist = float(args.min_dist)
    adjust = float(args.adjust)

    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    with open(args.dcd_file, 'r') as f:
        dcd_inputs = [x.strip() for x in f.readlines()]
    with open(args.top_file, 'r') as f:
        top_inputs = [x.strip() for x in f.readlines()]
    with open(args.ref_file, 'r') as f:
        ref_inputs = [x.strip() for x in f.readlines()]
    with open(args.stride_file, 'r') as f:
        stride_inputs = [x.strip() for x in f.readlines()]
    with open(args.output_file, 'r') as f:
        output_inputs = [x.strip() for x in f.readlines()]

    n_proc = None if args.n_proc is None else int(args.n_proc)
    pool = mp.Pool(n_proc)
    input = list(zip(dcd_inputs, top_inputs, ref_inputs, stride_inputs, output_inputs))
    pool.map(process_all, input)
