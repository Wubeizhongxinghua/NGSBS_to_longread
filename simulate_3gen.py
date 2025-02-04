'''
Copyright (c) 2025-01-01 by LiMingyang, YiLab, Peking University.

Author: Li Mingyang (limingyang200101@gmail.com)

Institute: AAIS, Peking University

File Name: /gpfs3/chengqiyi_pkuhpc/limingyang/bert/tools/simulate_3gen.py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''
from rich import print, pretty
from rich.traceback import install
pretty.install()
install(show_locals=True)

import pysam
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import rich_click as click
from tqdm.auto import tqdm
import re
from icecream import ic
import os
import subprocess
import multiprocessing
from multiprocessing import Pool, cpu_count
import glob

# Function to reverse complement a DNA sequence
def reverse_complement(seq):
	"""
	Return the reverse complement of a DNA sequence.
	"""
	complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
	return ''.join([complement[base] for base in reversed(seq)])

# Function to extract CpG positions and methylation states from a read
def extract_cpg_states(read, reference_seq, read_start):
	"""
	Extract methylation states (0 = unmethylated, 1 = methylated) for CpG sites in a read.
	Accounts for bisulfite conversion, insertions, deletions, and reverse strand alignment.
	"""
	# Skip reads with soft clips
	# if any(op[0] == 4 for op in read.cigartuples):
		# return {}

	# Handle reverse strand reads
	if read.flag in [83, 163, 1]:
		seq = reverse_complement(read.query_sequence)
	else:
		seq = read.query_sequence

	ref_positions = read.get_reference_positions(full_length=True)
	cigar = read.cigartuples
	states = {}

	# Iterate through the read sequence and reference positions
	read_index = 0
	ref_index = read.reference_start

	for op, length in cigar: #有缺陷，refseq要延长一下，看看最后的是不是被截断的CpG
		if op == 4:  # Soft clip (already skipped)
			seq = seq[length:] if read_index == 0 else seq[:-length]
			# read_index += length
		elif op == 0:  # Match
			for i in range(length):
				if ref_positions[read_index] is not None:
					# Calculate the position in the fetched reference sequence
					ref_seq_index = ref_index - read_start
					if ref_seq_index >= 0 and ref_seq_index + 1 < len(reference_seq):
						# Check if the current position is part of a CpG site
						if reference_seq[ref_seq_index].upper() == "C" and reference_seq[ref_seq_index + 1].upper() == "G":
							# Determine methylation state
							if seq[read_index].upper() == 'C':
								states[ref_index] = 1  # Methylated
							elif seq[read_index].upper() == 'T':
								states[ref_index] = 0  # Unmethylated
				read_index += 1
				ref_index += 1
		elif op == 1:  # Insertion
			read_index += length
		elif op == 2:  # Deletion
			ref_index += length

	return states

# Global default dictionary initialization function
def cpg_data_initializer():
	return {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0,
			"0<-0": 0, "0<-1": 0, "1<-0": 0, "1<-1": 0,
			"0": 0, "1": 0}

# Function to process a single BAM chunk
def process_bam_chunk(args):
	bam_file, reference_genome, chromosome, start, end = args
	reference_genome = pysam.FastaFile(reference_genome)
	cpg_data = defaultdict(cpg_data_initializer)

	with pysam.AlignmentFile(bam_file, "rb") as bam:
		for read in bam.fetch(chromosome, start, end):
			if read.is_unmapped:
				continue
			ref_start = read.reference_start
			ref_end = read.reference_end
			reference_seq = reference_genome.fetch(chromosome, ref_start, ref_end + 1)  # extra base
			states = extract_cpg_states(read, reference_seq, ref_start)
			sorted_positions = sorted(states.keys())

			for i in range(len(sorted_positions)):
				curr_pos = sorted_positions[i]
				curr_state = states[curr_pos]

				if i > 0:
					prev_pos = sorted_positions[i - 1]
					prev_state = states[prev_pos]
					transition_key_prev = f"{prev_state}->{curr_state}"
					cpg_data[(chromosome, curr_pos, curr_pos + 1)][transition_key_prev] += 1

				if i < len(sorted_positions) - 1:
					next_pos = sorted_positions[i + 1]
					next_state = states[next_pos]
					transition_key_next = f"{curr_state}<-{next_state}"
					cpg_data[(chromosome, curr_pos, curr_pos + 1)][transition_key_next] += 1

				cpg_data[(chromosome, curr_pos, curr_pos + 1)][str(curr_state)] += 1

	return cpg_data

# Function to merge results from workers
def merge_cpg_data(results):
	merged_data = defaultdict(cpg_data_initializer)
	for result in results:
		for key, value in result.items():
			for subkey, subvalue in value.items():
				merged_data[key][subkey] += subvalue
	return merged_data

# Main function to build CpG table with multiprocessing
def build_cpg_table(bam_file_path, reference_genome_path, chunk_size=100_000_000):
	with pysam.AlignmentFile(bam_file_path, "rb") as bam:
		references = bam.references
		lengths = bam.lengths

	# Generate intervals for processing
	intervals = [
		(bam_file_path, reference_genome_path, chromosome, start, min(start + chunk_size, length))
		for chromosome, length in zip(references, lengths)
		for start in range(0, length, chunk_size)
	]

	# Use multiprocessing pool to process chunks
	with Pool(cpu_count()) as pool:
		results = list(tqdm(pool.map(process_bam_chunk, intervals), total=len(intervals)))

	# Merge results from all workers
	merged_data = merge_cpg_data(results)

	# Convert to DataFrame
	cpg_table = pd.DataFrame.from_dict(merged_data, orient="index").reset_index()
	cpg_table.rename(columns={"level_0": "chromosome", "level_1": "start", "level_2": "end"}, inplace=True)
	cpg_table.set_index(["chromosome", "start", "end"], inplace=True)

	return cpg_table.sort_index()

# Function to simulate long reads in random mode
def simulate_long_reads_random(cpg_table, read_length, num_reads, output_bam, input_bam, reference_genome):
	"""
	Simulate long reads in random mode using the CpG table and save to a BAM file.
	"""
	# Convert the table to a list of CpG sites
	# Initialize BAM file for writing with the same header as the input BAM file
	with pysam.AlignmentFile(input_bam, "rb") as in_bam:
		header = in_bam.header
		with pysam.AlignmentFile(output_bam, "wb", header=header) as out_bam:
			for turn in tqdm(range(num_reads)):
				# Start from the first CpG site in the table
				# i = 0
				# pbar = tqdm(total = len(cpg_table.index))
				# while i < len(cpg_table.index):
					# ic(i)
				i = random.choice(range(len(cpg_table)))
				chromosome, start, end = cpg_table.index[i]
				# Initialize the read
				states = {}
				prev_state = None
				seq_list = []
				# Process chunks until stopping condition is met
				if random.choice([0,1]): ############################ forward ####################################
					chunk_start = start
					chunk_end = start + read_length
					reference_seq = reference_genome.fetch(chromosome, chunk_start, chunk_end).upper()
					# Find all CpG sites in the 1 kb sequence
					cpg_positions = [m.start() + chunk_start for m in re.finditer("CG", reference_seq)]
					for cpg_pos in tqdm(cpg_positions, leave=False):
						# ic((chromosome, cpg_pos, cpg_pos + 1))
						if (chromosome, cpg_pos, cpg_pos + 1) not in cpg_table.index:
							# Stop if the CpG site is not in the table
							break
						# Check if the CpG site has any coverage
						if cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] == 0:
							# Stop if the CpG site has no coverage
							break
						# Determine the methylation state
						if prev_state is None:
							# Use the beta value for the first CpG
							beta = cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"])
							states[cpg_pos] = 1 if random.random() < beta else 0
						else:
							# Use the transition counts if available
							transition_counts = [
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"]
							]
							if sum(transition_counts) == 0:
								# Stop if no transition data is available
								break
							# Check for stopping conditions
							if prev_state == 0 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"] == 0:
								break
							if prev_state == 1 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"] == 0:
								break
							# Use the transition probabilities
							if prev_state == 0:
								states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"]) else 0
							else:
								states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"]) else 0
						prev_state = states[cpg_pos]
					# Generate the sequence for the current chunk
					for p in tqdm(range(chunk_start, cpg_pos), leave=False):
						if p in states:
							# CpG site: use C or T based on methylation state
							seq_list.append('C' if states[p] == 1 else 'T')
						else:
							# Non-CpG site: convert C to T
							ref_base = reference_seq[p - chunk_start]
							seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
					readstart = cpg_table.index[i][1]
				else: ########################## reverse #############################################
					chunk_start = end - read_length
					chunk_end = end+1
					reference_seq = reference_genome.fetch(chromosome, chunk_start, chunk_end).upper()
					# Find all CpG sites in the 1 kb sequence
					cpg_positions = [m.start() + chunk_start for m in re.finditer("CG", reference_seq)]
					cpg_positions.reverse()
					for cpg_pos in tqdm(cpg_positions, leave=False):
						# ic(cpg_pos)
						# ic((chromosome, cpg_pos, cpg_pos + 1))
						if (chromosome, cpg_pos, cpg_pos + 1) not in cpg_table.index:
							# Stop if the CpG site is not in the table
							break
						# Check if the CpG site has any coverage
						if cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] == 0:
							# Stop if the CpG site has no coverage
							break
						# Determine the methylation state
						if prev_state is None:
							# Use the beta value for the first CpG
							beta = cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"])
							states[cpg_pos] = 1 if random.random() < beta else 0
						else:
							# Use the transition counts if available
							transition_counts = [
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"],
								cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"]
							]
							if sum(transition_counts) == 0:
								# Stop if no transition data is available
								break
							# Check for stopping conditions
							if prev_state == 0 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"] == 0:
								break
							if prev_state == 1 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"] == 0:
								break
							# Use the transition probabilities
							if prev_state == 0:
								states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"]) else 0
							else:
								states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"]) else 0
						prev_state = states[cpg_pos]

					
					# Generate the sequence for the current chunk
					for p in tqdm(range(cpg_pos+1, chunk_end), leave=False):
						if p in states:
							# CpG site: use C or T based on methylation state
							seq_list.append('C' if states[p] == 1 else 'T')
						else:
							# Non-CpG site: convert C to T
							ref_base = reference_seq[p - chunk_start]
							seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
					readstart = cpg_pos + 1
			
				# Create a read from the merged sequence
				seq = ''.join(seq_list)
				read = pysam.AlignedSegment()
				read.query_name = f"simulated_read_turn_{turn}_pos_{i}"
				read.query_sequence = seq
				read.reference_id = out_bam.get_tid(chromosome)  # Chromosome index
				read.reference_start = readstart
				read.cigar = [(0, len(seq))]  # Assume no indels
				read.mapping_quality = 60
				# Add read to BAM file
				out_bam.write(read)



################################################################################################################
################################################################################################################
#################################### Sequential ################################################################
################################################################################################################
################################################################################################################


# Function to simulate long reads in sequential mode with merged chunks
def simulate_long_reads_sequential(cpg_table, read_length, num_turns, output_bam, input_bam, reference_genome):
	"""
	Simulate long reads in sequential mode using the CpG table and save to a BAM file.
	"""
	# Initialize BAM file for writing with the same header as the input BAM file
	with pysam.AlignmentFile(input_bam, "rb") as in_bam:
		header = in_bam.header
		with pysam.AlignmentFile(output_bam, "wb", header=header) as out_bam:
			for turn in tqdm(range(num_turns)):
				# Start from the first CpG site in the table
				i = 0
				pbar = tqdm(total = len(cpg_table.index))
				while i < len(cpg_table.index):
					# ic(i)
					chromosome, start, end = cpg_table.index[i]
					# Initialize the read
					states = {}
					prev_state = None
					seq_list = []
					
					# start = start - 100 # -100 from the first CpG site
					# Process chunks until stopping condition is met
					while True: # per 1kb processing
						# Fetch the reference sequence for the current 1 kb region
						is_stopped = 0
						chunk_start = start
						chunk_end = start + 1000
						reference_seq = reference_genome.fetch(chromosome, chunk_start, chunk_end).upper()

						# states, prev_state = inference_methylation_on_sequence(cpg_table, chromosome, chunk_start, chunk_end, reference_seq, states, prev_state, reverse=False)
						
						# Find all CpG sites in the 1 kb sequence
						cpg_positions = [m.start() + chunk_start for m in re.finditer("CG", reference_seq)]
						for cpg_pos in tqdm(cpg_positions, leave=False):
							# ic((chromosome, cpg_pos, cpg_pos + 1))
							if (chromosome, cpg_pos, cpg_pos + 1) not in cpg_table.index:
								# Stop if the CpG site is not in the table
								is_stopped = 1
								break
							# Check if the CpG site has any coverage
							if cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] == 0:
								# Stop if the CpG site has no coverage
								is_stopped = 1
								break
							# Determine the methylation state
							if prev_state is None:
								# Use the beta value for the first CpG
								beta = cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"])
								states[cpg_pos] = 1 if random.random() < beta else 0
							else:
								# Use the transition counts if available
								transition_counts = [
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"]
								]
								if sum(transition_counts) == 0:
									# Stop if no transition data is available
									is_stopped = 1
									break
								# Check for stopping conditions
								if prev_state == 0 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"] == 0:
									is_stopped = 1
									break
								if prev_state == 1 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"] == 0:
									is_stopped = 1
									break
								# Use the transition probabilities
								if prev_state == 0:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0->1"]) else 0
								else:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1->1"]) else 0
							prev_state = states[cpg_pos]


						
						# Generate the sequence for the current chunk
						if not is_stopped: #not stopped
							for p in tqdm(range(chunk_start, chunk_end), leave=False):
								if p in states:
									# CpG site: use C or T based on methylation state
									seq_list.append('C' if states[p] == 1 else 'T')
								else:
									# Non-CpG site: convert C to T
									ref_base = reference_seq[p - chunk_start]
									seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
							# Move to the next chunk
							start = chunk_end
						else: #stopped
							for p in tqdm(range(chunk_start, cpg_pos), leave=False):
								if p in states:
									# CpG site: use C or T based on methylation state
									seq_list.append('C' if states[p] == 1 else 'T')
								else:
									# Non-CpG site: convert C to T
									ref_base = reference_seq[p - chunk_start]
									seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
							break
						# # Check if stopping condition was met
						# if len(cpg_positions) == 0 or (chromosome, cpg_positions[-1], cpg_positions[-1] + 1) not in cpg_table.index:
							# break
					# Create a read from the merged sequence
					seq = ''.join(seq_list)
					read = pysam.AlignedSegment()
					read.query_name = f"simulated_read_turn_{turn}_pos_{i}"
					read.query_sequence = seq
					read.reference_id = out_bam.get_tid(chromosome)  # Chromosome index
					read.reference_start = cpg_table.index[i][1]
					read.cigar = [(0, len(seq))]  # Assume no indels
					read.mapping_quality = 60
					# Add read to BAM file
					out_bam.write(read)
					# Move to the next CpG site in the table
					i += len(states.keys())
					pbar.update(len(states.keys()))


def simulate_long_reads_sequential_rev(cpg_table, read_length, num_turns, output_bam, input_bam, reference_genome):
	"""
	Simulate long reads in sequential mode using the CpG table and save to a BAM file.
	"""
	# Initialize BAM file for writing with the same header as the input BAM file
	with pysam.AlignmentFile(input_bam, "rb") as in_bam:
		header = in_bam.header
		with pysam.AlignmentFile(output_bam, "wb", header=header) as out_bam:
			for turn in tqdm(range(num_turns)):
				# Start from the first CpG site in the table
				i = len(cpg_table.index) - 1
				pbar = tqdm(total = len(cpg_table.index))
				while i >= 0:
					# ic(i)
					chromosome, start, end = cpg_table.index[i]
					# Initialize the read
					states = {}
					prev_state = None
					seq_list = []
					final_seq = []
					# end = end + 100 # -100 from the first CpG site
					# Process chunks until stopping condition is met
					while True: # per 1kb processing
						# Fetch the reference sequence for the current 1 kb region
						is_stopped = 0
						chunk_start = end - 1000
						chunk_end = end+1
						# ic(chunk_start)
						# ic(chunk_end)
						reference_seq = reference_genome.fetch(chromosome, chunk_start, chunk_end).upper()

						# states, prev_state = inference_methylation_on_sequence(cpg_table, chromosome, chunk_start, chunk_end, reference_seq, states, prev_state, reverse=False)
						
						# Find all CpG sites in the 1 kb sequence
						cpg_positions = [m.start() + chunk_start for m in re.finditer("CG", reference_seq)]
						cpg_positions.reverse()
						# ic(cpg_positions)
						for cpg_pos in tqdm(cpg_positions, leave=False):
							# ic(cpg_pos)
							# ic((chromosome, cpg_pos, cpg_pos + 1))
							if (chromosome, cpg_pos, cpg_pos + 1) not in cpg_table.index:
								# Stop if the CpG site is not in the table
								is_stopped = 1
								break
							# Check if the CpG site has any coverage
							if cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] == 0:
								# Stop if the CpG site has no coverage
								is_stopped = 1
								break
							# Determine the methylation state
							if prev_state is None:
								# Use the beta value for the first CpG
								beta = cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1"])
								states[cpg_pos] = 1 if random.random() < beta else 0
							else:
								# Use the transition counts if available
								transition_counts = [
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"],
									cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"]
								]
								if sum(transition_counts) == 0:
									# Stop if no transition data is available
									is_stopped = 1
									break
								# Check for stopping conditions
								if prev_state == 0 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"] == 0:
									is_stopped = 1
									break
								if prev_state == 1 and cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"] == 0:
									is_stopped = 1
									break
								# Use the transition probabilities
								if prev_state == 0:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-0"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-0"]) else 0
								else:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"] / (cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "0<-1"] + cpg_table.loc[(chromosome, cpg_pos, cpg_pos + 1), "1<-1"]) else 0
							prev_state = states[cpg_pos]


						# Generate the sequence for the current chunk
						if not is_stopped: #not stopped
							for p in tqdm(range(chunk_start, chunk_end), leave=False):
								if p in states:
									# CpG site: use C or T based on methylation state
									seq_list.append('C' if states[p] == 1 else 'T')
								else:
									# Non-CpG site: convert C to T
									ref_base = reference_seq[p - chunk_start]
									seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
							# Move to the next chunk
							final_seq  = seq_list + final_seq
							end = chunk_start
						else: #stopped
							for p in tqdm(range(cpg_pos+1, chunk_end), leave=False):
								if p in states:
									# CpG site: use C or T based on methylation state
									seq_list.append('C' if states[p] == 1 else 'T')
								else:
									# Non-CpG site: convert C to T
									ref_base = reference_seq[p - chunk_start]
									seq_list.append('T' if ref_base.upper() == 'C' else ref_base)
							final_seq  = seq_list + final_seq
							break
						# # Check if stopping condition was met
						# if len(cpg_positions) == 0 or (chromosome, cpg_positions[-1], cpg_positions[-1] + 1) not in cpg_table.index:
							# break
					# Create a read from the merged sequence
					seq = ''.join(final_seq)
					read = pysam.AlignedSegment()
					read.query_name = f"simulated_read_turn_{turn}_pos_{i}_rev"
					read.query_sequence = seq
					read.reference_id = out_bam.get_tid(chromosome)  # Chromosome index
					read.reference_start = cpg_pos+1 # to 1-based
					read.cigar = [(0, len(seq))]  # Assume no indels
					read.mapping_quality = 60
					# Add read to BAM file
					out_bam.write(read)
					# Move to the next CpG site in the table
					i -= len(states.keys())
					pbar.update(len(states.keys()))


####################################################################################################################
####################################################################################################################
#################################################### Region Bed ####################################################
####################################################################################################################
####################################################################################################################
def process_bed_chunk(args):
	"""
	Process a chunk of BED lines to simulate reads and write to a temporary BAM file.
	"""
	bed_chunk, chunk_id, num_turns, cpg_table, input_bam, reference_genome = args
	temp_bam_file = f"temp_chunk_{chunk_id}.bam"
	reference_genome = pysam.FastaFile(reference_genome)
	try:
		with pysam.AlignmentFile(input_bam, "rb") as in_bam:
			header = in_bam.header
			with pysam.AlignmentFile(temp_bam_file, "wb", header=header) as out_bam:
				for turn in tqdm(range(num_turns)):
					for line in bed_chunk:
						chromosome, start, end = line.strip().split()[:3]
						start = int(start)
						end = int(end)
						# Initialize the read
						seq_list = []
						reference_seq = reference_genome.fetch(chromosome, start, end + 1).upper()
						cpg_positions = [m.start() + start for m in re.finditer("CG", reference_seq)]
						states = {}
						is_break = 0
						prev_state = None
						for cpg_pos in cpg_positions:
							cpg_index = (chromosome, cpg_pos, cpg_pos + 1)
							if cpg_index not in cpg_table.index:
								# Stop if the CpG site is not in the table
								is_break = 1
								break
							# Check if the CpG site has any coverage
							if cpg_table.loc[cpg_index, "0"] + cpg_table.loc[cpg_index, "1"] == 0:
								# Stop if the CpG site has no coverage
								is_break = 1
								break
							# Determine the methylation state
							if prev_state is None:
								# Use the beta value for the first CpG
								beta = cpg_table.loc[cpg_index, "1"] / (
										cpg_table.loc[cpg_index, "0"] +
										cpg_table.loc[cpg_index, "1"])
								states[cpg_pos] = 1 if random.random() < beta else 0
							else:
								# Use the transition counts if available
								transition_counts = [
									cpg_table.loc[cpg_index, "0->0"],
									cpg_table.loc[cpg_index, "0->1"],
									cpg_table.loc[cpg_index, "1->0"],
									cpg_table.loc[cpg_index, "1->1"]
								]
								if sum(transition_counts) == 0:
									# Stop if no transition data is available
									is_break = 1
									break
								# Check for stopping conditions
								if prev_state == 0 and cpg_table.loc[cpg_index, "0->0"] + cpg_table.loc[cpg_index, "0->1"] == 0:
									is_break = 1
									break
								if prev_state == 1 and cpg_table.loc[cpg_index, "1->0"] + cpg_table.loc[cpg_index, "1->1"] == 0:
									is_break = 1
									break
								# Use the transition probabilities
								if prev_state == 0:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[cpg_index, "0->1"] / (cpg_table.loc[cpg_index, "0->0"] + cpg_table.loc[cpg_index, "0->1"]) else 0
								else:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[cpg_index, "1->1"] / (cpg_table.loc[cpg_index, "1->0"] + cpg_table.loc[cpg_index, "1->1"]) else 0
							prev_state = states[cpg_pos]

						endsite = cpg_pos if is_break else cpg_pos+1
						for p in range(start, endsite):
							if p in states:
								# CpG site: use C or T based on methylation state
								seq_list.append('C' if states[p] == 1 else 'T')
							else:
								# Non-CpG site: convert C to T
								ref_base = reference_seq[p - start]
								seq_list.append('T' if ref_base.upper() == 'C' else ref_base)

						seq = ''.join(seq_list)
						read = pysam.AlignedSegment()
						read.query_name = f"simulated_read_turn_{turn}_pos_{str(start)}"
						read.query_sequence = seq
						read.reference_id = out_bam.get_tid(chromosome)
						read.reference_start = start  # Chromosome index
						read.cigar = [(0, len(seq))]
						read.mapping_quality = 60  # Assume no indels
						# Add read to BAM file
						out_bam.write(read)
	except Exception as e:
		print(f"Error processing chunk {chunk_id}: {e}")


def process_bed_chunk_rev(args):
	"""
	Process a chunk of BED lines to simulate reads and write to a temporary BAM file.
	"""
	bed_chunk, chunk_id, num_turns, cpg_table, input_bam, reference_genome = args
	temp_bam_file = f"temp_chunk_{chunk_id}.bam"
	reference_genome = pysam.FastaFile(reference_genome)
	try:
		with pysam.AlignmentFile(input_bam, "rb") as in_bam:
			header = in_bam.header
			with pysam.AlignmentFile(temp_bam_file, "wb", header=header) as out_bam:
				for turn in tqdm(range(num_turns)):
					for line in bed_chunk:
						chromosome, start, end = line.strip().split()[:3]
						start = int(start)
						end = int(end)
						seq_list = []
						reference_seq = reference_genome.fetch(chromosome, start, end + 1).upper()
						cpg_positions = [m.start() + start for m in re.finditer("CG", reference_seq)]
						cpg_positions.reverse()
						states = {}
						is_break = 0
						prev_state = None
						for cpg_pos in cpg_positions:
							cpg_index = (chromosome, cpg_pos, cpg_pos + 1)
							if cpg_index not in cpg_table.index:
								is_break = 1
								break
							if cpg_table.loc[cpg_index, "0"] + cpg_table.loc[cpg_index, "1"] == 0:
								is_break = 1
								break
							if prev_state is None:
								beta = cpg_table.loc[cpg_index, "1"] / (
										cpg_table.loc[cpg_index, "0"] +
										cpg_table.loc[cpg_index, "1"])
								states[cpg_pos] = 1 if random.random() < beta else 0
							else:
								transition_counts = [
									cpg_table.loc[cpg_index, "0<-0"],
									cpg_table.loc[cpg_index, "0<-1"],
									cpg_table.loc[cpg_index, "1<-0"],
									cpg_table.loc[cpg_index, "1<-1"]
								]
								if sum(transition_counts) == 0:
									is_break = 1
									break
								if prev_state == 0 and cpg_table.loc[cpg_index, "0<-0"] + cpg_table.loc[cpg_index, "1<-0"] == 0:
									is_break = 1
									break
								if prev_state == 1 and cpg_table.loc[cpg_index, "0<-1"] + cpg_table.loc[cpg_index, "1<-1"] == 0:
									is_break = 1
									break
								# Use the transition probabilities
								if prev_state == 0:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[cpg_index, "1<-0"] / (cpg_table.loc[cpg_index, "0<-0"] + cpg_table.loc[cpg_index, "1<-0"]) else 0
								else:
									states[cpg_pos] = 1 if random.random() < cpg_table.loc[cpg_index, "1<-1"] / (cpg_table.loc[cpg_index, "0<-1"] + cpg_table.loc[cpg_index, "1<-1"]) else 0
							prev_state = states[cpg_pos]
						if cpg_pos == cpg_positions[0]:
							continue
						startsite = cpg_pos+1 if is_break else cpg_pos
						for p in range(startsite, end):
							if p in states:
								seq_list.append('C' if states[p] == 1 else 'T')
							else:
								ref_base = reference_seq[p - start]
								seq_list.append('T' if ref_base.upper() == 'C' else ref_base)

						seq = ''.join(seq_list)
						read = pysam.AlignedSegment()
						read.query_name = f"simulated_read_turn_{turn}_pos_{cpg_pos+1}_rev"
						read.query_sequence = seq
						read.reference_id = out_bam.get_tid(chromosome)
						read.reference_start = startsite
						read.cigar = [(0, len(seq))]
						read.mapping_quality = 60
						out_bam.write(read)
	except Exception as e:
		print(f"Error processing chunk {chunk_id}: {e}")


def simulate_long_reads_regionbed(cpg_table, num_turns, output_bam, input_bam, reference_genome, bed_file):
	"""
	Simulate long reads using multi-processing and consolidate results into a single BAM file.
	"""
	with open(bed_file, 'r') as bed:
		bed_lines = bed.readlines()

	num_chunks = cpu_count()
	chunk_size = len(bed_lines) // num_chunks
	bed_chunks = [bed_lines[i:i + chunk_size] if i + chunk_size < len(bed_lines) else bed_lines[i:] 
				  for i in range(0, len(bed_lines), chunk_size)]

	pool = Pool(processes=num_chunks)
	args = [(chunk, i, num_turns, cpg_table, input_bam, reference_genome) for i, chunk in enumerate(bed_chunks)]

	with tqdm(total=len(bed_chunks), desc="Processing BED chunks") as pbar:
		for _ in pool.imap_unordered(process_bed_chunk, args):
			pbar.update()

	pool.close()
	pool.join()

	temp_bam_files = glob.glob(f"temp_chunk_*.bam")
	pysam.merge("-f", f"fwd_{output_bam}", *temp_bam_files)
	for temp_file in temp_bam_files:
		os.remove(temp_file)
		
	# Reversely simulate reads
	pool = Pool(processes=num_chunks)
	args = [(chunk, i, num_turns, cpg_table, input_bam, reference_genome) for i, chunk in enumerate(bed_chunks)]

	with tqdm(total=len(bed_chunks), desc="Processing BED chunks") as pbar:
		for _ in pool.imap_unordered(process_bed_chunk_rev, args):
			pbar.update()

	pool.close()
	pool.join()

	temp_bam_files = glob.glob(f"temp_chunk_*.bam")
	pysam.merge("-f", f"rvs_{output_bam}", *temp_bam_files)
	for temp_file in temp_bam_files:
		os.remove(temp_file)


def postprocess(output_bam, mode):
	if mode == 'random':
		subprocess.run(f"samtools sort -@ 20 -o {output_bam} unsort_{output_bam}".split(' '))
		subprocess.run(f"rm -rf unsort_{output_bam}".split(' '))
		subprocess.run(f"samtools index -@ 20 {output_bam}".split(' '))
	else:
		subprocess.run(f"samtools merge -@ 20 -f unsort_{output_bam} fwd_{output_bam} rvs_{output_bam}".split(' '))
		subprocess.run(f"rm -rf fwd_{output_bam} rvs_{output_bam}".split(' '))
		subprocess.run(f"samtools sort -@ 20 -o {output_bam} unsort_{output_bam}".split(' '))
		subprocess.run(f"rm -rf unsort_{output_bam}".split(' '))
		subprocess.run(f"samtools index -@ 20 {output_bam}".split(' '))

# Main function with click for parameter handling
@click.command()
@click.option("--bam_file", required=True, help="Input BAM file containing short-read BS-seq data.")
@click.option("--cpgtable_file", required=True, help="CpG transition table. If file does not exist, it will create; if existed, it will read.")
@click.option("--is_only_cpgtable", is_flag=True, help="Whether only conduct generating the cpg table.")
@click.option("--reference_genome", required=True, help="Reference genome file in FASTA format.")
@click.option("--read_length", type=int, required=False, default=None, show_default=True, help="[Random Mode] Length of simulated long reads.")
@click.option("--num_reads", type=int, required=False, default=None, show_default=True, help="[Random Mode] Number of reads to simulate (random mode).")
@click.option("--num_turns", type=int, required=False, default=None, show_default=True, help="[Sequential or Region bed Mode] Number of turns to simulate (sequential mode).")
@click.option("--bed_file", required=False, default=None, show_default=True, help="[Region bed Mode] Bed recording region for generating sequences.")
@click.option("--mode", type=click.Choice(['random', 'sequential','regionbed']), required=True, help="Simulation mode: random or sequential.")
@click.option("--output_bam", required=True, help="Output BAM file for simulated long reads.")
# @click.option("--output_cpgtable", required=True, help="Output tabular recording methylation state transition.")
def main(bam_file, reference_genome, read_length, num_reads, num_turns, mode, output_bam, cpgtable_file, bed_file, is_only_cpgtable):
	reference_genome_path = reference_genome # Use path instead of pysam.FastaFile object to allow for parallel processing
	# Load the reference genome
	reference_genome = pysam.FastaFile(reference_genome)

	# Step 1: Build the CpG table
	if not os.path.exists(cpgtable_file): 
		print("CpG table does not exist! Building CpG table...")
		cpg_table = build_cpg_table(bam_file, reference_genome_path)
		print(cpg_table.head(10))
		cpg_table.reset_index().to_csv(cpgtable_file, index=False, header=True, sep='\t')
	else:
		print("CpG table exists! Reading...")
		cpg_table = pd.read_table(cpgtable_file, header=0, sep='\t').set_index(['chromosome','start','end'])
		print(cpg_table.head(10))
	if not is_only_cpgtable:
		# Step 2: Simulate long reads
		print("Simulating long reads...")
		if mode == 'random':
			print("Random mode...")
			simulate_long_reads_random(cpg_table, read_length, num_reads, f"unsort_{output_bam}", bam_file, reference_genome)
			postprocess(output_bam, mode)
		elif mode == 'sequential':
			print("Sequential mode...")
			simulate_long_reads_sequential(cpg_table, read_length, num_turns, f"fwd_{output_bam}", bam_file, reference_genome)
			simulate_long_reads_sequential_rev(cpg_table, read_length, num_turns, f"rvs_{output_bam}", bam_file, reference_genome)
			postprocess(output_bam, mode)
		elif mode == 'regionbed':
			print("Region bed mode...")
			simulate_long_reads_regionbed(cpg_table, num_turns, output_bam, bam_file, reference_genome_path, bed_file)
			postprocess(output_bam, mode)
	
		print(f"Simulated long reads saved to {output_bam}")

if __name__ == "__main__":
	main()
