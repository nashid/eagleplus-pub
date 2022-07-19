import argparse
import pickle
import glob, os
import tensorflow as tf

def check_dir_exist(d):
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise argparse.ArgumentTypeError("%s is not a valid work directory" % d)
    return d

def print_clusters(inputdir):
	for c in sorted(glob.glob1(inputdir, "cluster_*")):
		with open(c) as cfile:
			print("==== %s  ====" % c)
			for line in cfile:
				with open(line.rstrip('\n'), "r") as efile:
					exception_msg = efile.read()
					print(exception_msg)
		#print('\n')

def print_pairs(inputdir):
	for c in sorted(glob.glob1(inputdir, "cluster_*")):
		with open(c) as f:
			print("==== INPUT-OUTPUT Pair for %s  ====" % c)
			first_efile = f.readline().rstrip('\n')
			assert first_efile.endswith('.e')
			first_pfile = first_efile[:-2] + '.p'
			with open(inputdir + '/' + first_pfile, 'rb') as pfile:
				p_data = pickle.load(pfile)
				print("---- INPUT for %s ---- " % c)
				print(p_data)			
			with open(inputdir + '/' + first_efile, "r") as efile:
				e_data = efile.read()
				print("---- OUTPUT for %s ---- " % c)
				print(e_data)
			

def print_pickled_input(input):
	print("==== Pickled input : %s  ====" % input)
	with open(input, 'rb') as pfile:
		data = pickle.load(pfile)
		print(data)

if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--inputdir', type=check_dir_exist, help='Input directory that stores *.p, *.e, cluster_*, etc.')
	parser.add_argument('--printpairs', default=False, action='store_true', help='Flag whether to print input-output pairs for each cluster')
	parser.add_argument('--pfile', default=None, help='File name of one p file to be unpickled')
	args = parser.parse_args()
	inputdir = args.inputdir
	printpairs = args.printpairs
	pfile = args.pfile
	
	os.chdir(inputdir)
	if pfile is None:
		if printpairs:
			print_pairs(inputdir)
		else:
			print_clusters(inputdir)

	else:
		print_pickled_input(inputdir + '/' + pfile)
