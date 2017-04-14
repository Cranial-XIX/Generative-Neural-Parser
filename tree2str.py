import sys
import os

inp = sys.argv[1]

if os.path.isdir(inp):
	for file in os.listdir(inp):
		with open(os.path.join(inp, file), 'r') as input:
			lines = input.readlines()
			string = ""
			for i in xrange(len(lines)):
				string = string + " " + lines[i].strip()
				if ".)" in lines[i]:
					print string.strip()
					string = ""
else:
	with open(inp, 'r') as input:
		lines = input.readlines()
		string = ""
		for i in xrange(len(lines)):
			string = string + " " + lines[i].strip()
			if ".)" in lines[i]:
				print string.strip()
				string = ""




