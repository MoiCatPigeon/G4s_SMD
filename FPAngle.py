#!/usr/bin/python3

'''
Used for calculating the force-quartet plane angle in G4s SMD simulations.
'''

import MDAnalysis as mda
import numpy as np
import math

# Adjustable parameters for G4s and input file names
Qnum = 5
Fdirection = "1-7"
gro = "NOWAT.gro"
xtc = "tNOWAT.xtc"

# Tune the atom selections based on Q number
sel_anchor_top = "resid 1 and (name C2 or name C4 or name C6)"
if Qnum == 3:
	sel_topQ = "nucleicbase and resid 3 8 13 18"
	sel_botQ = "nucleicbase and resid 5 10 15 20"
	if Fdirection == "1-6":
		sel_anchor_bot = "resid 21 and (name C2 or name C4 or name C6)"
	elif Fdirection == "1-7":
		sel_anchor_bot = "resid 16 and (name C2 or name C4 or name C6)"
	else:
		sel_anchor_bot = "resid 11 and (name C2 or name C4 or name C6)"
elif Qnum == 4:
	sel_topQ = "nucleicbase and resid 3 9 15 21"
	sel_botQ = "nucleicbase and resid 6 12 18 24"
	if Fdirection == "1-6":
		sel_anchor_bot = "resid 25 and (name C2 or name C4 or name C6)"
	elif Fdirection == "1-7":
		sel_anchor_bot = "resid 19 and (name C2 or name C4 or name C6)"
	else:
		sel_anchor_bot = "resid 13 and (name C2 or name C4 or name C6)"
else:
	sel_topQ = "nucleicbase and resid 3 10 17 24"
	sel_botQ = "nucleicbase and resid 7 14 21 28"
	if Fdirection == "1-6":
		sel_anchor_bot = "resid 29 and (name C2 or name C4 or name C6)"
	elif Fdirection == "1-7":
		sel_anchor_bot = "resid 22 and (name C2 or name C4 or name C6)"
	else:
		sel_anchor_bot = "resid 15 and (name C2 or name C4 or name C6)"

def norm(A):
	# Calculate norm by Singular Value Decomposition
	# Get the center of mass of the quartet atoms
	com = np.mean(A,axis=0,keepdims = True)
	
	# Singular Value Decomposition, and get matrix U
	svd = np.linalg.svd(np.transpose(A - com), full_matrices = False)
	u   = svd[0]
	norm = u[:,-1]
	#nnorm = (A - com) @ norm
	
	return norm

def fdirection(top,bot):
	# Calculate the vector of spring direction
	# Spring anchored points defined by the T base C2, C4 and C6 atoms center
	
	top_anchor = np.mean(top,axis=0)
	bot_anchor = np.mean(bot,axis=0)
	
	# Returning 1x3 vector [x,y,z]
	return top_anchor - bot_anchor

def fpangle(qnorm,fd):
	# Calculate the spring-plane angle
	# 90 degree - force-norm angle
	
	angle = math.pi / 2 - math.acos(abs(np.dot(fd,qnorm)) / (np.linalg.norm(fd) * np.linalg.norm(qnorm)))
	return angle

if __name__ == "__main__":	
	
	traj = mda.Universe(gro, xtc, in_memory=True)
	
	start = 0
	
	# Extract the nucleic base coordinates from the top/bot quartets
	# Then extract the anchored point coordinates
	# No H coordinates
	for frame in traj.trajectory:
		# Quartet bases coordinates
		topQ = traj.select_atoms(sel_topQ).positions
		botQ = traj.select_atoms(sel_botQ).positions
		
		# Anchored points coordinates
		anchor_top = traj.select_atoms(sel_anchor_top).positions
		anchor_bot = traj.select_atoms(sel_anchor_bot).positions
		
		if start == 0:
			# If it is the first read/frame
			all_topQ = topQ
			all_botQ = botQ
			
			all_anchor_top = anchor_top
			all_anchor_bot = anchor_bot
		else:
			all_topQ = np.dstack((all_topQ,topQ))
			all_botQ = np.dstack((all_botQ,botQ))
			
			all_anchor_top = np.dstack((all_anchor_top,anchor_top))
			all_anchor_bot = np.dstack((all_anchor_bot,anchor_bot))
		start += 1
	
	all_angle = []
	#topQ.write('traj.xyz', frames = 'all')
	for frame in range(np.shape(all_topQ)[2]):
		# np.savetxt("traj" + str(frame) + ".dat",all_topQ[:,:,frame])
	
		# Calculate the norms of top and bot quartet planes
		tnorm = norm(all_topQ[:,:,frame])
		bnorm = norm(all_botQ[:,:,frame])
		
		if np.dot(tnorm,bnorm) < 0:
			bnorm = -1 * bnorm
		
		# Average the top and bot norms to the quartets norm
		qnorm = (tnorm + bnorm) / np.linalg.norm(tnorm + bnorm)
		
		# Obtain the force direction vector
		fd = fdirection(all_anchor_top[:,:,frame], all_anchor_bot[:,:,frame])
		
		# Calculate the angle between the spring and the quartet plane
		angle = fpangle(qnorm,fd)
		all_angle.append(str(angle) + "\n")
		#np.savetxt("tnorm" + str(frame) + ".dat", qnorm)
	
	with open("fpangle.dat", 'w') as op:
		op.writelines(all_angle)
	
	
	
	
