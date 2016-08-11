from constants import entropy_beta
import numpy as np

def print_loss_components(a,td,variance,mean,r,v):
	entropy = -0.5*np.log(2*3.14*variance) + 1
	print "Entropy: ", entropy*entropy_beta
	print "Policy loss (L2):", np.sum((mean - a) ** 2) / 2
	print "Policy loss (main): ", (np.sum((mean - a) ** 2) / 2)*td

	# Learning rate for critic is half of actor's, so multiply by 0.5
	value_loss = 0.5 * (np.sum((r - v) ** 2) / 2)
	print "Value loss: ", value_loss