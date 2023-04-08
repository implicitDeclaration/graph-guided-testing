import os
import sys
sys.path.append('../')
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from attacks.attack_type.differential_evolution import differential_evolution


class onepixel(object):

	def __init__(self,model,pixels=3,maxiter=100,popsize=400,target=False,verbose=True,device=None):

		self.model = model
		self.pixels = pixels
		self.maxiter = maxiter
		self.popsize = popsize
		self.target = target
		self.verbose = verbose
		self.device = device

	def perturb_image(self,xs, img):
		if xs.ndim < 2:
			xs = np.array([xs])
		batch = len(xs)
		imgs = img.repeat(batch, 1, 1, 1)
		xs = xs.astype(int)

		count = 0
		for x in xs:
			pixels = np.split(x, len(x)/5)

			for pixel in pixels:
				x_pos, y_pos, r, g, b = pixel
				imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
				imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
				imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
			count += 1

		return imgs

	def predict_classes(self,xs, img, target_calss, minimize=True):
		imgs_perturbed = self.perturb_image(xs, img.clone())
		with torch.no_grad():
			input = Variable(imgs_perturbed).to(self.device)
		predictions = F.softmax(self.model(input),dim=1).data.cpu().numpy()[:, target_calss]


		return predictions if minimize else 1 - predictions

	def attack_success(self,x, img, target_calss, targeted_attack=False, verbose=False):

		attack_image = self.perturb_image(x, img.clone())
		with torch.no_grad():
			input = Variable(attack_image).to(self.device)
		confidence = F.softmax(self.model(input),dim=1).data.cpu().numpy()[0]
		predicted_class = np.argmax(confidence)

		if (verbose):
			print ("Confidence: %.4f"%confidence[target_calss])
		if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
			return True


	def attack(self,img, label):
		# img: 1*3*W*H tensor
		# label: a number

		targeted_attack = self.target is not None
		target_calss =  label


		bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * self.pixels

		popmul = max(1, self.popsize/len(bounds))

		predict_fn = lambda xs: self.predict_classes(
			xs, img, target_calss, self.target is None)
		callback_fn = lambda x, convergence: self.attack_success(
			x, img, target_calss,targeted_attack, self.verbose)

		inits = np.zeros([int(popmul*len(bounds)), len(bounds)])
		for init in inits:
			for i in range(self.pixels):
				init[i*5+0] = np.random.random()*32
				init[i*5+1] = np.random.random()*32
				init[i*5+2] = np.random.normal(128,127)
				init[i*5+3] = np.random.normal(128,127)
				init[i*5+4] = np.random.normal(128,127)

		attack_result = differential_evolution(predict_fn, bounds, maxiter=self.maxiter, popsize=popmul,
			recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

		attack_image = self.perturb_image(attack_result.x, img)
		with torch.no_grad():
			attack_var = Variable(attack_image).to(self.device)
		predicted_probs = F.softmax(self.model(attack_var),dim=1).data.cpu().numpy()[0]

		predicted_class = np.argmax(predicted_probs)

		if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
			return 1, attack_var,predicted_class
		return 0, [None],predicted_class


	# def attack_all(self, loader):
	#
	# 	correct = 0
	# 	success = 0
	#
	# 	for batch_idx, (input, target) in enumerate(loader):
	#
	# 		img_var = Variable(input, volatile=True).cuda()
	# 		prior_probs = F.softmax(self.model(img_var))
	# 		_, indices = torch.max(prior_probs, 1)
	#
	# 		if target[0] != indices.data.cpu()[0]:
	# 			continue
	#
	# 		correct += 1
	# 		target = target.numpy()
	#
	# 		targets = [None] if not self.targeted else range(10)
	#
	# 		for target_calss in targets:
	# 			if (self.targeted):
	# 				if (target_calss == target[0]):
	# 					continue
	#
	# 			flag, x = self.attack(input, target[0])
	#
	# 			success += flag
	# 			if (self.targeted):
	# 				success_rate = float(success)/(9*correct)
	# 			else:
	# 				success_rate = float(success)/correct




