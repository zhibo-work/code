# -*- coding: utf-8 -*-

# This code is to port Blei's CTM C code in Python.
# The bone structure follows Hoffmann's OnlineVB Python code.

# use logging tools to keep track of the behaviors
import os  		# to do folder process
import random   	# to generate random number
import cPickle	# to write in files
import math 		# just math stuff
import logging	# having decided to use this or not
import itertools	# do iteration work, obviously

import numpy as np 	# standard numpy
from scipy.special import gammaln, digamma, psi 	# gamma function utils
# log(sum(exp(x))) that tries to avoid overflow
from scipy.maxentropy import logsumexp
# Minimize a function using a nonlinear conjugate gradient algorithm.
from scipy.optimize import fmin_cg
from scipy import stats  							# calculate pdf of gaussian
# take the advantages of gensim provides
from gensim import interfaces, utils
#  mainly to perform covariance shrinkage
from sklearn.covariance import LedoitWolf

meanchangethresh = 0.001

def dirichlet_expectation(alpha):
	"""
	For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
	"""
	if (len(alpha.shape) == 1):
		return(psi(alpha) - psi(np.sum(alpha)))
	return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

def log_sum(log_a,log_b):
	v = 0
	if log_a == -1:
		return log_b
	if log_a < log_b:
		v = log_b + np.log(1+ np.exp(log_a - log_b))
	else:
		v = log_a + np.log(1+ np.exp(log_b - log_a))
	return v

'''
parse docuemnts
'''
def parse_doc_list(docs, vocab):
	"""
	Parse a document into a list of word ids and a list of counts,
	or parse a set of documents into two lists of lists of word ids
	and counts.

	Arguments:
	docs:  List of D documents. Each document must be represented as
		   a single string. (Word order is unimportant.) Any
		   words not in the vocabulary will be ignored.
	vocab: Dictionary mapping from words to integer ids.

	Returns a pair of lists of lists.

	The first, wordids, says what vocabulary tokens are present in
	each document. wordids[i][j] gives the jth unique token present in
	document i. (Don't count on these tokens being in any particular
	order.)

	The second, wordcts, says how many times each vocabulary token is
	present. wordcts[i][j] is the number of times that the token given
	by wordids[i][j] appears in document i.
	"""
	if (type(docs).__name__ == 'str'):
		temp = list()
		temp.append(docs)
		docs = temp

	# number of all the documents in corpus
	D = len(docs)
	# wordids represent the index of each word
	wordids = list()
	#  wordcts represent the number each word appears
	wordcts = list()
	for d in range(0, D):
		docs[d] = docs[d].lower()
		docs[d] = re.sub(r'-', ' ', docs[d])
		docs[d] = re.sub(r'[^a-z ]', '', docs[d])
		docs[d] = re.sub(r' +', ' ', docs[d])
		words = string.split(docs[d])
		ddict = dict()
		for word in words:
			if (word in vocab):
				wordtoken = vocab[word]
				if (not wordtoken in ddict):
					ddict[wordtoken] = 0
				ddict[wordtoken] += 1
		wordids.append(ddict.keys())
		wordcts.append(ddict.values())

	return((wordids, wordcts))

class CTM:
	"""
	Correlated Topic Models in Python

	"""
	def __init__(self, vocab, K, D, mu, cov):
	''' initialization

	Arguments:
		K: Number of topics
		vocab: A set of words to recognize. When analyzing documents, any word not in this set will be ignored.
		D: Total number of documents in the population. For a fixed corpus,
		   this is the size of the corpus.

		mu and cov: the hyperparameters logistic normal distribution for prior on weight vectors theta
		'''
		self._vocab = dict()
		for word in vocab:
			word = word.lower()
			word = re.sub(r'[^a-z]', '', word)
			self._vocab[word] = len(self._vocab)

		self._K = K 					# number of topics
		self._W = len(self._vocab) 	# number of all the words
		self._D = D 					# number of documents

		(wordids, wordcts) = parse_doc_list(docs, self._vocab)
		self.wordids = wordids
		self.wordcts = wordcts
		# distinct number of terms, I don't know what's the use for this, but just
		# leave it here
		self.nterms = len(self.wordids)

		# mu : K-size vector with 0 as initial value
		# cov  : K*K matrix with 1 as initial value , together they make a Gaussian
		if mu is None:
			self.mu = np.zeros(self._K)
		else:
			self.mu = mu
		if cov is None:
			self.cov = np.ones(self._K, self._K)
		else:
			self.cov = cov

		self.inv_cov = np.linalg.inv(self.cov)
		self.log_det_inv_cov = np.log(np.linalg.det(self.inv_cov))

		self.ndata = 0 # cumulate count of number of docs processed

		# initialize topic distribution, i.e. self.log_beta
		seed = 1115574245
		sum = 0
		self.beta = np.zeros([self._K,self._W])
		self.log_beta = np.zeros([self._K,self._W]) # log 0 is inf, so just to make it 0

		# little function to perform element summation
		def element_add_1(x):
			return x + 1.0 + np.random.randint(seed)
		self.log_beta = map(element_add_1, wordcts)
		# for i in xrange(self._K):
		# 	for n in xrange(self._W):
		# 		self.log_beta[i,n] = wordcts[i,n] + 1.0 +np.random.randint(seed)
		# to initialize and smooth
		sum = np.log(np.sum(self.log_beta))

		# little function to normalize self.log_beta
		def element_add_2(x):
			return x + np.log(x-sum)
		self.log_beta = map(element_add_2,self.log_beta)

	'''
	before the actual variational inference
	below are some funtions to deal with the variational
	parameters to be used in variational inference, namely
	add '_v' to indicate variational parameter
	* zeta_v
	* phi_v
	* lambda_v, in order to distinguish python's own function name.
	* nu_v
	'''

	def opt_zeta(lambda_v,nu_v):
		# optimize zeta
		zeta_v = 1.0
		zeta_v += np.sum(np.exp(lambda_v + np.dot(0.5 ,nu_v)))
		return zeta_v

	def opt_phi(self, lambda_v,log_phi_v):
		# optimize phi
		log_sum_n = 0
		for n in range(self._W):
			log_sum_n = 0
			for i in range(self._K):
				log_phi_v[n,i] =  lambda_v[i] + self.log_beta[i,n]
				if i == 0:
					log_sum_n = log_phi_v[n,i]
				else:
					log_sum_n = log_sum(log_sum_n,log_phi_v[n,i])

			for i in range(self._K):
				log_phi_v[n,i] -= log_sum_n
				phi_v[n,i] = np.exp(log_phi_v[n,i])
		return (phi_v, log_phi_v)

	# next three functions to optimize lambda
	def f_lambda(self, sum_phi, phi_v, lambda_v, nu_v, zeta_v):
		temp1 = np.zeros(self._K)
		# temp = [[0 for i in range(self._K)] for j in range(4)]
		term1 = term2 = term3 = 0

		# compute lambda^T * \sum phi
		term1 = np.dot(lambda_v * sum_phi)
		# compute lambda - mu (= temp1)
		temp1 += np.subtract(lambda_v, self.mu)
		# compute (lambda - mu)^T Sigma^-1 (lambda - mu)
		term2 = (-0.5) * temp[1] * self.inv_cov * temp[1]
		# last term
		for i in range(self._K):
			term3 += np.exp(lambda_v[i] + 0.5 * nu_v[i])
		# need to figure out how term3 is calculated
		term3 =  -((1.0/zeta_v) * term3 - 1.0 + np.log(zeta_v)) * self._K
		return (-(term1 + term2 + term3))

	def df_lambda(self, sum_phi, lambda_v, nu_v, zeta_v):
		# compute \Sigma^{-1} (\mu - \lambda)
		temp0= np.zeros(self._K)
		temp1 = np.subtract(self.mu - lambda_v)
		temp0= self.inv_cov * temp1
		temp3 =  np.zeros(self._K)

		#  compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)
		for i in range(self._K):
			temp3[i] = np.dot((self._D / zeta_v), np.exp(lambda_v[i] + np.dot(0.5, nu_v[i])))

		# set return value (note negating derivative of bound)
		df = np.zeros(self._K)
		df -= np.subtract(np.subtract(temp0, sum_phi),temp3)
		return df

	def opt_lambda(self, phi_v, nu_v, zeta_v):
		sum_phi = np.zeros(self._K)
		for i in range(self._W):
			for j in range(self._K):
				sum_phi[j] = self.wordcts[i] * phi_v[i,j]

		lambda_v = fmin_cg(f_lambda, lambda_v, fprime = df_lambda,gtol = 1e-5, epsilon = 0.01, maxiter = 500)
		return lambda_v

	def opt_nu(self, lambda_v, zeta_v):
		# optimize nu
		df = d2f = 0
		nu_v = np.dot(10,np.ones(self._K))
		log_nu_v = np.log(nu_v)

		for i in range(self._K):
			while np.fabs(df) > 1e-10:
				nu_v[i] =  np.exp(log_nu_v[i])
				if math.isnan(nu_v[i]):
					nu_v[i] = 20
					log_nu_v[i] = np.log(nu[i]_v)
				df = - np.dot(0.5,self.inv_cov[i,i]) - np.dot((0.5 * self._W/zeta_v), np.exp(lambda_v[i] + nu_v[i]/2)) + (0.5 * (1.0 / nu_v[i]))
				d2f = - np.dot((0.25 * (self._W/zeta_v)), np.exp(lambda_v[i] + nu_v[i]/2)) - (0.5 * (1.0 / nu_v[i] * nu_v[i]))
				log_nu_v[i] = log_nu_v[i] - (df * nu_v[i])/(d2f * nu_v[i] * nu_v[i] + df * nu_v[i])
		nu_v = np.exp(log_nu_v)

		return nu_v

	'''
	the actual variational inference process

	'''

	def lhood_bnd(self, phi_v,log_phi_v, lambda_v, nu_v, zeta_v):
		''' compute the likelihood bound give the variational parameters

		Arguments:
			variational parameters

		Returns:
			likelihood bound

		'''
		topic_scores = np.zeros(self._K)

		# E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)
		lhood = (0.5) * self.log_det_inv_cov + 0.5 * self._K
		for i in range(self._K):
			v = - (0.5) * nu_v[i] * self.inv_cov[i,i]
			for j in range(self._K):
				v -= (0.5) * (lambda_v[i] - self.mu[i]) * self.inv_cov[i,j] * (lambda_v[j] - self.mu[j])
			v += (0.5) * np.log(nu_v[i])
			lhood += v

		# E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))

		# Equation 7 in paper, calculate the upper bound
		sum_exp = np.sum(np.exp(lambda_v) + 0.5 * nu_v)
		bound = (1.0 / zeta_v) * sum_exp - 1.0 + np.log(zeta_v)

		lhood -= bound * self._D

		for i in range(self._W):
			for j in range(self._K):
				if phi_v[i,j] > 0:
					topic_scores[j] = phi_v[i,j] * (topic_scores[j] + self.cts[i])
					lhood += self.cts[i] * phi_v[i,j] * (lambda_v[j] + self.log_beta[j,i] - log_phi_v[i,j])
		lhood_v = lhood
		return lhood_v

	# variational inference
	def var_inference(self, phi_v, log_phi_v, lambda_v, nu_v, zeta_v):
		
		niter = 0.0
		lhood_v = 0.0
		lhood_old = 0.0
		convergence = 0.0

		lhood_v = lhood_bnd(self, phi_v,log_phi_v, lambda_v, nu_v, zeta_v)
		while ((convergence > 1e-5) & (niter < 500)):
			niter += 1
			zeta_v = opt_zeta(lambda_v,nu_v)
			lambda_v = opt_lambda(self, phi_v, nu_v, zeta_v)
			zeta_v = opt_zeta(lambda_v,nu_v)
			nu_v = opt_nu(self, lambda_v, zeta_v);
			zeta_v = opt_zeta(lambda_v,nu_v)
			(phi_v, log_phi_v) = opt_phi(self, lambda_v,log_phi_v);

			lhood_old = lhood_v
			lhood_v = lhood_bnd(self, phi_v,log_phi_v, lambda_v, nu_v, zeta_v)

			convergence = np.fabs((lhood_old - lhood_v)/lhood_old)

			if ((lhood_old > lhood_v)& (niter>1)):
				print "WARNING: iter ",niter, "lhood_old: ", lhood_old, ">", "lhood_v: ", lhood_v
		
		if convergence > 1e-5:
			converged_v = 0
		else:
			converged_v = 1
		return (lhood_v,phi_v, log_phi_v, lambda_v, nu_v, zeta_v)

	def update_expected_ss(self, lambda_v, nu_v, phi_v, ids, cts):
		'''
		Arguments:
			variational paraments and doc paraments
		Returns:
			sufficient statistics
		'''

		ids = ids
		cts = cts

		# covariance and mean suff stats
		for i in range(self._K):
			self.mu[i] = lambda_v[i]
			for j in range(self._K):
				lilj = lambda_v[i] * lambda_v[j]
				if i == j:
					self.cov[i,j] = self.cov[i,j] + nu_v[i] + lilj
				else:
					self.cov[i,j] = self.cov[i,j] + lilj
		# topics suff stats
		for i in range(self._W):
			for j in range(self._K):
				w = ids[i] # d->word[i], is it the index of the i-th word?
				self.beta[j,w] = self.beta[j,w] + cts[i] * phi_v[i,j]
		# number of data
		self.ndata += 1

	def sample_term(self, eta, lambda_v, nu_v):
		'''
		importance sampling the likelihood based on the variational posterior
		
		Arguments:
			eta : natural parameter of logistic normal distribution
			theta : mean parameter of logistic normal distribution
			The mapping between them is equation 3 in the paper:
					eta[i] = log theta[i] / theta[K]
		Returns:
			value of p(w | eta) - q(eta)
		'''
		t1 = 0.5 * self.log_det_inv_cov
		t1 += -(0.5) * self._K * 1.837877 # 1.837877 is the natural logarithm of 2*pi
		for i in range(self._K):
			for j in range(self._K):
				t1 -= (0.5) * (eta[i] - self.mu[i]) * self.inv_cov[i,j] * (eta[j] - self.mu[j])
		# compute theta
		theta = eta[:]
		sum_t = np.sum(np.exp(eta))
		theta = np.divide(theta, sum_t)

		# compute word probabilities
		for n in range(self._W):
			word_term = 0
			for i in range(self._K):
				word_term += theta[i] * np.exp(self.log_beta[i,n])
			t1 += count[n] * np.log(word_term)
		# log(q(\eta | lambda, nu))
		t2 = 0
		for i in range(self._K):
			t2 += stats.norm.pdf(eta[i] - lambda_v[i], np.log(lambda_v[i]), loc=0, np.sqrt(nu_v[i]),1.0)
			# pdf of lognorm dist. parameters are (x, scale(mu), loc, shape(sigma))
		return(t1-t2)


	def expected_theta(self, lambda_v, nu_v):
		''' Return expected theta under a variational distribution

		Args:
			self : use all the parameters initialized before
			lambda_v : variational parameter lambda
			nu_v : variational parameter nu

		Returns:
			val : the expected theta
		'''
		nsamples = 100
		eta = np.zeros(self._K)
		theta = eta[:]
		# initialize e_theta
		e_theta = -1.0 * np.ones(self._K)
		# for each sample
		for n in range(self._W):
			# sample eta from q(\eta)
			for i in range(self._K):
				eta[i] = random.gauss(0, np.sqrt(nu[i])) + lambda_v[i]
			# compute p(w | \eta) - q(\eta)
			log_prob = sample_term(self,eta, lambda_v, nu_v)
			# compute theta
			theta = eta[:]
			sum_t = np.sum(np.exp(eta))
			theta = np.divide(theta, sum_t)

			# update e_theta
			for i in range(self._K):
				e_theta[i] = log_sum(e_theta[i], log_prob + np.log(theta[i]))
		# normalize e_theta and set return vector
		sum_et = -1.0
		for i in range(self._K):
			e_theta[i] -= np.log(nsamples)
			sum_et = log_sum(sum_et, e_theta[i])
		e_theta = np.exp(np.subtract(e_theta, sum_et))
		return e_theta

	def log_mult_prob(self, cts, e_theta):
		# log probability of the document under proportions theta and topics beta
		val = 0
		for i in range(self._W):
			term_prob = 0
			for k in range(len(self.log_beta)):
				term_prob += e_theta[k] * np.exp(self.log_beta[k,i])
			val += np.log(term_prob) * cts[i]
		return val

	'''
	estimate stage
	'''
	def em(self):

		# load model parameters
		# load gaussian
		with open('ctm_nu','rb') as ctm_nu_dump:
			self.nu = cPickle.load(ctm_nu_dump)
		with open('ctm_cov','rb') as ctm_cov_dump:
			self.cov = cPickle.load(ctm_cov_dump)
		with open('ctm_inv_cov','rb') as ctm_inv_cov_dump:
			self.inv_cov = cPickle.load(ctm_inv_cov_dump)
		with open('ctm_log_det_inv_cov','rb') as ctm_log_det_inv_cov_dump:
			self.log_det_inv_cov = cPickle.load(ctm_log_det_inv_cov_dump)
		# load topic matrix 
		with open('ctm_log_beta','rb') as ctm_log_beta_dump:
			self.log_beta = cPickle.load(log_beta_dump)

		# the main function
		iteration = 0
		convergence = 1.0
		lhood = lhood_old =  0.0
		avg_niter = converged_pct = old_conv = 0.0
		reset_var = 1
		var_max_iter = 500
		var_convergence = 1e-5

		corpus_lambda = np.zeros((self._D,self._K))
		corpus_nu = np.zeros((self._D,self._K))
		corpus_phi_sum = np.zeros((self._D,self._K))

		while ((iteration < 1000) and ((convergence > 1e-3) or (convergence < 0))):
			# e-step
			lhood= expectation(self, docs)
			convergence = (lhood_old - lhood) / lhood_old

			# m-step
			if convergence < 0:
				reset_var = 0
				if var_max_iter > 0:
					var_max_iter += 10
				else:
					var_max_iter = var_max_iter / 10
			else:
				maximization(self)
				lhood_old = lhood
				reset_var = 1
				iteration += 1
			old_conv = convergence

	def expectation(self, docs):
		''' E-step of EM algorithm
		Arguments:
			corpus: the docs needed to be worked on, need to get ids and cts
		Returns:
			sufficient statistics : lhood, self.mu, self.cov, self.beta, self.ndata
		'''
		avg_niter = 0.0
		converged_pct = 0.0
		total = 0.0

		phi_sum = np.zeros(self._K)

		(wordids, wordcts) = parse_doc_list(docs, self._vocab)
		ndocs = len(docs)		# number of docs in this corpus 

		for i in range(ndocs):
			ids = self.wordids[i]
			cts = self.wordcts[i]

			# initialize the variational parameters
			phi_v = np.dot(1.0/self._K , np.ones((self._K,self._W)))
			log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K,self._W)))
			zeta_v = 0.0
			nu_v = np.zeros(self._K)
			lambda_v = np.zeros(self._K)

			(lhood_v,phi_v, log_phi_v, lambda_v, nu_v, zeta_v) = var_inference(self, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)
			update_expected_ss(self, lambda_v, nu_v, phi_v, ids, cts)
			total += lhood
			avg_niter = niter_v
			converged_pct = converged_v
			corpus_lambda[i] = lambda_v
			corpus_nu[i] = nu_v
			for j in range(self._W):
				for n in range(self._K):
					phi_sum[n] = phi_v[j,n]
			corpus_phi_sum[i] = phi_sum
		avg_niter avg_niter / self._D
		converged_pct = converged_pct / self._D
		total_lhood = total
		return total_lhood

	# m-step
	def maximization(self):
		'''
		M-step of EM algorithm, use scikit.learn's LedoitWolf method to perfom 
		covariance matrix shrinkage. 
		Arguments:
			sufficient statistics, i.e. model parameters
		Returns:
			the updated sufficient statistics which all in self definition, so no 
			return values
		'''
		# mean maximization
		mu = np.divide(self.mu, self.ndata)
		# covariance maximization
		for i in range(self._K):
			for j in range(self._K):
				cov[i,j] = (1.0/ self.ndata) * self.cov[i,j] + self.ndata * mu[i] * mu[j] - self.mu[i] * mu[j] - self.mu[j] * mu[i]
		# covariance shrinkage
		lw = LedoitWolf()
		cov_result = lw.fit(cov,assume_centered=True).covariance_
		self.inv_cov = np.linalg.inv(cov_result)
		self.log_det_inv_cov = np.log(np.linalg.det(self.inv_cov))

		# topic maximization
		for i in range(self._K):
			sum_m = 0
			for j in range(self._W):
				sum_m += self.beta[i,j]

			if sum_m == 0:
				sum_m = -1000 * self._W
			else:
				sum_m = np.log(sum_m)
			for j in range(self._W):
				self.log_beta[i,j] = np.log(self.beta[i,j] - sum_m)

		# write model parameters to file
		# write gaussian
		with open('ctm_nu','w') as ctm_nu_dump:
			cPickle.dump(self.nu,ctm_nu_dump)
		with open('ctm_cov','w') as ctm_cov_dump:
			cPickle.dump(self.cov,ctm_cov_dump)
		with open('ctm_inv_cov','w') as ctm_inv_cov_dump:
			cPickle.dump(self.inv_cov, ctm_inv_cov_dump)
		with open('ctm_log_det_inv_cov','w') as ctm_log_det_inv_cov_dump:
			cPickle.dump(self.log_det_inv_cov,ctm_log_det_inv_cov_dump)
		# write topic matrix 
		with open('ctm_log_beta','w') as ctm_log_beta_dump:
			cPickle.dump(self.log_beta,log_beta_dump)


	def inference(self):
		''' Perform inference on corpus (seen or unseen)
		load a model, and do approximate inference for each document in a corpus
		'''
		# load model parameters
		# load gaussian
		with open('ctm_nu','rb') as ctm_nu_dump:
			self.nu = cPickle.load(ctm_nu_dump)
		with open('ctm_cov','rb') as ctm_cov_dump:
			self.cov = cPickle.load(ctm_cov_dump)
		with open('ctm_inv_cov','rb') as ctm_inv_cov_dump:
			self.inv_cov = cPickle.load(ctm_inv_cov_dump)
		with open('ctm_log_det_inv_cov','rb') as ctm_log_det_inv_cov_dump:
			self.log_det_inv_cov = cPickle.load(ctm_log_det_inv_cov_dump)
		# load topic matrix 
		with open('ctm_log_beta','rb') as ctm_log_beta_dump:
			self.log_beta = cPickle.load(log_beta_dump)

		# corpus level parameter initialization
		lhood_corpus = np.zeros(self._D)
		nu_corpus = np.zeros((self._D, self._K))
		lambda_corpus = np.zeros((self._D, self._K))
		phi_sums_corpus = np.zeros((self._D, self._K))

		# approximate inference
		for i in range(self._D):
			ids_doc = self.ids[i]
			cts_doc = self.cts[i]
			temp_sum = 0

		# initialize the variational parameters
		phi_v = np.dot(1.0/self._K , np.ones((self._K,self._W)))
		log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K,self._W)))
		zeta_v = 0.0
		nu_v = np.zeros(self._K)
		lambda_v = np.zeros(self._K)

		(lhood[i],phi_v, log_phi_v, lambda_corpus[i], nu_corpus[i], zeta_v) = var_inference(self, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)

		for j in range(self._K):
			for n in range(self._W):
				phi_sums_corpus[i,j] += phi_v[n,j]

		# output likelihood and some variational parameters
		# write them to files
		with open('ctm_lhood','w') as ctm_lhood_dump:
			cPickle.dump(lhood,ctm_looh_dump)
		with open('corpus_lambda','w') as corpus_lambda_dump:
			cPickle.dump(corpus_lambda,corpus_lambda_dump)
		with open('corpus_nu','w') as corpus_nu_dump:
			cPickle.dump(corpus_nu, corpus_nu_dump)
		with open('phi_sums','w') as phi_sums_dump:
			cPickle.dump(phi_sums,phi_sums_dump)


	def pod_experiment(self, docs, proportions = 0.5):
		''' Calculate perplexity value

		read in corpus, and split it into observed data and held-out data
		 ` proportions` indicates the ratio of the split

		for each partially observed document: (a) perform inference on the
		 observations (b) take expected theta and compute likelihood

		 Args:
			docs : the corpus
			proportions : the split ratio, 0.5 as initial value, can be assigned manually
		 Returns:
			perplexity : currently, the only evaluation value, add others later

		'''
		permute_docs = np.random.permutation(docs)
		split_point = proportions * len(docs)
		obs_docs = permute_docs[:split_point]
		heldout_docs = permute_docs[split_point:]

		# load model parameters
		# load gaussian
		with open('ctm_nu','rb') as ctm_nu_dump:
			self.nu = cPickle.load(ctm_nu_dump)
		with open('ctm_cov','rb') as ctm_cov_dump:
			self.cov = cPickle.load(ctm_cov_dump)
		with open('ctm_inv_cov','rb') as ctm_inv_cov_dump:
			self.inv_cov = cPickle.load(ctm_inv_cov_dump)
		with open('ctm_log_det_inv_cov','rb') as ctm_log_det_inv_cov_dump:
			self.log_det_inv_cov = cPickle.load(ctm_log_det_inv_cov_dump)
		# load topic matrix 
		with open('ctm_log_beta','rb') as ctm_log_beta_dump:
			self.log_beta = cPickle.load(log_beta_dump)

		log_lhood = np.zeros(self._D)
		e_theta = np.zeros(self._K)

		# for the sake of simplicity, proportion between 
		# observed doc and held-out doc are set to 0.5, no other value
		for i in range(len(obs_docs)):
			# get observed and heldout documents
			obs_doc = obs_docs[i]
			heldout_doc = heldout_docs[i]
			#  compute variational distribution
			
		# read in the model parameter learnt by training process

		# initialize the variational parameters
		phi_v = np.dot(1.0/self._K , np.ones((self._K,self._W)))
		log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K,self._W)))
		zeta_v = 0.0
		nu_v = np.zeros(self._K)
		lambda_v = np.zeros(self._K)

		(lhood_v,phi_v, log_phi_v, lambda_v, nu_v, zeta_v) = var_inference(self, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)
		e_theta = expected_theta(self, lambda_v, nu_v)
		for j in range(len(heldout_docs)):
			#  approximate inference of held out data
			l = log_mult_prob(self, cts, e_theta)
			log_lhood[i] = l
			total_words += len(heldout_doc[0])
			# TODO : make clear here  whether it is `heldout_doc[0]
			# or `heldout_doc`
			total_lhood += l
		perplexity = np.exp(- total_lhood / total_words)
		print 'the perplexity is:', perplexity



