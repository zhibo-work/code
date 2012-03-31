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
	'''
	Arguments:
		K: Number of topics
		vocab: A set of words to recognize. When analyzing documents, any word not in this set will be ignored.
		D: Total number of documents in the population. For a fixed corpus,
		   this is the size of the corpus. 
		eta: Hyperparameter for prior on topics beta

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

		# initialize topic distribution, i.e. log_beta
		seed = 1115574245
		sum = 0
		log_beta = np.zeros([self._K,self._W])

		# little function to perform element summation
		def element_add_1(x):
			return x + 1.0 + np.random.randint(seed)
		log_beta = map(element_add_1, wordcts)
		# for i in xrange(self._K):
		# 	for n in xrange(self._W):
		# 		log_beta[i,n] = wordcts[i,n] + 1.0 +np.random.randint(seed) 
		# to initialize and smooth
		sum = np.log(np.sum(log_beta))
		
		# little function to normalize log_beta
		def element_add_2(x):
			return x + np.log(x-sum)
		log_beta = map(element_add_2,log_beta)

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
	
	# the next function correspond to eq.7 in ctm paper, which 
	# is the upper bound
	def expect_mult_norm(self, lambda_v, nu_v, zeta_v):
		sum_exp = np.sum(np.exp(lambda_v) + 0.5 * nu_v)
		bound = (1.0 / zeta_v) * sum_exp - 1.0 + np.log(zeta_v)

	def lhood_bnd(self):
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
		lhood -= expect_mult_norm(lambda_v, nu_v, zeta_v) * self._D
		for i in range(self._W):
			for j in range(self._K):
				if phi_v[i,j] > 0: 
					topic_scores[j] = phi_v[i,j] * (topic_scores[j] + self.cts[i]) 
					lhood += self.cts[i] * phi_v[i,j] * (lambda_v[j] + log_beta[j,i] - log_phi_v[i,j])
		lhood_v = lhood

	# optimize zeta
	def opt_zeta(self):
		zeta_v = 1.0
		zeta_v += np.sum(np.exp(lambda_v + np.dot(0.5 ,nu_v)))
		# for  i in range(self._K):
		# 	zeta_v += np.exp(lambda_v[i] + (0.5) * nu_v[i])

	# optimize phi
	def opt_phi(self):
		log_sum_n = 0

		for n in range(self._W):
			log_sum_n = 0
			for i in range(self._K):
				log_phi_v[n,i] =  lambda_v[i] + log_beta[i,n]
				if i == 0:
					log_sum_n = log_phi_v[n,i]
				else:
					log_sum_n = log_sum(log_sum_n,log_phi_v[n,i])

			for i in range(self._K):
				log_phi_v[n,i] -= log_sum_n
				phi_v[n,i] = np.exp(log_phi_v[n,i])

	# optimize lambda
	def f_lambda(self):
		temp = np.zeros((4,self._K))
		# temp = [[0 for i in range(self._K)] for j in range(4)]
		term1 = term2 = term3 = 0

		# compute lambda^T * \sum phi
		term1 = np.dot(lambda_v * sum_phi)
		# compute lambda - mu (= temp1)
		temp[1] += np.subtract(lambda_v, self.mu)
		# compute (lambda - mu)^T Sigma^-1 (lambda - mu)
		term2 = (-0.5) * temp[1] * self.inv_cov * temp[1]
		# last term
		for i in range(self._K):
			term3 += np.exp(lambda_v[i] + 0.5 * nu_v[i])
		# need to figure out how term3 is calculated 
		term3 = -(np.add(np.subtract(np.dot((1.0/zeta_v), term3), 1.0), np.log(zeta_v))) * self._W
		return -(term1 + term2 + term3)

	def df_lambda(self):
		# compute \Sigma^{-1} (\mu - \lambda)
		temp[0] = np.zeros(self._K)
		temp[1] = np.subtract(self.mu - lambda_v)
		temp[0] = self.inv_cov * temp[1] 

		#  compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)
		for i in range(self._K):
			temp[3][i] = np.dot((self._D / zeta_v), np.exp(lambda_v[i] + np.dot(0.5, nu_v[i])))

		# set return value (note negating derivative of bound)
		df = np.zeros(self._K)
		df -= np.subtract(np.subtract(temp[0], sum_phi),temp[3])

	def opt_lambda(self): 
		sum_phi = np.zeros(self._K)
		for i in range(self._W):
			for j in range(self._K):
				sum_phi[j] = self.wordcts[i] * phi_v[i,j]

		lambda_v = fmin_cg(f_lambda,x0, fprime = df_lambda,gtol = 1e-5, epsilon = 0.01, maxiter = 500)

	# optimize nu
	def df_nu(self,nu):
		v = np.array([0.0 for i in range(self._K)])
		for i in range(self._K):
			v[i] = - np.dot(0.5,inv_cov[i,i]) - np.dot((0.5 * self._W/zeta_v), np.exp(lambda_v[i] + nu[i]/2)) + (0.5 * (1.0 / nu[i]))
		return v

	def d2f_nu(self,nu):
		v = [0.0 for i in range(self._K)]
		for i in range(self._K):
			v[i] = - np.dot((0.25 * (self._W/zeta_v)), np.exp(lambda_v[i] + nu[i]/2)) - (0.5 * (1.0 / nu[i] * nu[i]))
		return v

	def opt_nu(self):
		df = d2f = 0
		nu = np.array([10 for i in range(self._K)])
		log_nu = np.log(nu)

		for i in range(self._K):
			while np.fabs(df) > 1e-10:
				df = df_nu(nu[i])
				d2f = d2f_nu(nu[i])
				log_nu[i] = log_nu[i] - (df * nu[i])/(d2f * nu[i] * nu[i] + df * nu[i])
		nu = np.exp(log_nu)

	# initial variational parameters
	def init_var_para(self):
		phi_v = np.array([[1.0/self._K for i in range(self._W)] for j in range(self._K)])
		log_phi_v = np.array([[-(np.log(self._K)) for i in range(self._W)] for j in range(self._K)])
		zeta_v = 0
		nu_v = np.array([0 for i in range(self._K)])
		lambda_v = np.array([0 for i in range(self._K)])

		niter = 0
		lhood_v = 0

	# variational inference
	def var_inference(self):
		lhood_old = 0
		convergence = 0

		lhood_bnd(self)
		while ((convergence > 1e-5) & (niter < 500)):
			niter += 1
			opt_zeta(self)
			opt_lambda(self)
			opt_zeta(self);
			opt_nu(self);
			opt_zeta(self);
			opt_phi(self);

			lhood_old = lhood_v
			lhood_bnd(self)

			convergence = np.fabs((lhood_old - lhood_v)/lhood_old)

			if ((lhood_old > lhood_v)& (niter>1)):
				print "WARNING: iter ",niter, "lhood_old: ", lhood_old, ">", "lhood_v: ", lhood_v
			if convergence > 1e-5:
				converged_v = 0
			else:
				converged_v = 1
			return lhood_v

	def update_expected_ss(self):
		# init
		mu_ss    = np.zeros(self._K)
		cov_ss   = np.zeros((self._K, self._K))
		beta_ss  = np.zeros((self._K, self._W))
		ndata_ss = 0

		# covariance and mean suff stats
		for i in range(self._K):
			mu_ss[i] = lambda_v[i]
			for j in range(self._K):
				lilj = lambda_v[i] * lambda_v[j]
				if i == j:
					cov_ss[i,j] = cov_ss[i,j] + nu_v[i] + lilj
				else:
					cov_ss[i,j] = cov_ss[i,j] + lilj
				# topics suff stats
				for i in range(self._W):
					for j in range(self._K):
						w = word[i] # d->word[i], is it the index of the i-th word?
						c = count[i]
						beta_ss[j,w] = beta_ss[j,w] + c * phi_v[i,j]
				# number of data
				ndata_ss += 1

	 # importance sampling the likelihood based on the variational posterior
	def sample_term(self):
		t1 = 0.5 * self.log_det_inv_cov
		t1 += -(0.5) * self._K * 1.837877 # 1.837877 is the natural logarithm of 2*pi
		for i in range(self._K):
			for j in range(self._K):
				t1 -= (0.5) * (eta[i] - self.mu[i]) * self.inv_cov[i,j] * (eta[j] - self.mu[j])
		# compute theta
		sum_t = 0 
		for i in range(self._K):
			theta[i] = np.exp(eta[i])
			sum_t += theta[i]
		for i in range(self._K):
			theta[i] = theta[i] / sum_t
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

	 def sample_lhood(self):
		# for each sample
		for n in range(self._W):
			# sample eta from q(\eta)
			for i in range(self._K): 
				v = random.gauss(0, np.sqrt(nu[i]))
				eta[i] = v + lambda_v[i]
			# compute p(w | \eta) - q(\eta)
			log_prob = sample_term(self)
			# update log sum
			if n == 0:
				sum_l = log_prob
			else:
				sum_l = log_sum(sum, log_prob)
		sum_l = sum_l - np.log(nsamples)
		return sum_l

	 # expected theta under a variational distribution
	 # (v is assumed allocated to the right length.)
	def expected_theta(self):
		nsamples = 100
		# initialize e_theta
		e_theta = -1 * np.ones(self._K)
		# for each sample
		for n in range(self._W):
			# sample eta from q(\eta)
			for i in range(self._K):
				v = random.gauss(0, np.sqrt(nu[i]))
				eta[i] = v + lambda_v[i]
			# compute p(w | \eta) - q(\eta)
			w = sample_term(self)
			# compute theta
			sum_t = 0
			for i in range(self._K):
				theta[i] = np.exp(eta[i])
				sum_t += theta[i]
			for i in range(self._K):
				theta[i] = theta[i] / sum_t
			# update e_theta
			for i in range(self._K):
				e_theta[i] = log_sum(e_theta[i], w+np.log(theta[i]))
		# normalize e_theta and set return vector
		sum_et = -1
		for i in range(self._K):
			e_theta[i] -= np.log(nsamples)
			sum_et = log_sum(sum_et, e_theta[i])
		for i in range(self._K):
			val[i] = np.exp(e_theta[i] - sum_et)

	 # log probability of the document under proportions theta and topics beta
	def log_mult_prob(self):
		ret = 0
		for i in range(self._W):
			term_prob = 0
			for k in range(len(log_beta)):
				term_prob+=theta[k] * np.exp(log_beta[k,i])
			ret += np.log(term_prob) * count[i]
		return ret

	'''
	estimate stage
	'''
	# the main function
	def em(self):
		iteration = 0
		convergence = lhood = lhood_old = 1.0
		avg_niter = converged_pct = old_conv = 0.0
		reset_var = 1
		var_max_iter = 500
		var_convergence = 1e-5

		corpus_lambda = np.zeros((self._D,self._K))
		corpus_nu = np.zeros((self._D,self._K))
		corpus_phi_sum = np.zeros((self._D,self._K))

		while ((iteration < 1000) and ((convergence > 1e-3) or (convergence < 0))):
			expectation(self)
			convergence = (lhood_old - lhood_bnd) / lhood_old
			if (((iteration % 1) == 0) or math.isnan(lhood)):
				pass
				# write a bunch of data: iteration lambda nu

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
			# reset all the sufficient statistics, is this necessary?

	# e-step
	def  expectation(self):
		avg_niter = 0.0
		converged_pct = 0
		total = 0

		phi_sum = np.zeros(self._K)

		for i in range(self._D):
			ids = self.wordids[d]
           		cts = self.wordcts[d]
           		lhood = var_inference(self)
			update_expected_ss(self)
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

	# m-step
	def maximization(self):
		# mean maximization
		for i in range(self._K):
			mu[i] = mu_ss[i] / ndata_ss
		# covariance maximization
		for i in range(self._K):
			for j in range(self._K):
				cov[i,j] = (1.0/ ndata_ss) * cov_ss[i,j] + ndata_ss * mu[i] * mu[j] - mu_ss[i] * mu[j] - mu_ss[j] * mu[i]
		# covariance shrinkage
		lw = LedoitWolf()
		cov_result = lw.fit(cov,assume_centered=True).covariance_
		inv_cov = np.linalg.inv(cov_result)
		log_det_inv_cov = np.log(np.linalg.det(inv_cov))

		# topic maximization
		for i in range(self._K):
			sum_m = 0 
			for j in range(self._W):
				sum_m += beta_ss[i,j]

			if sum_m == 0:
				sum_m = -1000 * self._W
			else:
				sum_m = np.log(sum_m)
			for j in range(self._W):
				log_beta[i,j] = np.log(beta_ss[i,j] - sum_m)

	# load a model, and do approximate inference for each document in a corpus
	def inference(self):
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

			# figure out how to pass the parameters
			lhood[i] = var_inference(self)
			lambda_corpus[i] = lambda_v
			nu_corpus[i] = nu_v
			for j in range(self._K):
				for n in range(self._W):
					phi_sums_corpus[i,j] += phi_v[n,j]

		# output likelihood and some variational parameters
		# write them to files
		with open('ctm_lhood','w') as ctm_lhood_dump:
			cPickle.dump(lhood,ctm_looh_dump)
		with open('corpus_lambda','w') as corpus_lambda-dump:
			cPickle.dump(corpus_lambda,corpus_lambda_dump)
		with open('corpus_nu','w') as corpus_nu_dump:
			cPickle.dump(corpus_nu, corpus_nu_dump)
		with open('phi_sums','w') as phi_sums_dump:
			cPickle.dump(phi_sums,phi_sums_dump)

	
	def pod_experiment(self, docs, proportions = 0.5):
		'''
		read in corpus ,and split it into observed data and held-out data
		 ` proportions` indicates the ratio of the split

		for each partially observed document: (a) perform inference on the
		 observations (b) take expected theta and compute likelihood

		'''
		permute_docs = np.random.permutation(docs)
		split_point = proportions * len(docs)
		obs_docs = permute_docs[:split_point]
		heldout_docs = permute_docs[split_point:]

		log_lhood = np.zeros(self._D)
		e_theta = np.zeros(self._K)
		for i in range(len(obs_docs)):
			# get observed and heldout documents
			obs_doc = obs_docs[i]
			heldout_doc = heldout_docs[i]
			#  compute variational distribution
			# initial variational parameters
			init_var_para()
			var_inference()
			expected_theta()
			#  approximate inference of held out data
			l = log_mult_prob(heldout_doc, e_theta, log_beta)
			log_lhood[i] = l
			total_words += len(heldout_doc[0]) 
			# TODO : make clear here  whether it is `heldout_doc[0] 
			# or `heldout_doc`
			total_lhood += l
		perplexity = np.exp(- total_lhood / total_words)



