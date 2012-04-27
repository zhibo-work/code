# -*- coding: utf-8 -*-

'''
This code is to port Blei's CTM C code in Python.
The bone structure follows Hoffmann's OnlineVB Python code.
'''

import os                # to do folder process
import random       # to generate random number
import cPickle    # to write in files
import math           # just math stuff
import logging    # tracking events that happen when program runs

# set up logging to file 
# log messages with levels of DEBUG and higher to file, and those messages at level INFO and higher to the console.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='ctm_log.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


import numpy as np  # standard numpy
# import scipy as sp  # standard scipy

# Minimize a function using a nonlinear conjugate gradient algorithm.
from scipy.optimize import fmin_cg
from scipy import stats   # calculate pdf of gaussian

#  perform covariance shrinkage
from sklearn.covariance import LedoitWolf

import preprocess


def log_sum(log_a, log_b):
	v = 0
	if log_a == -1:
		return log_b
	if log_a < log_b:
		v = log_b + np.log(1 + np.exp(log_a - log_b))
	else:
		v = log_a + np.log(1 + np.exp(log_b - log_a))
	return v


def safe_log(x):
	if x == 0:
		return -1000
	else:
		return np.log(x)


class CTM:
	"""
	Correlated Topic Models in Python

	TODO : USAGE NEEDED

	"""
	def __init__(self, K, mu=None, cov=None):
		'''
		Arguments:
			docs: list of documents to be processed
			K: Number of topics
			D: Total number of documents in the population. For a fixed corpus,
			   this is the size of the corpus.
			mu and cov: the hyperparameters logistic normal distribution for prior on weight vectors theta
		'''
		logging.info('Start CTM.')
		if K is None is None:
			raise ValueError('number of topics have to be specified.')
		# get the folder name which containing all the training files
		# we will have to manually specific the observed and heldout folders
		obs_filenames = os.listdir('/Users/sean/personal_research_doc/lda_explained/state/observed')

		logging.info("initializing id mapping from corpus, assuming identity")
		#initial a string to save all the file contents
		txt_corpus = []
		for thefile in obs_filenames:
			with open("./state/observed" + thefile, "rb") as f:
				strings = f.read()
				txt_corpus.append(strings)
		(dictionary, corpus) = preprocess.get_dict_and_corp(txt_corpus)
		logging.info("dictionary and corpus are generated")

		self.dictionary = dictionary
		self.corpus = corpus

		self._K = K                     # number of topics
		self._W = len(dictionary)   # number of all the words
		self._D = len(corpus)       # number of documents

		# initialize wordid and wordcount list for the whole corpus
		self.wordids = list()
		self.wordcts = list()

		for d, doc in enumerate(self.corpus):
			wordidsd = [id for id, _ in doc]
			wordctsd = np.array([cnt for _, cnt in doc])
			self.wordids.append(wordidsd)
			self.wordcts.append(wordctsd)

		# mu   : K-size vector with 0 as initial value
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
		self.log_det_inv_cov = safe_log(np.linalg.det(self.inv_cov))

		self.ndata = 0  # cumulate count of number of docs processed

		# initialize topic distribution, i.e. self.log_beta
		sum = 0
		self.beta = np.zeros([self._K, self._W])
		self.log_beta = np.zeros([self._K, self._W])

		for i in range(self._K):
			# initialize beta with a randomly chosen doc
			# stuff K topics randomly
			doc_no = np.random.randint(self._D)
			for j in range(len(self.wordids[doc_no])):
				self.log_beta[i, j] = self.wordcts[doc_no][j] + 1.0 + np.random.ranf()

		# to initialize and smooth
		sum = safe_log(np.sum(self.log_beta))

		# little function to normalize self.log_beta
		def element_add_2(x):
			return x + safe_log(x - sum)
		self.log_beta = map(element_add_2, self.log_beta)

	def opt_zeta(lambda_v, nu_v):
		logging.info("calculating variational parameter ZETA")
		# optimize zeta
		zeta_v = 1.0
		zeta_v += np.sum(np.exp(lambda_v + np.dot(0.5, nu_v)))
		return zeta_v

	def opt_phi(self, lambda_v, log_phi_v):
		logging.info("calculating variational parameter PHI")
		# optimize phi
		log_sum_n = 0
		phi_v = np.zeros_like(log_phi_v)
		for n in range(self._W):
			log_sum_n = 0
			for i in range(self._K):
				log_phi_v[n, i] = lambda_v[i] + self.log_beta[i, n]
				if i == 0:
					log_sum_n = log_phi_v[n, i]
				else:
					log_sum_n = log_sum(log_sum_n, log_phi_v[n, i])

			for i in range(self._K):
				log_phi_v[n, i] -= log_sum_n
				phi_v[n, i] = np.exp(log_phi_v[n, i])
		return (phi_v, log_phi_v)

	def opt_lambda(self, d, lambda_v, phi_v, nu_v, zeta_v):
		logging.info("calculating variational parameter LAMBDA")
		# optimize lambda
		# d : current working docutment index
		lambda_ini = lambda_v
		sum_phi = np.zeros(self._K)
		for i in range(self._W):
			for j in range(self._K):
				ids = self.wordids[d].index(i)
				sum_phi[j] = self.wordcts[d][ids] * phi_v[i, j]
		# inline function, define f for fmin_cg

		def f_lambda(self, sum_phi, phi_v, lambda_v, nu_v, zeta_v):
			temp1 = np.zeros(self._K)

			term1 = term2 = term3 = 0
			# compute lambda^T * \sum phi
			term1 = np.dot(lambda_v * sum_phi)
			# compute lambda - mu (= temp1)
			temp1 += np.subtract(lambda_v, self.mu)
			# compute (lambda - mu)^T Sigma^-1 (lambda - mu)
			term2 = (-0.5) * temp1 * self.inv_cov * temp1
			# last term
			for i in range(self._K):
				term3 += np.exp(lambda_v[i] + 0.5 * nu_v[i])
			# need to figure out how term3 is calculated
			term3 = - ((1.0 / zeta_v) * term3 - 1.0 + safe_log(zeta_v)) * self._K
			return (-(term1 + term2 + term3))

		# inline function, define f_prime for fmin_cg
		def df_lambda(self, sum_phi, lambda_v, nu_v, zeta_v):
			# compute \Sigma^{-1} (\mu - \lambda)
			temp0 = self.inv_cov * np.subtract(self.mu - lambda_v)
			temp3 = np.zeros(self._K)

			#  compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)
			for i in range(self._K):
				temp3[i] = np.dot((self._D / zeta_v), np.exp(lambda_v[i] + np.dot(0.5, nu_v[i])))

			# set return value (note negating derivative of bound)
			df = np.zeros(self._K)
			df -= np.subtract(np.subtract(temp0, sum_phi), temp3)
			return df
		# here, lambda_ini serves as initial value of lambda_v
		lambda_v = fmin_cg(f_lambda, lambda_ini, fprime=df_lambda, gtol=1e-5, epsilon=0.01, maxiter=500)
		return lambda_v

	def opt_nu(self, lambda_v, zeta_v):
		logging.info("calculating variational parameter NU")
		# optimize nu
		df = d2f = 0
		nu_v = np.dot(10, np.ones(self._K))
		log_nu_v = np.log(nu_v)

		for i in range(self._K):
			while np.fabs(df) > 1e-10:
				nu_v[i] = np.exp(log_nu_v[i])
				if math.isnan(nu_v[i]):
					nu_v[i] = 20
					log_nu_v[i] = safe_log(nu_v[i])
				df = - np.dot(0.5, self.inv_cov[i, i]) - np.dot((0.5 * self._W / zeta_v), np.exp(lambda_v[i] + nu_v[i] / 2)) + (0.5 * (1.0 / nu_v[i]))
				d2f = - np.dot((0.25 * (self._W / zeta_v)), np.exp(lambda_v[i] + nu_v[i] / 2)) - (0.5 * (1.0 / nu_v[i] * nu_v[i]))
				log_nu_v[i] = log_nu_v[i] - (df * nu_v[i]) / (d2f * nu_v[i] * nu_v[i] + df * nu_v[i])
		nu_v = np.exp(log_nu_v)

		return nu_v

	def lhood_bnd(self, d, phi_v, log_phi_v, lambda_v, nu_v, zeta_v):
		'''
		compute the likelihood bound given the variational parameters

		Arguments:
			d : current working docutment index
			variational parameters

		Returns:
			likelihood bound

		'''
		logging.info("calculating likelihood bound")
		# E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)
		lhood = (0.5) * self.log_det_inv_cov + 0.5 * self._K
		for i in range(self._K):
			v = - (0.5) * nu_v[i] * self.inv_cov[i, i]
			for j in range(self._K):
				v -= (0.5) * (lambda_v[i] - self.mu[i]) * self.inv_cov[i, j] * (lambda_v[j] - self.mu[j])
			v += (0.5) * safe_log(nu_v[i])
			lhood += v

		# E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))
		# Equation 7 in paper, calculate the upper bound
		sum_exp = np.sum(np.exp(lambda_v) + 0.5 * nu_v)
		bound = (1.0 / zeta_v) * sum_exp - 1.0 + safe_log(zeta_v)
		lhood -= bound * self._D

		for i in range(self._W):
			for j in range(self._K):
				if phi_v[i, j] > 0:
					lhood += self.wordcts[d][i] * phi_v[i, j] * (lambda_v[j] + self.log_beta[j, i] - log_phi_v[i, j])
		return lhood

	def var_inference(self, d, phi_v, log_phi_v, lambda_v, nu_v, zeta_v):
		'''Variational inference
		Arguments:
			d: current working docutment index
			variational parameters.

		Returns:
			likelihood bound and updated variational parameters
		'''
		logging.info("performing variational inference")
		niter = 0
		lhood_v = 0.0
		lhood_old = 0.0
		convergence = 0.0

		lhood_v = self.lhood_bnd(self, d, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)
		while ((convergence > 1e-5) & (niter < 500)):
			niter += 1
			logging.info("start iteration")
			logging.info("iteration no. %i", niter)
			zeta_v = self.opt_zeta(lambda_v, nu_v)
			lambda_v = self.opt_lambda(self, lambda_v, phi_v, nu_v, zeta_v)
			zeta_v = self.opt_zeta(lambda_v, nu_v)
			nu_v = self.opt_nu(self, lambda_v, zeta_v)
			zeta_v = self.opt_zeta(lambda_v, nu_v)
			(phi_v, log_phi_v) = self.opt_phi(self, lambda_v, log_phi_v)

			lhood_old = lhood_v
			lhood_v = self.lhood_bnd(self, d, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)

			convergence = np.fabs((lhood_old - lhood_v) / lhood_old)

			if ((lhood_old > lhood_v) & (niter > 1)):
				logging.warning("ITERATION %i , lhood_old : %f  >  lhood_v %f", niter, lhood_old, lhood_v)

		if convergence > 1e-5:
			converged_v = 0
			logging.info("variational inference ended with converge")
		else:
			converged_v = 1
			logging.info("variational inference ended without converge, but reached iteration limit")
		return (lhood_v, phi_v, log_phi_v, lambda_v, nu_v, zeta_v, niter, converged_v)

	def update_expected_ss(self, lambda_v, nu_v, phi_v, wordids, wordcts):
		'''
		Update sufficient statistics, mu, cov and beta.

		Arguments:
			variational paraments and doc paraments
		Returns:
			sufficient statistics
		'''
		logging.info("updating expected sufficient statistics")
		# covariance and mean suff stats
		for i in range(self._K):
			self.mu[i] = lambda_v[i]
			for j in range(self._K):
				lilj = lambda_v[i] * lambda_v[j]
				if i == j:
					self.cov[i, j] = self.cov[i, j] + nu_v[i] + lilj
				else:
					self.cov[i, j] = self.cov[i, j] + lilj
		# topics suff stats
		for i in range(self._W):
			for j in range(self._K):
				w = wordids[i]
				self.beta[j, w] = self.beta[j, w] + wordcts[i] * phi_v[i, j]
		# number of data
		self.ndata += 1

	'''
	estimate stage
	'''
	def em(self, docs):
		logging.info("running level 1 function : em")
		logging.info("checking model parameters, if none then load them from files")

		if self.nu is None:
			logging.info("load self.nu")
			with open('ctm_nu', 'rb') as ctm_nu_dump:
				self.nu = cPickle.load(ctm_nu_dump)
		if self.cov  is None:
			logging.info("load self.cov")
			with open('ctm_cov', 'rb') as ctm_cov_dump:
				self.cov = cPickle.load(ctm_cov_dump)
		if self.inv_cov  is None:
			logging.info("load self.inv_cov")
			with open('ctm_inv_cov', 'rb') as ctm_inv_cov_dump:
				self.inv_cov = cPickle.load(ctm_inv_cov_dump)
		if self.log_det_inv_cov  is None:
			logging.info("load self.log_det_inv_cov")
			with open('ctm_log_det_inv_cov', 'rb') as ctm_log_det_inv_cov_dump:
				self.log_det_inv_cov = cPickle.load(ctm_log_det_inv_cov_dump)
		if self.log_beta  is None:
			logging.info("load self.log_beta")
			with open('ctm_log_beta', 'rb') as ctm_log_beta_dump:
				self.log_beta = cPickle.load(ctm_log_beta_dump)

		iteration = 0
		convergence = 1.0
		lhood = lhood_old = 0.0
		reset_var = 1
		var_max_iter = 500

		corpus_lambda = np.zeros((self._D, self._K))
		corpus_nu = np.zeros((self._D, self._K))
		corpus_phi_sum = np.zeros((self._D, self._K))

		while ((iteration < 1000) and ((convergence > 1e-3) or (convergence < 0))):
			# e-step
			if np.mod(iteration, 20) == 0:
				logging.info("******em iteration no. %i******", iteration)
			lhood = self.expectation(self, reset_var, docs, corpus_lambda, corpus_nu, corpus_phi_sum)
			convergence = (lhood_old - lhood) / lhood_old

			# m-step
			if convergence <1e-3:
				logging.info("process convergenced at %f, quit", convergence)
				break
			if convergence < 0:
				reset_var = 0
				if var_max_iter > 0:
					var_max_iter += 10
				else:
					var_max_iter = var_max_iter / 10
			else:
				self.maximization(self)
				lhood_old = lhood
				reset_var = 1
				iteration += 1
			# old_conv = convergence

	def expectation(self, reset_var, docs, corpus_lambda, corpus_nu, corpus_phi_sum):
		''' E-step of EM algorithm
		Arguments:
			corpus: the docs needed to be worked on, need to get ids and cts
		Returns:
			sufficient statistics : lhood, self.mu, self.cov, self.beta, self.ndata
		'''
		logging.info("running expectation function")
		total_lhood = 0.0
		phi_sum = np.zeros(self._K)

		for d, doc in enumerate(self.corpus):
			wordidsd = [id for id, _ in doc]
			wordctsd = np.array([cnt for _, cnt in doc])

			if reset_var:
				logging.info("reset all variational parameters")
				phi_v = np.dot(1.0 / self._K, np.ones((self._K, self._W)))
				log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K, self._W)))
				zeta_v = 10.0
				nu_v = np.zeros(self._K)
				lambda_v = np.zeros(self._K)
				lhood_v = 0.0
				niter_v = 0
			else:
				phi_v = np.dot(1.0 / self._K, np.ones((self._K, self._W)))
				log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K, self._W)))
				zeta_v = 10.0
				lhood_v = 0.0
				niter_v = 0
				with open('corpus_lambda_dump', 'rb') as ctm_lambda_dump:
					lambda_v_c = cPickle.load(ctm_lambda_dump)
				with open('corpus_nu_dump', 'rb') as ctm_nu_dump:
					nu_v_c = cPickle.load(ctm_nu_dump)
				lambda_v = lambda_v_c[d]
				nu_v = nu_v_c[d]

			(lhood_v, phi_v, log_phi_v, lambda_v, nu_v, zeta_v, niter_v, converged_v) = self.var_inference(self, d, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)
			self.update_expected_ss(self, lambda_v, nu_v, phi_v, wordidsd, wordctsd)

			total_lhood += lhood_v
			corpus_lambda[d] = lambda_v
			corpus_nu[d] = nu_v

			for j in range(self._W):
				for n in range(self._K):
					phi_sum[n] = phi_v[j, n]

			corpus_phi_sum[d] = phi_sum
		return total_lhood

	# m-step
	def maximization(self):
		'''
		M-step of EM algorithm, use scikit.learn's LedoitWolf method to perfom
		covariance matrix shrinkage.
		Arguments:
			sufficient statistics, i.e. model parameters
		Returns:
			the updated sufficient statistics which all in self definition, so no return values
		'''
		logging.info("running maximization function")
		logging.info("mean maximization")
		mu = np.divide(self.mu, self.ndata)
		logging.info("covariance maximization")
		for i in range(self._K):
			for j in range(self._K):
				self.cov[i, j] = (1.0 / self.ndata) * self.cov[i, j] + self.ndata * mu[i] * mu[j] - self.mu[i] * mu[j] - self.mu[j] * mu[i]
		logging.info(" performing covariance shrinkage using sklearn module")
		lw = LedoitWolf()
		cov_result = lw.fit(self.cov, assume_centered=True).covariance_
		self.inv_cov = np.linalg.inv(cov_result)
		self.log_det_inv_cov = safe_log(np.linalg.det(self.inv_cov))

		logging.info("topic maximization")
		for i in range(self._K):
			sum_m = 0
			sum_m += np.sum(self.beta, axis=0)[i]

			if sum_m == 0:
				sum_m = -1000 * self._W
			else:
				sum_m = np.log(sum_m)

			for j in range(self._W):
				self.log_beta[i, j] = safe_log(self.beta[i, j] - sum_m)

		logging.info("write model parameters to file")
		logging.info("write gaussian")
		with open('ctm_nu', 'w') as ctm_nu_dump:
			cPickle.dump(self.nu, ctm_nu_dump)
		with open('ctm_cov', 'w') as ctm_cov_dump:
			cPickle.dump(self.cov, ctm_cov_dump)
		with open('ctm_inv_cov', 'w') as ctm_inv_cov_dump:
			cPickle.dump(self.inv_cov, ctm_inv_cov_dump)
		with open('ctm_log_det_inv_cov', 'w') as ctm_log_det_inv_cov_dump:
			cPickle.dump(self.log_det_inv_cov, ctm_log_det_inv_cov_dump)
		logging.info("write topic matrix")
		with open('ctm_log_beta', 'w') as ctm_log_beta_dump:
			cPickle.dump(self.log_beta, ctm_log_beta_dump)

	def inference(self):
		'''
		Perform inference on corpus (seen or unseen)
		load a model, and do approximate inference for each document in a corpus
		'''
		logging.info("running level 1 function : inference")
		logging.info("checking model parameters, if none then load them from files")

		if self.nu is None:
			logging.info("load self.nu")
			with open('ctm_nu', 'rb') as ctm_nu_dump:
				self.nu = cPickle.load(ctm_nu_dump)
		if self.cov  is None:
			logging.info("load self.cov")
			with open('ctm_cov', 'rb') as ctm_cov_dump:
				self.cov = cPickle.load(ctm_cov_dump)
		if self.inv_cov  is None:
			logging.info("load self.inv_cov")
			with open('ctm_inv_cov', 'rb') as ctm_inv_cov_dump:
				self.inv_cov = cPickle.load(ctm_inv_cov_dump)
		if self.log_det_inv_cov  is None:
			logging.info("load self.log_det_inv_cov")
			with open('ctm_log_det_inv_cov', 'rb') as ctm_log_det_inv_cov_dump:
				self.log_det_inv_cov = cPickle.load(ctm_log_det_inv_cov_dump)
		if self.log_beta  is None:
			logging.info("load self.log_beta")
			with open('ctm_log_beta', 'rb') as ctm_log_beta_dump:
				self.log_beta = cPickle.load(ctm_log_beta_dump)

		logging.info("initialize corpus level parameter")
		lhood_corpus = np.zeros(self._D)
		nu_corpus = np.zeros((self._D, self._K))
		lambda_corpus = np.zeros((self._D, self._K))
		phi_sums_corpus = np.zeros((self._D, self._K))

		logging.info("approximate inference")
		for i in range(self._D):
			logging.info("initialize the variational parameters")
			phi_v = np.dot(1.0 / self._K, np.ones((self._K, self._W)))
			log_phi_v = np.dot(-(np.log(self._K)), np.ones((self._K, self._W)))
			zeta_v = 0.0
			nu_v = np.zeros(self._K)
			lambda_v = np.zeros(self._K)
			logging.info("conducting variational inference")
			(lhood_corpus[i], phi_v, log_phi_v, lambda_corpus[i], nu_corpus[i], zeta_v, _) = self.var_inference(self, i, phi_v, log_phi_v, lambda_v, nu_v, zeta_v)

			phi_v_col_sum = np.sum(phi_v, axis=0)
			for j in range(self._K):
				phi_sums_corpus[i, j] += phi_v_col_sum[j]

		logging.info("write likelihood and some variational parameters to files")
		with open('ctm_lhood', 'w') as ctm_lhood_dump:
			cPickle.dump(lhood_corpus, ctm_lhood_dump)
		with open('corpus_lambda', 'w') as corpus_lambda_dump:
			cPickle.dump(lambda_corpus, corpus_lambda_dump)
		with open('corpus_nu', 'w') as corpus_nu_dump:
			cPickle.dump(nu_corpus, corpus_nu_dump)
		with open('phi_sums', 'w') as phi_sums_dump:
			cPickle.dump(phi_sums_corpus, phi_sums_dump)

	def sample_term(self, eta, lambda_v, nu_v, obs_wordidsd, obs_wordctsd):
		'''
		Importance sampling the likelihood based on the variational posterior

		Arguments:
			eta : natural parameter of logistic normal distribution
			theta : mean parameter of logistic normal distribution
			The mapping between them is equation 3 in the paper:
					eta[i] = log theta[i] / theta[K]
		Returns:
			value of p(w | eta) - q(eta)
		'''
		t1 = 0.5 * self.log_det_inv_cov
		t1 += -(0.5) * self._K * 1.837877  # 1.837877 is the natural logarithm of 2*pi
		for i in range(self._K):
			for j in range(self._K):
				t1 -= (0.5) * (eta[i] - self.mu[i]) * self.inv_cov[i, j] * (eta[j] - self.mu[j])
		# compute theta
		theta = eta[:]
		sum_t = np.sum(np.exp(eta))
		theta = np.divide(theta, sum_t)

		# compute word probabilities
		for n in range(self._W):
			word_term = 0
			for i in range(self._K):
				word_term += theta[i] * np.exp(self.log_beta[i, n])
			ids = obs_wordidsd.index(i)
			t1 += obs_wordctsd[ids] * safe_log(word_term)
		# log(q(\eta | lambda, nu))
		t2 = 0
		for i in range(self._K):
			t2 += stats.norm.pdf(eta[i] - lambda_v[i], np.sqrt(nu_v[i]))
		return(t1 - t2)

	def expected_theta(self, obs_wordidsd, obs_wordctsd, lambda_v, nu_v):
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
				eta[i] = random.gauss(0, np.sqrt(nu_v[i])) + lambda_v[i]
			# compute p(w | \eta) - q(\eta)
			log_prob = self.sample_term(self, eta, lambda_v, nu_v, obs_wordidsd, obs_wordctsd)
			# compute theta
			theta = eta[:]
			sum_t = np.sum(np.exp(eta))
			theta = np.divide(theta, sum_t)

			# update e_theta
			for i in range(self._K):
				e_theta[i] = log_sum(e_theta[i], log_prob + safe_log(theta[i]))
		# normalize e_theta and set return vector
		sum_et = -1.0
		for i in range(self._K):
			e_theta[i] -= np.log(nsamples)
			sum_et = log_sum(sum_et, e_theta[i])
		e_theta = np.exp(np.subtract(e_theta, sum_et))
		return e_theta

	def log_mult_prob(self, cts, e_theta):
		'''
		 log probability of the document under proportions theta and topics beta
		 used to calculate the held-out data's probability

		 '''
		val = 0
		for i in range(self._W):
		# here the number W should be the number of held-out data
		# log_beta should be initialized, not the old self.log_beta
			term_prob = 0
			for k in range(self._K):
				term_prob += e_theta[k] * np.exp(self.log_beta[k, i])
			# TODO: FIX THIS SELF.WORDCTS
			val += safe_log(term_prob) * self.wordcts[i]
		return val

	def get_perplexity(self):
		'''
		Calculate perplexity value. Read in the model parameters got from above procedures

		Right now, held out documents have to be manually put in a seperate folder to read in. 

		Returns:
			perplexity : currently, the only evaluation value, add others later

		'''

		log_lhood = np.zeros(len(self.corpus))
		e_theta = np.zeros((len(self.corpus),self._K))
		total_words = 0
		total_lhood = 0
		
		# create held_out corpus
		heldout_filenames = os.listdir('/Users/sean/personal_research_doc/lda_explained/state/heldout')
		heldout_corpus = []
		for thefile in heldout_filenames:
			with open("./state/heldout" + thefile, "rb") as f:
				strings = f.read()
				heldout_corpus.append(strings)
		(held_dictionary, held_corpus) = preprocess.get_dict_and_corp(heldout_corpus)
		total_words = len(held_dictionary)

		# load model parameters for calculating e_theta
		with open('corpus_lambda_dump', 'rb') as ctm_lambda_dump:
				lambda_v_c = cPickle.load(ctm_lambda_dump)
		with open('corpus_nu_dump', 'rb') as ctm_nu_dump:
				nu_v_c = cPickle.load(ctm_nu_dump)

		# calculate e_theta using observed data
		for d, doc in enumerate(self.corpus):
			obs_wordidsd = [id for id, _ in doc]
			obs_wordctsd = np.array([cnt for _, cnt in doc])

			lambda_v = lambda_v_c[d]
			nu_v = nu_v_c[d]

			# e_theta is calculated on each document
			# since obs_doc number is always no lesser than held_out doc number
			# so on the held out inference stage, e_theta won't indexed out
			e_theta[d] = self.expected_theta(self, obs_wordidsd, obs_wordctsd, lambda_v, nu_v)

		for d, doc in enumerate(held_corpus):
			# held_wordidsd = [id for id, _ in doc]
			held_wordctsd = np.array([cnt for _, cnt in doc])

			# approximate inference of held out data
			# randomly choose a number to index e_theta file
			# MEMO: this is dirty work, but right now, I can't think of a better way
			rand_etheta_index = np.random.randint(len(held_corpus))
			etheta = e_theta[rand_etheta_index]
			log_lhood[d] = self.log_mult_prob(self, held_wordctsd, etheta)

		total_lhood = np.sum(log_lhood)
		perplexity = np.exp(- total_lhood / total_words)
		print 'the perplexity is:', perplexity

if __name__ == '__main__':
	print 'This program is being run by itself'
else:
	print 'I am being imported from another module'
