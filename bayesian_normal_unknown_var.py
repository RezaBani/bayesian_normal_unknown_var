## Author: Reza Banivakil
## Estimating mean of a normal distribution with Bayesian approch
## formula for posterior hyperparameters (update method) from https://en.wikipedia.org/wiki/Conjugate_prior
##

import numpy as np
from scipy.stats import norm, invgamma
import matplotlib.pyplot as plt
plt.style.use("ggplot")

BANDIT_MEANS = [10, 12]
BANDIT_VARS = [1.5, 1.5]
N_SAMPLE = 2 # must be >= 2
NUM_TRIALS = 2000 + N_SAMPLE

class Bandit:
    def __init__(self, true_mean, true_var):
        self.true_mean = true_mean
        self.true_var = true_var
        self.loc = 0
        self.lambda_ = 0
        self.alpha = 0
        self.beta = 1
        self.sum_x = 0
        self.N = N_SAMPLE

    def pull(self):
        return np.random.normal(loc = self.true_mean, scale = np.sqrt(self.true_var), size = N_SAMPLE)

    def sample(self):
        return norm.rvs(self.loc, np.sqrt(invgamma.rvs(self.alpha, scale = self.beta)/self.lambda_))

    def update(self, x):
        self.sum_x, self.lambda_, self.alpha, self.loc, self.beta = (self.sum_x + x, self.lambda_ + len(x), self.alpha + len(x)/2,
        (sum(x)+self.lambda_*self.loc)/(self.lambda_+len(x)),
        self.beta + (len(x)*np.std(x) + (self.lambda_*len(x)*(np.mean(x)-self.loc)**2)/(self.lambda_+len(x)))/2)

def plot(bandits, trial):
    x = np.linspace(0, 10, 201)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for b in bandits:
        y = invgamma.pdf(x, b.alpha, scale = b.beta)
        low_var = invgamma.ppf(0.025, b.alpha, scale = b.beta)
        mid_var = invgamma.ppf(0.50, b.alpha, scale = b.beta)
        up_var = invgamma.ppf(0.975, b.alpha, scale = b.beta)
        low_mean = norm.ppf(0.025, b.loc, scale = np.sqrt(mid_var/b.lambda_))
        up_mean = norm.ppf(0.975, b.loc, scale = np.sqrt(mid_var/b.lambda_))
        ax.plot(x, y, label=f"real var: {b.true_var:.4f}, predicted var: {low_var:.4f} , {up_var:.4f}\nreal mean: {b.true_mean:.4f}, predicted mean: {low_mean:.4f} , {up_mean:.4f}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(mean,var) for mean,var in zip(BANDIT_MEANS,BANDIT_VARS)]
    sample_points = np.array([len(bandits),5,10,15,20,30,40,50,100,150,200,300,400,500,1000,1500,1998]) + N_SAMPLE
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(0,NUM_TRIALS,N_SAMPLE):
        # Thompson sampling
        if i < len(bandits)*N_SAMPLE:
            j = int(i/N_SAMPLE)
        else:
            j = np.argmax([b.sample() for b in bandits])

        # plot the posteriors
        if i in sample_points:
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        
        # update rewards
        rewards[i : i + N_SAMPLE] = x
        
        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each 
    for b in bandits:
        low_var = invgamma.ppf(0.025, b.alpha, scale = b.beta)
        mid_var = invgamma.ppf(0.50, b.alpha, scale = b.beta)
        up_var = invgamma.ppf(0.975, b.alpha, scale = b.beta)
        low_mean = norm.ppf(0.025, b.loc, scale = np.sqrt(mid_var/b.lambda_))
        up_mean = norm.ppf(0.975, b.loc, scale = np.sqrt(mid_var/b.lambda_))
        print(f"95% Chance that variance is between: ({low_var:.4f}, {up_var:.4f})\tTrue variance is: {b.true_var:.4f}")
        print(f"Based on middle value of variance {mid_var:.4f} estimate of mean is as follows:")
        print(f"95% Chance that mean is between: ({low_mean:.4f}, {up_mean:.4f})\tTrue variance is: {b.true_mean:.4f}")
        print("="*60)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("mean of rewards:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [(b.lambda_) for b in bandits])


if __name__ == "__main__":
    experiment()
