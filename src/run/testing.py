'''
Created on Apr 3, 2021

@author: immanueltrummer
'''
from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn
from environment.single_doc import TuningEnv
from benchmark.evaluate import OLAP
from dbms.postgres import PgConfig
from doc.collection import DocCollection

# Create environment
dbms = PgConfig(db='tpch', user='immanueltrummer')
docs = DocCollection('../../manuals/AllSentences2.csv', dbms='pg')
benchmark = OLAP(
    dbms, '/Users/immanueltrummer/git/literateDBtuners/benchmarking/tpch/q2rep.sql')
# for p in docs.passages_by_doc[1]:
    # print(f'{p}\n')
    
env = TuningEnv(docs, dbms, benchmark)
env = GymEnvironment(env)

# set device
device = 'cpu'

# set writer
writer = DummyWriter()

def make_model(env):
    return nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )
model = make_model(env)
    
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)

# create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)

# create a Policy object derived from the Q-function
policy = GreedyPolicy(q, env.action_space.n, epsilon=0.1)

# instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)

# start experiment
run_experiment(dqn(model_constructor=make_model), env, 10000)

# print out benchmark statistics
benchmark.print_stats()

#env.dbms.close_conn()
#env.close()