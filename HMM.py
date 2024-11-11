

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        emissions = {}
        transitions = {}
        #Read the emissions
        with open(basename+".emit", 'r') as file:
            emissionsFile = file.read()
            for line in emissionsFile.split("\n"):
                currLines = line.split(" ")
                emissionState = currLines[0]
                emissionProbability = currLines[-1]
                for state in currLines[1:-1]:
                    if emissionState not in emissions:
                        emissions[emissionState] = {}
                    emissions[emissionState][state] = emissionProbability
        #Read the transitions
        with open(basename+".trans", 'r') as file:
            transitionsFile = file.read()
            for line in transitionsFile.split("\n"):
                currLines = line.split(" ")
                transitionState = currLines[0]
                transitionProbability = currLines[-1]
                for state in currLines[1:-1]:
                    if transitionState not in transitions:
                        transitions[transitionState] = {}
                    transitions[transitionState][state] = transitionProbability
        self.emissions = emissions
        self.transitions = transitions

    ## you do this.
    def generate(self, n):
        transitions = []
        observations = []
        states = [i for i in self.transitions['#']]
        probabilities = [float(self.transitions['#'][i]) for i in self.transitions['#']]
        curr_state = random.choices(states, weights=probabilities, k=1)[0]
        i = 0
        while i < 20:
            emissionStates = [i for i in self.emissions[curr_state]]
            emissionProbabilities = [float(self.emissions[curr_state][i]) for i in self.emissions[curr_state]]
            transitions.append(curr_state)
            observation = random.choices(emissionStates, weights=emissionProbabilities, k=1)
            observations.append(observation[0])
            states = [i for i in self.transitions[curr_state]]
            probabilities = [float(self.transitions[curr_state][i]) for i in self.transitions[curr_state]]
            curr_state = random.choices(states, weights=probabilities, k=1)[0]
            i+=1


        """return an n-length Sequence by randomly sampling from this HMM."""
        return transitions, observations

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename")
    parser.add_argument("--generate", type=int, help="Amount to generate")
    args = parser.parse_args()
    h = HMM()
    h.load(args.filename)
    t, o = h.generate(args.generate)
    print("Transitions: ")
    [print(item, end=' ') for item in t]
    print("\nObservations: ")
    [print(item, end=' ') for item in o]
    print("\n")