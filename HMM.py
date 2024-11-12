

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
        metgoal = False
        transitions = []
        observations = []
        states = [i for i in self.transitions['#']]
        probabilities = [float(self.transitions['#'][i]) for i in self.transitions['#']]
        curr_state = random.choices(states, weights=probabilities, k=1)[0]
        i = 0
        while i < n:
            emissionStates = [i for i in self.emissions[curr_state]]
            emissionProbabilities = [float(self.emissions[curr_state][i]) for i in self.emissions[curr_state]]
            transitions.append(curr_state)
            observation = random.choices(emissionStates, weights=emissionProbabilities, k=1)
            observations.append(observation[0])
            if self.transitions.get(curr_state) is None:
                metgoal = True
                return transitions, observations, metgoal
            states = [i for i in self.transitions[curr_state]]
            probabilities = [float(self.transitions[curr_state][i]) for i in self.transitions[curr_state]]
            curr_state = random.choices(states, weights=probabilities, k=1)[0]
            i+=1


        """return an n-length Sequence by randomly sampling from this HMM."""
        return transitions, observations, metgoal

    def forward(self, sequence):
        T = len(sequence.stateseq)
        states = list(self.transitions.keys())
        M = numpy.zeros((T, len(states)))

        #Initialize for first observation
        for s_idx, s in enumerate(states):
            initial_prob = float(self.transitions.get('#', {}).get(s, 0))
            emission_prob = float(
                self.emissions.get(s, {}).get(sequence.stateseq[0], 0))
            M[0, s_idx] = initial_prob * emission_prob

        for t in range(1, T):
            for s_idx, s in enumerate(states):
                total_prob = 0
                for prev_s_idx, prev_s in enumerate(states):
                    transition_prob = float(self.transitions.get(prev_s, {}).get(s, 0))  # Transition probability
                    emission_prob = float(self.emissions.get(s, {}).get(sequence.stateseq[t], 0))  # Emission probability
                    total_prob += M[t - 1, prev_s_idx] * transition_prob * emission_prob
                M[t, s_idx] = total_prob
        most_probable_states = []
        for t in range(T):
            max_state_idx = numpy.argmax(M[t, :])
            most_probable_states.append(states[max_state_idx])

        #Give a list of the probable states, with the resulting being the last
        sequence.outputseq = most_probable_states

    def viterbi(self, sequence):
        T = len(sequence.stateseq)
        states = list(self.transitions.keys())
        N = len(states)
        #Use numpy to make matrices full of zeroes initiailzed
        M = numpy.zeros((T, N))
        backpointers = numpy.zeros((T, N), dtype=int)

        # Step 1: Initialization
        for s_idx, s in enumerate(states):
            initial_prob = float(self.transitions.get('#', {}).get(s, 0))  #Transition probability for the first observation
            emission_prob = float(
                self.emissions.get(s, {}).get(sequence.stateseq[0], 0)) #Emission probability for the first observation
            M[0, s_idx] = initial_prob * emission_prob

        for t in range(1, T):
            for s_idx, s in enumerate(states):
                max_prob = -1e6 #Make it a really small probability to compare against before changing it
                best_prev_state = 0

                for prev_s_idx, prev_s in enumerate(states):
                    transition_prob = float(self.transitions.get(prev_s, {}).get(s, 0))
                    emission_prob = float(self.emissions.get(s, {}).get(sequence.stateseq[t], 0))
                    prob = M[t - 1, prev_s_idx] * transition_prob * emission_prob

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_s_idx

                M[t, s_idx] = max_prob
                backpointers[t, s_idx] = best_prev_state

        state_sequence = []
        final_state_idx = numpy.argmax(M[T - 1, :])
        state_sequence.append(states[final_state_idx])

        for t in range(T - 2, -1, -1):
            final_state_idx = backpointers[t + 1, final_state_idx]
            state_sequence.insert(0, states[final_state_idx])
        sequence.outputseq = state_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename")
    parser.add_argument("--generate", type=int, help="Amount to generate")
    parser.add_argument("--forward", type=str, help="Forward algorithm")
    parser.add_argument("--viterbi", type=str, help="Viterbi algorithm")
    args = parser.parse_args()
    h = HMM()
    h.load(args.filename)
    if args.generate:
        t, o, success = h.generate(5)
        print(success)
        print("Transitions: ")
        [print(item, end=' ') for item in t]
        print("\nObservations: ")
        [print(item, end=' ') for item in o]
        print("\n")
    if args.forward:
        sequences = []
        outputs = []
        with open(args.forward, 'r') as file:
            for line in file:
                obs_sequence = line.strip().split()
                if line != "\n":
                    sequences.append(Sequence(obs_sequence, outputs))
        for i, sequence in enumerate(sequences, start=1):
            h.forward(sequence)
            print(f"Most probable state: {sequence.outputseq[-1]}")
            if sequence.outputseq[-1] == '#':
                print("Success! Landing spot found!")
            else:
                print("Landing spot not found..")
    if args.viterbi:
        sequences = []
        outputs = []
        with open(args.viterbi, 'r') as file:
            for line in file:
                obs_sequence = line.strip().split()
                if line != "\n":
                    sequences.append(Sequence(obs_sequence, outputs))
        for i, sequence in enumerate(sequences, start=1):
            h.viterbi(sequence)
            print(f"Most probable sequence for {sequence.stateseq}:\n{sequence.outputseq}")