import numpy as np
import pandas as pd

class DataGenerator(object):

    def __init__(self, statuses, frequency, results, regular):
        self.statuses = statuses
        self.frequency = frequency
        self.results = results
        self.regular = regular
        self.combinations = []
        self.combination_probs = []

    def add_specific(self, combination, prob):
        combination = [status in combination for status in self.statuses]
        if not combination in self.combinations:
            self.combinations.append(combination)
            self.combination_probs.append(prob)

    def generate_data(self):
        status = np.random.binomial(1, self.frequency)
        result_probs = np.array(self.regular)
        for i in range(len(self.combinations)):
            if status[self.combinations[i]].all():
                result_probs += self.combination_probs[i]
        result_probs = np.clip(result_probs, 0.0, 1.0)
        result = np.random.binomial(1, result_probs)
        retval = np.concatenate([status, result])
        return retval

def main():
    statuses = ["sA", "sB", "sC", "sD"]
    frequency = [0.5, 0.5, 0.5, 0.5]
    results = ["r1", "r2", "r3"]
    reguler = [0.01, 0.01, 0.01]
    generator = DataGenerator(statuses, frequency, results, reguler)
    combination1 = ["sA", "sB"]
    prob1 = [0.5, 0.0, 0.0]
    generator.add_specific(combination1, prob1)
    combination2 = ["sC", "sD"]
    prob2 = [0.0, 0.5, 0.0]
    generator.add_specific(combination2, prob2)
    combination3 = ["sA", "sD"]
    prob3 = [0.0, 0.5, 0.5]
    generator.add_specific(combination3, prob3)
    num_data = 100000
    val = [generator.generate_data() for _ in range(num_data)]
    val = np.array(val)
    df = pd.DataFrame(val, columns=statuses+results)
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()