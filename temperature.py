import random
import gym
import numpy as np
import math
import sys

print("running")

ranks = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "jack": 10,
    "queen": 10,
    "king": 10,
    "ace": (1, 11),
}


class Card:
    def __init__(self, value):
        self.value = value


class Deck:
    def __init__(self, number_decks=1):
        self.cards = []
        for i in range(number_decks):
            for i in range(4):
                for rank, value in ranks.items():
                    self.cards.append(Card(value))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop(0)

    def peek(self):
        if len(self.cards) > 0:
            return self.cards[0]

    def add_to_bottom(self, card):
        self.cards.append(card)

    """
    Found via simulation:
    Deciles:
    [7.15217391 7.22454708 7.26267281 7.29019608 7.31168831 7.33529412
    7.36538462 7.40740741 7.47916667]
     Min: ~ 6.354166666666667 Max: ~ 8.242105263157894
    """

    def categorize(self, value):
        # Define the breakpoints for the categories (calculated as deciles of distribution)
        # This approach ensures approximately equal distribution in states across categories
        # and thus should theoretically maximize the utility of the heuristic.
        deciles = [
            7.14893617,
            7.22043011,
            7.26219512,
            7.28911565,
            7.30964467,
            7.33218467,
            7.36220472,
            7.40425532,
            7.47742178,
            8.242105263157894,
        ]

        # Find the category for the given value
        for i, breakpoint in enumerate(deciles):
            if value < breakpoint:
                return i

        return 9  # Return the last category if value is not less than any breakpoint

    def calc_temperature(self):
        avg_value = self.get_avg()

        temperature = self.categorize(avg_value)
        return temperature

    def get_avg(self):
        deck_value = 0
        for card in self.cards:
            if isinstance(card.value, tuple):
                # Treat ace as 11 (not one) in temperature calculations
                _, value = card.value
            else:
                # Take the value as is
                value = card.value
            deck_value += value
        return float(deck_value) / len(self)

    def __str__(self):
        result = ""
        for card in self.cards:
            result += str(card) + "\n"
        return result

    def __len__(self):
        return len(self.cards)


def main(sim_name, num_simulations):
    num_decks = 6
    shuffle_at = 0.7
    draws_between_shuffle = math.floor(num_decks * 52 * shuffle_at)
    if sim_name == "FindRange":
        findRange(num_simulations, draws_between_shuffle)
    if sim_name == "FindProbabilities":
        findProbabilities(num_simulations, draws_between_shuffle)


# Min: ~ 6.354166666666667 Max: ~ 8.242105263157894
def findRange(num_simulations, draws_between_shuffle):
    deck = Deck(number_decks=6)
    deck.shuffle()
    values = []
    index = 0

    for i in range(num_simulations):
        if i % 7000 == 0:
            # Calculate completion percentage
            completion_percentage = (i / num_simulations) * 100
            sys.stdout.write(f"\r{completion_percentage:.2f}% complete")
            sys.stdout.flush()

        avg_value = deck.get_avg()
        values.append(avg_value)

        card = deck.deal()
        index += 1
        if index > draws_between_shuffle:
            deck = Deck(number_decks=6)
            deck.shuffle()
            index = 0

    # Calculate deciles
    deciles = np.percentile(values, np.arange(10, 100, 10))
    print("Deciles:", deciles)
    print("Min:", min(values), "Max:", max(values))

    return deciles, min(values), max(values)


def findProbabilities(num_simulations, draws_between_shuffle):
    deck = Deck(number_decks=6)
    deck.shuffle()

    count = np.zeros((10, 10))
    index = 0

    temp_count = np.zeros(10)  # Array to track the count of each temperature

    for i in range(num_simulations):
        if i % 7000 == 0:
            completion_percentage = (i / num_simulations) * 100
            sys.stdout.write(f"\r{completion_percentage:.2f}% complete")
            sys.stdout.flush()

        temp = deck.calc_temperature()
        temp_count[temp] += 1  # Increment the count for the current temperature
        card = deck.deal()
        if isinstance(card.value, tuple):
            _, value = card.value
        else:
            value = card.value
        # Adjust indices based on the actual ranges
        count[temp][value - 2] += 1
        index += 1
        if index > draws_between_shuffle:
            deck = Deck(number_decks=6)
            deck.shuffle()
            index = 0

    probabilities = np.zeros((10, 10))

    for i in range(10):
        total = np.sum(count[i])
        if total > 0:
            probabilities[i] = count[i] / total
        else:
            probabilities[i] = np.zeros(10)  # or another placeholder if appropriate

    for i in range(10):
        print(f"Temperature {i}:")
        print(f"Count: {temp_count[i]}")
        print(
            f"Implied odds for each card at this temp after: {num_simulations} simulations:"
        )
        for j in range(10):
            print(f"Card value {j+2}: {probabilities[i][j]:.2%}")
        total_probability = np.sum(probabilities[i])
        print(f"Law of total probability: {total_probability:.2f} (Should equal 1)")

    # Currently saved with 10000000 simulations
    np.save("probabilities.npy", probabilities)


if __name__ == "__main__":
    main("FindProbabilities", 10000000)
