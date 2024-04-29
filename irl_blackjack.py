import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
import random
import math
import time


def cmp(a, b):
    return float(a > b) - float(a < b)


class DeckTracker:
    def __init__(self, number):
        self.number = number
        self.reset()

    def reset(self):
        self.cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * self.number
        self.temperature = self.calc_temperature()

    def scale_value(self, value):
        scaled_value = ((value - 6.0) / (7.8 - 6.0)) * 9
        return scaled_value

    def shuffle(self):
        print("Shuffling the deck...")
        self.reset()

    def __str__(self):
        result = ""
        for card in self.cards:
            result += str(card) + "\n"
        return result

    def __len__(self):
        return len(self.cards)

    def this_card_dealt(self, value):
        try:
            self.cards.remove(value)
        except ValueError:
            print(f"Card {value} not in deck.")
        self.temperature = self.calc_temperature()
        return self.temperature

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
            deck_value += card
        return float(deck_value) / len(self)

    def process_message(self, message):
        if message == "shuffle":
            self.shuffle()
        else:
            self.this_card_dealt(message)


def simulate_card_stream(deck_tracker):
    # This function would interface with your computer vision system
    for _ in range(100):  # simulate 100 messages
        message = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "shuffle"])
        deck_tracker.process_message(message)
        print(f"Processed {message}, Temperature: {deck_tracker.temperature}")


class EventDispatcher:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def emit(self, event_type, *args, **kwargs):
        results = []
        for handler in self.handlers.get(event_type, []):
            result = handler(*args, **kwargs)
            if result is not None:
                results.append(result)
        if results:
            return results[-1]  # Return the result from the last handler


if __name__ == "__main__":
    deck_tracker = DeckTracker(6)
    dispatcher = EventDispatcher()

    # Register event handlers
    dispatcher.subscribe("card_dealt", deck_tracker.this_card_dealt)
    dispatcher.subscribe("shuffle", deck_tracker.shuffle)

    # Simulating events
    events = [
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "card_dealt",
        "shuffle",
    ]
    cards = [5, 9, 10, 11, 11, 2, 2, 2, 2, 4, 9, 10, 3, 3, 3]

    for event in events:
        if event == "shuffle":
            dispatcher.emit(event)
            print("Updated Deck Temperature: " + str(temperature))
            print("Updated Deck Temperature: " + str(deck_tracker.temperature))
        elif event == "card_dealt" and cards:
            print("Card dealt: " + str(cards[0]))
            temperature = dispatcher.emit(event, cards.pop(0))
            print("Updated Deck Temperature: " + str(temperature))
            print("Updated Deck Temperature: " + str(deck_tracker.temperature))
        time.sleep(1)  # Simulate time delay between events
