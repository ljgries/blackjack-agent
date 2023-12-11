import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
import random
import math

def cmp(a, b):
    return float(a > b) - float(a < b)

class Deck:
    def __init__(self, np_random):
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.np_random.shuffle(self.cards)

    def draw_card(self):
        if not self.cards:
            self.reset()
        return self.cards.pop()

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def scale_value(self, value):
        scaled_value = ((value - 6.0) / (7.8 - 6.0)) * 9
        return scaled_value
    
    def __str__(self):
        result = ""
        for card in self.cards:
            result += str(card) + "\n"
        return result
    
    def __len__(self):
        return len(self.cards)

    def calc_temperature(self):
        deck_value = sum(self.cards)
        avg_value = deck_value / len(self.cards)
        temperature = math.floor(self.scale_value(avg_value))
        return temperature

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    return sorted(hand) == [1, 10]

class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(10))
        )
        self.np_random = np.random.RandomState()
        self.deck = Deck(self.np_random)
        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit
            self.player.append(self.deck.draw_card())
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5
        temperature = self.deck.calc_temperature()
        if self.render_mode == "human":
            self.render()
        return (*self._get_obs(), temperature), reward, terminated, False, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.deck.reset()
        self.dealer = self.deck.draw_hand()
        self.player = self.deck.draw_hand()
        temperature = self.deck.calc_temperature()
        _, dealer_card_value, _ = self._get_obs()
        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)
        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)
        if self.render_mode == "human":
            self.render()
        return (*self._get_obs(), temperature), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 15)
        dealer_text = small_font.render("Dealer: " + str(dealer_card_value), True, white)
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(os.path.join("img", f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png"))
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (screen_width // 2 - card_img_width - spacing // 2, dealer_text_rect.bottom + spacing)
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (screen_width // 2 + spacing // 2, dealer_text_rect.bottom + spacing)
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (screen_width // 2 - player_sum_text.get_width() // 2, player_text_rect.bottom + spacing)
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (screen_width // 2 - usable_ace_text.get_width() // 2, player_sum_text_rect.bottom + spacing // 2)
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame
            pygame.display.quit()
            pygame.quit()
