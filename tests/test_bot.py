"""Unit tests for the Clash Royale tactical decision engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from use_fonction.bot import bot, PLAYABLE_X_MIN, PLAYABLE_X_MAX, PLAYABLE_Y_MIN, PLAYABLE_Y_MAX


def make_state(
    in_game=True,
    elixir=6,
    hand=None,
    units=None,
    enemy_right_alive=True,
    enemy_left_alive=True,
):
    if not in_game:
        return [False]
    if hand is None:
        hand = [
            "chevalier_vignette.jpg",
            "archere_vignette.jpg",
            "fireball_vignette.jpg",
            "PK_vignette.jpg",
        ]
    state = [
        True,
        [0, "elixir :", elixir],
        [1, [278, 204], "alive tower" if enemy_right_alive else "destroyed tower", [47, 63]],
        [1, [72, 204], "alive tower" if enemy_left_alive else "destroyed tower", [47, 63]],
        [1, [72, 493], "alive tower", [48, 57]],
        [1, [278, 493], "alive tower", [47, 56]],
    ]
    for i, card in enumerate(hand):
        state.append([0, f"carte {i}", card])
    for pos, label in (units or []):
        state.append([1, list(pos), label, [40, 40]])
    return state


def test_no_action_when_out_of_game():
    agent = bot()
    assert agent.get_action(make_state(in_game=False), now=10.0) == []


def test_counters_giant_with_mini_pekka():
    agent = bot()
    state = make_state(
        elixir=6,
        hand=[
            "chevalier_vignette.jpg",
            "archere_vignette.jpg",
            "fireball_vignette.jpg",
            "PK_vignette.jpg",
        ],
        units=[([110, 430], "geant_red")],
    )
    action = agent.get_action(state, now=10.0)
    assert len(action) == 2
    assert action[0] == 3  # Mini PEKKA is top counter for Giant in matrice_choix_def_response
    x, y = action[1]
    assert PLAYABLE_X_MIN <= x <= PLAYABLE_X_MAX
    assert PLAYABLE_Y_MIN <= y <= PLAYABLE_Y_MAX


def test_casts_aoe_spell_on_enemy_cluster():
    agent = bot()
    state = make_state(
        elixir=5,
        hand=[
            "chevalier_vignette.jpg",
            "archere_vignette.jpg",
            "fireball_vignette.jpg",
            "geant_vignette.jpg",
        ],
        units=[([320, 410], "archere_red"), ([345, 425], "mousquetaire_red")],
    )
    action = agent.get_action(state, now=10.0)
    assert len(action) == 2
    assert action[0] == 2  # Fireball on dense cluster


def test_supports_allied_giant_push_with_musketeer():
    agent = bot()
    state = make_state(
        elixir=6,
        hand=[
            "chevalier_vignette.jpg",
            "mousquetaire_vignette.jpg",
            "fireball_vignette.jpg",
            "fleches_vignette.jpg",
        ],
        units=[([115, 390], "geant_blue")],
    )
    action = agent.get_action(state, now=10.0)
    assert len(action) == 2
    assert action[0] == 1  # Musketeer behind allied Giant
    assert action[1][1] > 390


def test_proactive_push_targets_remaining_enemy_tower():
    agent = bot(proactive_elixir_threshold=8)
    state = make_state(
        elixir=9,
        hand=[
            "geant_vignette.jpg",
            "archere_vignette.jpg",
            "fireball_vignette.jpg",
            "fleches_vignette.jpg",
        ],
        enemy_left_alive=False,
        enemy_right_alive=True,
    )
    action = agent.get_action(state, now=10.0)
    assert len(action) == 2
    assert action[0] == 0  # Giant in the back
    assert action[1][0] > 222  # Right lane where enemy tower is still alive


def test_cooldown_prevents_double_play():
    agent = bot(action_cooldown=1.5)
    state = make_state(elixir=9, units=[([110, 430], "geant_red")])
    first = agent.get_action(state, now=10.0)
    second = agent.get_action(state, now=10.5)
    third = agent.get_action(state, now=12.0)
    assert first != []
    assert second == []
    assert third != []
