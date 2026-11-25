"""
Nom du fichier: bot.py
Auteur: Étienne-Théodore PRIN (base), Hippolyte Valentin (stratégie tactique)
Date: 2026-03-18

Description:
Moteur de décision tactique pour le bot Clash Royale.
Exploite l'état extrait par la pipeline Computer Vision (YOLOv8 + VGG10 + template matching)
pour arbitrer entre défense réactive (matrice de counters + AoE spells), renfort de push,
et gestion proactive de l'élixir (anti-leak).
"""

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

from use_fonction.configuration.matrice_def import (
    card_dictionnary,
    dict_trad,
    dictionnaire_translation,
    elixir_price,
    matrice_choix_def_response,
)

# Géométrie de l'arène normalisée (hauteur = 800 px, largeur ≈ 445 px)
ARENA_MID_X = 222
RIVER_Y = 360
PLAYABLE_X_MIN, PLAYABLE_X_MAX = 55, 390
PLAYABLE_Y_MIN, PLAYABLE_Y_MAX = 395, 625

BRIDGE_POS = {"left": (115, 405), "right": (330, 405)}
BACK_POS = {"left": (115, 610), "right": (330, 610)}

SPELL_CARDS = {"fireball", "flèches"}
RANGED_SUPPORT_CARDS = {"mousquetaire", "archers", "gargouilles"}
TANK_CARDS = {"géant", "chevalier"}

# Priorité de menace intrinsèque par unité ennemie
UNIT_THREAT_WEIGHT = {
    "géant": 1.6,
    "mini_PK": 1.5,
    "mousquetaire": 1.3,
    "chevalier": 1.2,
    "gargouilles": 1.15,
    "archers": 1.0,
}


@dataclass
class HandCard:
    slot: int
    name: str
    card_id: int
    cost: int


@dataclass
class DetectedUnit:
    name: str
    unit_id: int
    x: int
    y: int
    lane: str
    is_enemy: bool


class bot:
    """
    Agent décisionnel temps réel pour Clash Royale.

    Stratégie hiérarchique en 4 niveaux :
      1. Défense AoE : utilise un sort (Fireball / Flèches) sur un groupe d'ennemis dense.
      2. Défense ciblée : sélectionne le meilleur counter disponible via la matrice de matchups
         et calcule un placement adapté (kite au centre, support à distance, ou sort anticipé).
      3. Renfort de push : soutient un tank allié (Géant / Chevalier) avec une unité à distance.
      4. Initiative & anti-leak d'élixir : lance un push ou cycle une carte peu coûteuse dès que
         l'élixir atteint le seuil haut, en ciblant une voie avec une tour ennemie encore debout.
    """

    def __init__(self, action_cooldown: float = 1.2, proactive_elixir_threshold: int = 8):
        self.action_cooldown = action_cooldown
        self.proactive_elixir_threshold = proactive_elixir_threshold
        self.last_action_time = 0.0
        self._lane_toggle = "left"

    def get_action(self, state, now: Optional[float] = None) -> list:
        """
        Choisit l'action optimale à partir de l'état courant du jeu.

        Args:
            state: état produit par Screen_analyse.get_state(img).
            now: horodatage optionnel (utile pour les tests déterministes).

        Returns:
            [] si aucune action n'est prise, sinon [index_carte (0..3), [x, y]].
        """
        if not state or not state[0]:
            return []

        current_time = time.time() if now is None else now
        if current_time - self.last_action_time < self.action_cooldown:
            return []

        elixir, hand, enemies, allies, enemy_towers = self._parse_state(state)

        # Fallback si aucune carte n'a été reconnue mais l'élixir déborde
        if not hand:
            if elixir >= 9:
                lane = self._choose_attack_lane(enemy_towers)
                self.last_action_time = current_time
                return [0, list(BACK_POS[lane])]
            return []

        # 1. Défense réactive contre les menaces ennemies
        action = self._defensive_action(elixir, hand, enemies)
        if action:
            self.last_action_time = current_time
            return action

        # 2. Renfort d'un push allié existant
        action = self._support_push_action(elixir, hand, allies)
        if action:
            self.last_action_time = current_time
            return action

        # 3. Initiative offensive / anti-leak d'élixir
        if elixir >= self.proactive_elixir_threshold:
            action = self._proactive_action(elixir, hand, enemy_towers)
            if action:
                self.last_action_time = current_time
                return action

        return []

    def _parse_state(self, state):
        elixir = 0
        if len(state) > 1 and len(state[1]) > 2:
            try:
                elixir = int(state[1][2])
            except (TypeError, ValueError):
                elixir = 0

        # Tours : indices 2 (ennemi droite), 3 (ennemi gauche), 4 (allié gauche), 5 (allié droite)
        enemy_towers = {"right": True, "left": True}
        if len(state) >= 6:
            enemy_towers["right"] = state[2][2] == "alive tower"
            enemy_towers["left"] = state[3][2] == "alive tower"

        # Cartes en main : indices 6..9
        hand: List[HandCard] = []
        if len(state) >= 10:
            for slot_idx, entry in enumerate(state[6:10]):
                vignette_name = entry[2] if len(entry) > 2 else 0
                card_name = card_dictionnary.get(vignette_name)
                if card_name and card_name in dict_trad:
                    card_id = dict_trad[card_name]
                    hand.append(HandCard(slot_idx, card_name, card_id, elixir_price[card_id]))

        # Unités détectées par YOLOv8 : indices 10+
        enemies: List[DetectedUnit] = []
        allies: List[DetectedUnit] = []
        for entry in state[10:]:
            if len(entry) < 3 or entry[0] != 1:
                continue
            pos, raw_label = entry[1], str(entry[2])
            if raw_label in ("msg", "horloge_blue", "horloge_red", "fire_ball", "fleches"):
                continue

            translated = dictionnaire_translation.get(raw_label)
            if not translated or translated not in dict_trad:
                continue

            x, y = int(pos[0]), int(pos[1])
            lane = "left" if x < ARENA_MID_X else "right"
            is_enemy = raw_label.endswith("_red")
            is_ally = raw_label.endswith("_blue")

            unit = DetectedUnit(translated, dict_trad[translated], x, y, lane, is_enemy)
            if is_enemy:
                enemies.append(unit)
            elif is_ally:
                allies.append(unit)

        return elixir, hand, enemies, allies, enemy_towers

    def _defensive_action(
        self, elixir: int, hand: List[HandCard], enemies: List[DetectedUnit]
    ) -> Optional[list]:
        threats = [e for e in enemies if e.y >= RIVER_Y - 50]
        if not threats:
            return None

        affordable = [c for c in hand if c.cost <= elixir]
        if not affordable:
            return None

        # 1a. Valeur AoE : sort si >= 2 ennemis regroupés dans un rayon de 95 px
        cluster_center = self._find_enemy_cluster(threats, radius=95)
        if cluster_center is not None:
            for spell_name in ("fireball", "flèches"):
                spell_card = next((c for c in affordable if c.name == spell_name), None)
                if spell_card:
                    tx = self._clamp(cluster_center[0], PLAYABLE_X_MIN, PLAYABLE_X_MAX)
                    ty = self._clamp(cluster_center[1] + 20, 140, PLAYABLE_Y_MAX)
                    return [spell_card.slot, [tx, ty]]

        # 1b. Sélection de la menace la plus urgente
        top_threat = max(threats, key=self._threat_score)
        preferred_counter_ids = matrice_choix_def_response[top_threat.unit_id]

        # Choisit le meilleur counter disponible en main
        chosen_card = None
        for counter_id in preferred_counter_ids:
            match = next((c for c in affordable if c.card_id == counter_id), None)
            if match is not None:
                chosen_card = match
                break

        if chosen_card is None:
            return None

        pos = self._compute_defensive_position(chosen_card, top_threat)
        return [chosen_card.slot, pos]

    def _support_push_action(
        self, elixir: int, hand: List[HandCard], allies: List[DetectedUnit]
    ) -> Optional[list]:
        if elixir < 5:
            return None

        pushing_tanks = [a for a in allies if a.name in TANK_CARDS and a.y <= RIVER_Y + 120]
        if not pushing_tanks:
            return None

        lead_tank = min(pushing_tanks, key=lambda u: u.y)
        support_cards = [
            c for c in hand if c.name in RANGED_SUPPORT_CARDS and c.cost <= elixir
        ]
        if not support_cards:
            return None

        # Priorise Mousquetaire > Archers > Gargouilles
        priority = {"mousquetaire": 0, "archers": 1, "gargouilles": 2}
        chosen = min(support_cards, key=lambda c: priority.get(c.name, 99))

        px = self._clamp(lead_tank.x, PLAYABLE_X_MIN, PLAYABLE_X_MAX)
        py = self._clamp(max(lead_tank.y + 75, PLAYABLE_Y_MIN), PLAYABLE_Y_MIN, PLAYABLE_Y_MAX)
        return [chosen.slot, [px, py]]

    def _proactive_action(
        self, elixir: int, hand: List[HandCard], enemy_towers: dict
    ) -> Optional[list]:
        affordable = [c for c in hand if c.cost <= elixir and c.name not in SPELL_CARDS]
        if not affordable:
            affordable = [c for c in hand if c.cost <= elixir]
        if not affordable:
            return None

        lane = self._choose_attack_lane(enemy_towers)

        # Priorise un tank en fond de terrain pour construire un push
        tank = next((c for c in affordable if c.name == "géant"), None) or next(
            (c for c in affordable if c.name == "chevalier"), None
        )
        if tank:
            return [tank.slot, list(BACK_POS[lane])]

        # Sinon cycle la carte non-sort la moins chère en fond de terrain
        cheapest = min(affordable, key=lambda c: c.cost)
        pos = BRIDGE_POS[lane] if cheapest.name == "mini_PK" else BACK_POS[lane]
        return [cheapest.slot, list(pos)]

    def _choose_attack_lane(self, enemy_towers: dict) -> str:
        left_alive = enemy_towers.get("left", True)
        right_alive = enemy_towers.get("right", True)
        if left_alive and not right_alive:
            return "left"
        if right_alive and not left_alive:
            return "right"
        lane = self._lane_toggle
        self._lane_toggle = "right" if lane == "left" else "left"
        return lane

    def _compute_defensive_position(self, card: HandCard, threat: DetectedUnit) -> List[int]:
        if card.name in SPELL_CARDS:
            # Anticipation du déplacement vers le bas
            tx = self._clamp(threat.x, PLAYABLE_X_MIN, PLAYABLE_X_MAX)
            ty = self._clamp(threat.y + 25, 140, PLAYABLE_Y_MAX)
            return [tx, ty]

        if card.name in RANGED_SUPPORT_CARDS:
            # Placement sécurisé en retrait de la menace
            tx = self._clamp(threat.x, PLAYABLE_X_MIN, PLAYABLE_X_MAX)
            ty = self._clamp(max(threat.y + 75, PLAYABLE_Y_MIN + 25), PLAYABLE_Y_MIN, PLAYABLE_Y_MAX)
            return [tx, ty]

        # Kite / interception vers le centre de l'arène pour activer le feu croisé des tours
        pull_x = int(threat.x * 0.65 + ARENA_MID_X * 0.35)
        tx = self._clamp(pull_x, PLAYABLE_X_MIN, PLAYABLE_X_MAX)
        ty = self._clamp(max(threat.y + 35, PLAYABLE_Y_MIN), PLAYABLE_Y_MIN, PLAYABLE_Y_MAX)
        return [tx, ty]

    @staticmethod
    def _threat_score(unit: DetectedUnit) -> float:
        weight = UNIT_THREAT_WEIGHT.get(unit.name, 1.0)
        proximity = max(0.1, (unit.y - (RIVER_Y - 100)) / 300.0)
        return weight * proximity

    @staticmethod
    def _find_enemy_cluster(
        enemies: List[DetectedUnit], radius: int = 95
    ) -> Optional[Tuple[int, int]]:
        if len(enemies) < 2:
            return None
        best_group: List[DetectedUnit] = []
        for anchor in enemies:
            group = [
                other
                for other in enemies
                if math.hypot(anchor.x - other.x, anchor.y - other.y) <= radius
            ]
            if len(group) > len(best_group):
                best_group = group
        if len(best_group) >= 2:
            cx = int(sum(u.x for u in best_group) / len(best_group))
            cy = int(sum(u.y for u in best_group) / len(best_group))
            return (cx, cy)
        return None

    @staticmethod
    def _clamp(val: int, low: int, high: int) -> int:
        return max(low, min(high, int(val)))
