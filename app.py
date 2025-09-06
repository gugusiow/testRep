from flask import Flask, request, jsonify
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any

app = Flask(__name__)

@app.route('/payload_crackme', methods=['GET'])
def get_payload():
    return "111-1111111"

@app.route('/payload_sqlinject', methods=['GET'])
def get_sql():
    return "Alice'; UPDATE salary SET salary=999999 WHERE name='Alice'; --"

@app.route('/payload_stack', methods=['GET'])
def get_stack():
    return "congratulations!_you_got_the_flag!"

@app.route('/chasetheflag', methods=['POST'])
def chase_flags():
    flags = {
        "challenge1": "your_flag_1",
        "challenge2": "your_flag_2",
        "challenge3": "your_flag_3",
        "challenge4": "your_flag_4",
        "challenge5": "your_flag_5"
    }
    return jsonify(flags), 201


####### mages gambit start
def earliest_clear_time(intel: List[List[int]], reserve: int, stamina: int) -> int:
    """Compute earliest total minutes to clear all waves and end in cooldown.

    Assumptions:
    - intel is a sequence of [front, mana_cost] and must be processed in order.
    - Each processed intel requires spending mana_cost mana and 1 stamina (a "spell").
    - Starting (or switching) to a front costs 10 minutes (target + AOE cast).
      If the previous processed intel was on the same front and no cooldown occurred in between,
      extending costs 0 additional minutes (still consumes mana + stamina).
    - Cooldown costs 10 minutes and restores mana to full reserve and stamina to full.
    - Final state MUST be immediately after a cooldown (one last 10-minute cooldown after final wave).
    - Cooldown is only taken when required (cannot cast next wave due to mana/stamina) since taking it
      early never reduces time (it always adds 10 and doesn't avoid a future 10 when switching fronts).
    """

    waves: List[Tuple[int, int]] = [(f, c) for f, c in intel]
    n = len(waves)

    @lru_cache(maxsize=None)
    def dp(i: int, mana: int, stam: int, last_front: Optional[int]) -> int:
        # All waves cleared -> mandatory final cooldown
        if i == n:
            return 10  # last cooldown

        front, cost = waves[i]
        best = float('inf')

        can_cast = (cost <= mana and stam > 0)

        if can_cast:
            time_cost = 0 if last_front == front else 10
            best = min(best, time_cost + dp(i + 1, mana - cost, stam - 1, front))

        # If we cannot cast we must cooldown (restore resources, lose front continuity)
        if not can_cast:
            best = min(best, 10 + dp(i, reserve, stamina, None))

        return best

    return dp(0, reserve, stamina, None)


@app.route('/the-mages-gambit', methods=['POST'])
def get_gambit():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    if not isinstance(payload, list):
        return jsonify({"error": "Top-level JSON must be a list of scenario objects"}), 400

    results: List[Dict[str, Any]] = []
    required_fields = ["intel", "reserve", "stamina", "fronts"]

    for idx, scenario in enumerate(payload):
        if not isinstance(scenario, dict):
            return jsonify({"error": f"Scenario at index {idx} is not an object"}), 400
        missing = [f for f in required_fields if f not in scenario]
        if missing:
            return jsonify({"error": f"Scenario {idx} missing fields: {', '.join(missing)}"}), 400

        intel = scenario["intel"]
        reserve = scenario["reserve"]
        stamina = scenario["stamina"]
        fronts = scenario["fronts"]  # currently unused but validated

        # Validations
        if not isinstance(intel, list) or any(not isinstance(x, list) or len(x) != 2 for x in intel):
            return jsonify({"error": f"Scenario {idx}: intel must be list of [front, mana_cost]"}), 400
        if not all(isinstance(f, int) and isinstance(c, int) and f >= 0 and c > 0 for f, c in intel):
            return jsonify({"error": f"Scenario {idx}: each intel entry must be positive integers [front>=0, mana_cost>0]"}), 400
        if not (isinstance(reserve, int) and reserve > 0):
            return jsonify({"error": f"Scenario {idx}: reserve must be positive int"}), 400
        if not (isinstance(stamina, int) and stamina > 0):
            return jsonify({"error": f"Scenario {idx}: stamina must be positive int"}), 400
        if not (isinstance(fronts, int) and fronts >= 0):
            return jsonify({"error": f"Scenario {idx}: fronts must be non-negative int"}), 400

        time_needed = earliest_clear_time(intel, reserve, stamina)
        results.append({"time": time_needed})

    return jsonify(results)

###### mages gambit end

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      