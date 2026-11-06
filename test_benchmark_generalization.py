"""No real game required; verifies split isolation and right-censoring metrics."""
from benchmark_generalization import reference_seeds, holdout_seeds, summarize


def run():
    ref = reference_seeds(100)
    held = holdout_seeds(1000, excluded=ref)
    assert len(ref) == 100 and len(held) == 1000 and not set(ref).intersection(held)
    assert ref == reference_seeds(100)
    assert held == holdout_seeds(1000, excluded=ref)
    rows = [
        {'v1_time_s':10.0,'v2_time_s':20.0,'v1_game_state':'game_over','v2_game_state':'playing',
         'v1_decisions':150,'v2_decisions':300,'v1_policy_compute_s':1.5,'v2_policy_compute_s':6.0,
         'v2_v2_fall_overrides':5},
        {'v1_time_s':15.0,'v2_time_s':12.0,'v1_game_state':'game_over','v2_game_state':'game_over',
         'v1_decisions':225,'v2_decisions':180,'v1_policy_compute_s':2.25,'v2_policy_compute_s':3.6,
         'v2_v2_fall_overrides':2}]
    s = summarize(rows, steps=300, bootstraps=100)
    assert s['episodes'] == 2 and s['v2_wins'] == 1 and s['v2_losses'] == 1
    assert s['v2_censored_at_limit'] == 1
    assert s['survivors_at_s']['20']['v2'] == 1
    assert '40' not in s['survivors_at_s']
    assert s['v1_compute_ms_per_decision_weighted'] == 10
    assert s['v2_compute_ms_per_decision_weighted'] == 20
    print('Generalization benchmark regression passed!')


if __name__ == '__main__':
    run()
