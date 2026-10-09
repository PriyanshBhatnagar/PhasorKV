"""Rate-distortion allocation: one format per unit under a bit budget.

Minimise sum_u D[u, f_u] subject to sum_u cost[u, f_u] <= budget. With a
Lagrange multiplier mu each unit independently picks argmin_f D + mu * cost,
which is optimal on the convex hull of every unit's (cost, D) points; mu is
found by bisection, and leftover budget is spent greedily on the upgrades with
the best distortion saved per bit. 'drop' (cost 0, D = the unit's energy) is
always on the menu, so rank and precision come out of the same choice.
"""
import torch


def allocate(D: torch.Tensor, cost: torch.Tensor, budget: float, iters: int = 80) -> torch.Tensor:
    """D, cost: [U, F] (float64). Returns the chosen format index per unit [U]."""
    D, cost = D.double(), cost.double()
    pick = lambda mu: torch.argmin(D + mu * cost, dim=1)
    total = lambda ch: cost.gather(1, ch[:, None]).sum().item()
    best = pick(0.0)
    if total(best) <= budget:
        return best
    lo, hi = 0.0, 1.0
    while total(pick(hi)) > budget:
        hi *= 4
        if hi > 1e30:
            raise ValueError("budget below the cheapest allocation")
    for _ in range(iters):
        mid = (lo * hi) ** 0.5 if lo > 0 else hi / 2
        if total(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
        if hi / max(lo, 1e-300) < 1 + 1e-9:
            break
    ch = pick(hi)
    # greedy fill: best distortion-per-bit upgrades that still fit
    spent = total(ch)
    cur_D = D.gather(1, ch[:, None])
    cur_c = cost.gather(1, ch[:, None])
    while True:
        dD = cur_D - D                     # gain of switching unit u to format f
        dc = cost - cur_c
        ok = (dD > 0) & (dc > 0) & (dc <= budget - spent)
        if not ok.any():
            break
        ratio = torch.where(ok, dD / dc, torch.full_like(dD, -1.0))
        u, f = divmod(int(torch.argmax(ratio)), D.shape[1])
        spent += dc[u, f].item()
        ch[u] = f
        cur_D[u, 0], cur_c[u, 0] = D[u, f], cost[u, f]
    return ch
