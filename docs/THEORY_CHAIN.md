# Theory dependency chain (v2, 2026-11-05)

This note records the exact dependency chain used in `paper/main.tex` §3.  The goal is a
rigorous but not over-strong set of assumptions.  There is no abstract
`Gen(D,n,delta)` and no `Theorem` that depends on it.

## Objects

For a prefix distribution `pi`:

- `q_pi = P_pi(f(x) != h_Q(x))`, the greedy top-1 rejection rate.
- `delta(x) = P_x(x^(1)) - P_x(x^(2))`, the target top-1 margin.
- `d(x) = TV(P_x, Q_x)`.
- `M_pi(eps) = P_pi(delta <= 2 eps)`.

## Unconditional statements

### Lemma 1 (margin--TV upper bound)

For every `eps > 0`,

```
q_pi <= M_pi(eps) + E_pi[d] / eps.                       (1)
```

Proof: if the top-1 disagrees, then `delta <= 2 d`; the disagreement event is contained in
`{delta <= 2 eps} union {d > eps}`; apply Markov to the second event.

This is deterministic.  No model class, no training assumption, no uniform convergence.

### Proposition 1 (two-sided shift inequality)

For any two prefix distributions `pi, mu`,

```
q_mu - TV(pi, mu) <= q_pi <= q_mu + TV(pi, mu).          (2)
```

Proof: the 0--1 loss is bounded in `[0,1]`, so its expectation is 1-Lipschitz in TV.

This is deterministic and assumption-free.  It is the only place where the measurement
distribution enters.  It replaces the `Gen(D,n,delta)` term of the old draft.

## Empirical assumptions

### Assumption 1 (margin tail)

There exist `c, gamma, eps0 > 0` such that

```
M_pi(eps) <= c eps^gamma      for all 0 < eps <= eps0.   (3)
```

This is an empirical regularity of the target margin distribution.  It is checked on the
small-margin tail; the fit window and `R^2` are reported in Appendix D.

### Assumption 2 (TV capacity law)

There exist `A, beta > 0` and `0 < D1 < D2` such that

```
E_pi[d] <= A D^{-beta}        for all D in [D1, D2].     (4)
```

This is a local empirical law over the measured capacity range, not an extrapolation.

## Conditional capacity bound

### Proposition 2 (upper bound)

Under Assumptions 1--2, define

```
eps_D = (E_pi[d] / (c gamma))^(1/(gamma+1)).
```

If `eps_D <= eps0`, then

```
q_pi <= K E_pi[d]^(gamma/(gamma+1))
     <= K A^(gamma/(gamma+1)) D^(-beta gamma/(gamma+1)),   (5)

K = (1 + 1/gamma) (c gamma)^(1/(gamma+1)).
```

Proof: minimize `c eps^gamma + E[d]/eps`; the derivative gives `eps_D`; substitution gives
`K`.  The condition `eps_D <= eps0` ensures Assumption 1 applies at the optimum; otherwise
restrict `eps` to `(0, eps0]` and enlarge the constant.

For `gamma = 1`, this is `q_pi <= 2 sqrt(c E_pi[d])`, i.e. the square-root law

```
q_pi <~ D^(-beta/2).                                      (6)
```

For `gamma > 1`, the exponent `beta gamma/(gamma+1)` is closer to `beta`; attenuation weakens.

## Conditional lower bound

### Proposition 3 (lower bound)

If `q_mu >= q0` and `TV(pi, mu) <= t`, then

```
q_pi >= q0 - t.                                           (7)
```

This is immediate from (2).  It is only as strong as the available lower bound on `q_mu`
and upper bound on `TV(pi, mu)`.  We do **not** have a universal positive `q_mu`; therefore
we do not claim a universal positive lower bound on OOD rejection.  We report empirical
one-sided confidence bounds instead.

## Finite-sample evaluation (computable)

For `n` i.i.d. tokens from `mu`, Hoeffding gives, with probability at least `1-delta`,

```
|q_hat_mu - q_mu| <= sqrt(log(2/delta) / (2n)).            (8)
```

Combining (2) and (8):

```
q_hat_mu - sqrt(log(2/delta)/(2n)) - TV(pi,mu)
  <= q_pi <=
q_hat_mu + sqrt(log(2/delta)/(2n)) + TV(pi,mu).           (9)
```

The only remaining unknown is `TV(pi,mu)`, for which one needs an estimate or an upper
bound.  This is the only sampling/generalization term in the paper; it is computable and
does not depend on a hypothesis-class capacity.

## What is and is not claimed

**Unconditional:** Lemma 1 and Proposition 1 (and (9) given i.i.d. sampling and a TV bound).

**Conditional:** Proposition 2 (on Assumptions 1--2) and Proposition 3 (on `q0` and `t`).

**Not claimed:**

- a universal positive lower bound on `q_pi`;
- a numerical certificate from the Markov relaxation (it exceeds 1 on raw OOD at 27B);
- a total-parameter super-capacity result (no draft has more total parameters than the target);
- a sampling-matched cache (the real standard-SD loop is reported separately).
