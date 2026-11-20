# GTA VI trophy rarity forecaster

`main.py` forecasts, for each of the game's 55 trophies, the percent of tracked players who
will have earned it at a future target time, as five quantiles (5/25/50/75/95%).

**States, not reads.** Exophase refreshes its counts in jumps hours apart, so most reads in a
case are the same snapshot repeated. History is compressed to distinct states first (null
rates carried forward), and everything is fitted on those states only.

**Two views, averaged on the logit scale.** Time runs on x = ln(1 + hours since release),
because earn rates saturate. (a) *Trend view*: weighted least squares of logit(rate) on x —
weights sqrt(players) × 36 h-half-life recency — with the slope shrunk toward the pooled
slope of the trophy's tier by n/(n+4); flat or short series borrow the tier slope, anchored
on their last logit. (b) *Earners view*: the same fit on ln(earners), divided by a log-fit of
players_tracked at the target, separating audience growth from earning share.

**Sanity and spread.** Medians pass tier caps (platinum ≤ 3%, the 100%-completion gold ≤ 8%,
…) and a floor for story-progression trophies. Sigma combines the trend fit's residual
variance with a horizon term; quantiles are a normal read off the logit scale.

**Never invalid.** Missing data degrades to flat rows, and any internal failure falls back to
a persistence forecast for that case rather than crashing.
