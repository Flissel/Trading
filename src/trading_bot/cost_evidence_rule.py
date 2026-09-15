"""The one admissible reading of a Binance cost journal receipt, in one place.

Section 5 of ``docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md``
fixed this sentence before any number existed. It is a leaf module - it imports
nothing from ``trading_bot`` - so the journal that publishes a receipt and the
family declaration that cites one bind the very same words without importing
each other: a receipt may not be published under another rule, and a measured
family may not cite one.
"""

DECLARATION_RULE = (
    "a v3 family sets `slippage_bps_per_side_tier_<t>` in its base table to "
    "`tier_p50_of_p50` of the *worse leg* at 5 000 USDT and in its adverse table to "
    "`tier_p50_of_p90` at 50 000 USDT, rounded **up** to the next whole basis point, and "
    "cites the receipt hash. No other reading of the receipt is admissible for a declaration."
)
