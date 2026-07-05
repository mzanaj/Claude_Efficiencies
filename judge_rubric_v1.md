# Judge rubric — v1

This file is a versioned artifact: any edit means bumping the version (v2, v3, ...)
and re-running the harness so results stay comparable.

Write this rubric independently from the labeler prompt, starting from business
intent. Do NOT paste the labeler prompt here — the whole point is decorrelation.
The bike-sales content below is illustrative; replace it with your domain.

## Task
Assign exactly one label to a short text snippet: `positive`, `negative`, or `unsure`.

## What `positive` means
A concrete bike sales opportunity: the text carries explicit commercial signal
around a product — an offer, a price, availability, a stated intent to buy or
sell, or a purchase inquiry.
Example: "Bike model A is on sale."

## What `negative` means
No commercial opportunity, even when the text shares vocabulary with positives
(bikes, riding, gear, weather, hobbies).
Example: "Today the weather is quite nice, we should go for a bike ride."

## What `unsure` means
Genuinely ambiguous under the decision rules below: evidence points both ways,
or the commercial signal only exists if you assume unstated context.

## Decision rules (apply in order)
1. Judge intent, not keywords. Product words alone never make a positive.
2. A positive requires explicit commercial signal in the text itself: price,
   offer, availability, buy/sell intent, or purchase inquiry.
3. If the commercial signal requires assuming context that is not in the text,
   label `unsure`, not `positive`.
4. If the text spans multiple topics, label based only on the commercially
   relevant span.

## Edge cases
Every time the harness surfaces a disagreement the rules above don't settle,
add a line here and bump the version. Examples to decide now:
- Product mentioned in a personal or leisure context → `negative`
- Opinion question with no buying signal ("is model A any good?") → decide: `unsure` or `positive`, write it down
- Secondhand/marketplace chatter ("selling my old bike") → decide and write it down
