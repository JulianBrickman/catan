# Catan Requirements Audit

This file compares the requirements/spec captured in `Documents/catan_base_rules_research.md` against the current implementation in `catan.py`.

## Summary

Current status:
- The project now supports a largely playable base-game Catan rules engine with setup, turn flow, robber, resource production, building, maritime trading, simplified domestic trading, development cards, Largest Army, Longest Road, SVG rendering, CLI play, and a fairly broad regression test suite.
- Most of the rules in the research file are implemented.
- A few items are still missing or only partially modeled.
- A few behaviors now intentionally differ from the original research file because later design decisions changed the spec during development.

Test status:
- `python3 -m unittest -q test_catan.py` passes with 36 tests.

## Fully Implemented Against The Research File

### Core board and state
Implemented:
- 19 hexes, 54 intersections, 72 edges, 9 harbors.
- Terrain/resource mapping.
- Desert and robber start state.
- Development deck composition.
- Piece counts per player.
- Graph-based board representation.

### Setup phase
Implemented:
- Snake-order setup: `0-1-2-3-3-2-1-0`.
- Settlement then adjacent road.
- Distance Rule during setup.
- Second settlement does not require connection to the first.
- Second road must attach to the second settlement.
- Second settlement grants adjacent starting resources.
- Starting player gets the first normal turn after setup.

### Turn structure
Implemented:
- Explicit `turn_start`, `robber_discard`, `robber_move`, `main`, `road_building`, `game_over` phases.
- Roll-before-normal-actions flow.
- Development cards can be played before rolling.
- After rolling a 7, players must resolve discard/robber before continuing.

### Resource production
Implemented:
- All players with adjacent settlements/cities receive resources on matching rolls.
- Settlements produce 1.
- Cities produce 2.
- Robber blocks production on the occupied hex.
- Multiple adjacent owned intersections can produce multiple cards.

### Robber and 7 handling
Implemented:
- On 7, no normal production occurs.
- Players with more than 7 cards discard half, rounded down.
- Current player must move robber to a different hex.
- Current player chooses which opponent to rob if multiple are adjacent.
- Stolen resource is random from the chosen victim.
- Knight moves robber without forcing discard.
- Victims with zero cards are excluded from legal victims.

### Building legality and costs
Implemented:
- Roads cost brick + lumber.
- Settlements cost brick + lumber + wool + grain.
- Cities cost 3 ore + 2 grain.
- Development cards cost ore + wool + grain.
- Roads must connect to the player’s network.
- Opponent settlements/cities block road continuation.
- Settlements after setup require one of the player’s roads and satisfy the Distance Rule.
- Cities upgrade the player’s own settlement only.
- Settlement piece returns to supply when upgraded into a city.
- Piece exhaustion is enforced.

### Maritime trade
Implemented:
- 4:1 bank trades.
- 3:1 generic harbor trades.
- 2:1 specific harbor trades.
- Harbor access depends on owning a settlement/city on the harbor edge.
- Opponent harbor ownership does not grant the discount.
- Robber does not block harbor use.

### Development cards
Implemented:
- Buy development card from shuffled deck.
- Newly bought development cards are not playable until end-of-turn reveal.
- One development card may be played per turn.
- Knight.
- Road Building.
- Year of Plenty.
- Monopoly.
- Victory Point cards counted passively toward VP.
- Largest Army updates from played Knights.

### Longest Road / Largest Army
Implemented:
- Longest Road minimum length 5.
- Tie does not transfer Longest Road from current owner.
- Opponent settlement can break a road.
- Forks counted correctly as a single best branch.
- Loop behavior covered by tests.
- Largest Army first claimed at 3 Knights.
- Larger Knight count transfers Largest Army.
- Tie does not transfer Largest Army.

### Observation / simulator state
Implemented:
- Public and private player state.
- Board topology.
- Robber location.
- Development deck state.
- Bank resource counts.
- Turn controller state.
- Pending trade offer / responder state.
- Legal action export.

### Tooling / debugging
Implemented:
- Terminal CLI for manual play.
- SVG board rendering.
- Intersection labels and edge labels in SVG.
- Event log.
- Sample observation JSON.

## Implemented But Diverges From The Research File

These are real differences between `Documents/catan_base_rules_research.md` and current `catan.py`.

### Bank shortage rule
Research file says:
- If the bank cannot satisfy all players for a resource, nobody gets that resource.
- Exception: if the shortage affects only one player, that player gets the remainder.

Current implementation:
- If supply is short, remaining cards are distributed across all claim units with a random tie-break.
- This was added later by direct user instruction and is now tested.

Conclusion:
- `catan.py` does **not** currently match the bank-shortage rule in the research file.
- It instead matches the later custom project rule.

### Win timing
Research file says:
- A player who begins their own turn already at the winning threshold may win without rolling.

Current implementation:
- The engine follows your later custom rule instead of the research file here.
- A player does not passively win just by starting their turn already at 10+ points.
- Victory is checked when a point-changing event happens on the active player's turn.
- In your intended model, a player should only gain victory points on their own turn, so a player should never meaningfully begin a turn already above the threshold from a previous turn.
- Passive Victory Point development cards still count automatically when they are acquired on the player's own turn.
- Pre-roll development cards may still create a win if they change VP state, such as a Knight awarding Largest Army.

Conclusion:
- `catan.py` does **not** match the official start-of-turn win rule in the research file.
- It matches your custom "win on active-turn point gain" model.

### Domestic trading model
Research file describes broad domestic trading rules.
Current implementation:
- Only `1-for-1` domestic trades are modeled.
- Active player proposes one resource for one resource to all players.
- Other players respond in sequence with accept/decline.

Conclusion:
- This is a simplified subset of the domestic trade system in the research file.

## Partially Implemented / Simplified Relative To The Research File

### Domestic trade richness
Partially implemented:
- Only current player may propose a trade.
- Trade is offered to other players.
- Accept / decline flow exists.
- Same-resource trades are prevented.

Still simplified or missing:
- Multi-card offers are not supported.
- “Any positive exchange” is not modeled; only `1-for-1` is.
- No explicit “secret trades” concept because negotiation is not modeled beyond structured offers.
- No explicit “credit trade” representation.
- No explicit triangular trade representation.

### Victory Point card reveal behavior
Research file says VP cards stay hidden and are revealed on the player’s turn when needed to win.
Current implementation:
- VP cards are counted passively in `total_victory_points`.
- There is no separate explicit reveal action.

Conclusion:
- Functional for winning logic, but not modeled as a reveal event.

### Trade/build phase semantics
Research file notes strict trade/build phases vs combined phase.
Current implementation:
- Uses a combined post-roll main phase.

Conclusion:
- This is acceptable and consistent with the research file’s recommended implementation choice.



### Some explicit FAQ restrictions are not strongly modeled as distinct rules
Likely not explicitly enforced as separate mechanics:
- “No giving cards away for nothing” beyond the fact that only structured exchange actions exist.
- “No secret trades” is effectively outside the model, not explicitly enforced.


### Victory Point reveal UX
Missing:
- Explicit VP reveal event for human play / UI clarity.


## Likely Correct But Worth Further Targeted Tests

These are implemented areas that are historically bug-prone and still worth more direct scenario tests if exact faithfulness matters.

- Robber moved to desert with robbable victim adjacent.
- Knight after a rolled 7 in the same turn.
- Harbor timing when building onto a harbor and using it immediately in the same combined main phase.
- Multiple Victory Point cards contributing to a win in one turn.
- Exact domestic-trade sequencing when offerer’s resources change before later responders act.


If the question is “does it fully match every requirement in `Documents/catan_base_rules_research.md`?” the answer is:
- **Not completely**.
- The biggest mismatches are:
  - bank-shortage handling,
  - start-of-turn win timing,
  - simplified domestic trading,
  - no explicit Victory Point reveal action.

## Recommended Next Fixes If The Goal Is Strict Fidelity To The Research File

5. Add targeted tests for the remaining robber/harbor/dev-card edge cases.
