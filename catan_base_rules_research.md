# Settlers of Catan (Base Game) Rules Research

## Scope
This document summarizes the **base 3-4 player version** of *CATAN / Settlers of Catan* using the official rulebook and official base-game FAQ, with emphasis on details that matter for implementing a correct simulator for reinforcement learning.

I excluded 5-6 player rules, Seafarers, Cities & Knights, Traders & Barbarians, and other variants unless an FAQ entry helped clarify a base-game edge case.

## Primary sources
- Official rules page: https://www.catan.com/understand-catan/game-rules
- Official base-game rules PDF: https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf
- Official base-game FAQ: https://www.catan.com/faq/basegame
- OpenSpiel overview: https://openspiel.readthedocs.io/en/stable/intro.html
- NFSP paper summary / open-access mirror: https://discovery.ucl.ac.uk/id/eprint/1523603/
- Deep CFR paper: https://proceedings.mlr.press/v97/brown19b.html
- ReBeL / imperfect-information self-play + search: https://papers.nips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
- Example Catan RL project: https://settlers-rl.github.io/

## Core game facts
- Players: 3-4.
- Win condition: first player to reach **10 victory points on their own turn** wins immediately.
- Board contents:
  - 19 terrain hexes.
  - 18 number tokens.
  - 9 harbor pieces.
  - 95 resource cards.
  - 25 development cards.
  - 1 robber.
- Terrain/resource mapping:
  - Hills -> brick
  - Forest -> lumber
  - Mountains -> ore
  - Fields -> grain
  - Pasture -> wool
  - Desert -> produces nothing
- Number tokens use values 2-12 except 7.
- The robber starts on the desert.

## Development deck composition
The official rulebook states there are 25 development cards:
- 14 Knight cards
- 6 Progress cards
  - 2 Road Building
  - 2 Year of Plenty
  - 2 Monopoly
- 5 Victory Point cards

## Pieces per player
Each player has:
- 5 settlements
- 4 cities
- 15 roads

This matters for legality: once a player runs out of a piece type, they cannot build more of that type until a piece returns to supply, such as when a settlement is upgraded into a city.

## Board geometry and placement vocabulary
For a simulator, the important graph objects are:
- **Hexes / terrain tiles**: produce resources.
- **Intersections**: where 3 hexes meet, or where land meets the frame on the coast.
- **Paths / edges**: where roads can be built.

Placement rules:
- Settlements and cities are built only on intersections.
- Roads are built only on paths.
- At most 1 road may occupy a path.
- A city replaces an existing settlement.

## Set-up phase
The official rules define a two-round snake draft setup.

### Set-up order
1. Determine a starting player.
2. **Round one** proceeds clockwise: each player places 1 settlement, then 1 adjacent road.
3. **Round two** proceeds counterclockwise starting with the player who placed last in round one: each player places 1 settlement, then 1 adjacent road.
4. After placing the **second** settlement, that player immediately collects 1 resource for each adjacent non-desert hex.
5. The starting player from round one takes the first normal turn after setup.

### Important setup legality
- The **Distance Rule** applies during setup too.
- The second settlement does **not** need to connect to the first settlement.
- The second road must attach to the second settlement.
- A settlement may be placed at a harbor during setup.

## Distance Rule
This is one of the most important legality constraints.

A settlement may only be placed on an unoccupied intersection if **all 3 adjacent intersections are empty of settlements/cities**, regardless of owner.

Equivalent implementation rule:
- For every occupied intersection, all neighboring intersections are forbidden settlement locations.

The FAQ clarifies this must hold **at all times during the game** for both settlements and cities.

## Normal turn structure
On your turn, in the basic rules, the sequence is:
1. Roll for resource production.
2. Trade.
3. Build.

You may play **one** development card at any time during your turn, including before rolling.

The rulebook also notes advanced players often merge trade/build into a combined phase. The FAQ clarifies certain answers differ depending on whether you model a strict separate trade phase or a combined trade/build phase, so your simulator should choose one convention and keep it consistent.

For an RL environment, I would recommend a **combined trade/build action phase** after the dice roll because it is simpler and matches how many experienced players actually play.

## Resource production
At the start of each normal turn, the current player rolls 2 dice.

If the roll is not 7:
- Every terrain hex whose number matches the roll produces.
- Each adjacent settlement gives its owner 1 matching resource.
- Each adjacent city gives its owner 2 matching resources.
- A single player can receive multiple cards from the same hex if they have multiple adjacent settlements/cities.
- If the robber is on that hex, it produces nothing.

### Bank shortage rule
The official rulebook says:
- If the supply does not contain enough of a resource to satisfy **all** players entitled to that resource, nobody gets that resource that turn.
- Exception: if the shortage affects only **one** player, that player gets whatever remains and the rest is lost.

This is subtle and worth implementing exactly.

## Rolling a 7 and the robber
If the roll is 7:
- No terrain produces resources.
- Every player with more than 7 resource cards discards **half**, rounded down.
- Then the current player must move the robber to a **different** terrain hex or to the desert.
- Then the current player steals 1 random resource card from one player with a settlement/city adjacent to that destination hex.
- If multiple opponents are adjacent, the current player chooses which opponent to rob.
- If the chosen opponent has 0 cards, nothing is stolen.
- After resolving the robber, the turn continues into trade/build.

FAQ clarifications:
- You may **not** trade before resolving a rolled 7.
- You may move the robber to the desert.
- If you move the robber to the desert after a 7 or a Knight, you **still** steal from an adjacent player if one exists.
- You may not pick up the robber and return it to the same hex.
- A Knight card does **not** cause players with more than 7 cards to discard; that discard rule applies only when a 7 is rolled.

## Trading
There are two trade classes.

### Domestic trade
- Only the current player may trade with others.
- Other players may not trade with each other during someone else's turn.
- Trades can involve any positive exchange of resources.
- You may not give resources away for nothing.
- You may not trade matching resources, like 3 ore for 1 ore.
- Secret trades are not allowed according to the FAQ.
- Credit trades are not allowed.
- Triangular trades are not allowed.

### Maritime trade
- Any player on their own turn may always trade 4 identical resources for 1 chosen resource.
- With a 3:1 harbor, the player may trade 3 identical resources for 1 chosen resource.
- With a 2:1 harbor, the player may trade 2 of the pictured resource for 1 chosen resource.
- To use a harbor, the player must own a settlement or city on a coastal intersection touching that harbor.
- You cannot use a harbor just because an opponent has a settlement there.
- The robber does **not** block harbor usage.

### Harbor timing edge case
The rulebook's combined trade/build explanation says you can use a harbor on the same turn you build onto it.
The FAQ says:
- In a strict separate trade phase model: no.
- In a combined trade/build model: yes.

For an RL simulator, this is an implementation choice you should lock down early.

## Building costs and legality
### Roads
Cost:
- 1 brick
- 1 lumber

Legality:
- A new road must connect to one of your own roads, settlements, or cities.
- Roads cannot be built through an opponent's settlement or city.
- Your own settlements/cities do **not** interrupt your own road network.
- An interrupted road may still be extended from a legal connected endpoint.
- Roads may be built along the coast.

### Settlements
Cost:
- 1 brick
- 1 lumber
- 1 wool
- 1 grain

Legality after setup:
- Must connect to at least one of your own roads.
- Must satisfy the Distance Rule.
- You cannot build disconnected new settlements elsewhere on the map after setup.

### Cities
Cost:
- 3 ore
- 2 grain

Legality:
- A city can only upgrade one of your existing settlements.
- You cannot build a city directly onto an empty intersection.
- Upgrading returns the old settlement piece to your supply.

### Development cards
Cost:
- 1 ore
- 1 wool
- 1 grain

Legality:
- Draw from the top of the shuffled development deck.
- Cards stay hidden in hand until played or revealed.
- You may buy multiple development cards on your turn if you can pay.
- You may play only **one** development card per turn.
- You may **not** play a development card on the same turn you bought it.
- Exception: if the newly purchased card is a Victory Point card that immediately gives you 10+ points on your turn, it may be revealed and you win.
- Development cards cannot be traded or given away.

## Development card effects
### Knight
- Move the robber exactly as if a 7 had been rolled, except nobody discards for hand size.
- Then steal 1 random resource from an adjacent player if possible.
- Played Knight cards stay face-up in front of the player.
- Knights may be played before rolling.

### Road Building
- Immediately place 2 free roads.
- They must obey normal road-building legality.
- The FAQ clarifies this card can be played during trade; in a strict phase model it does not forcibly end trading.

### Year of Plenty
- Take any 2 resource cards from the supply.
- They may be the same or different resources.
- Those cards can be used later in the same turn.

### Monopoly
- Name a resource type.
- Every other player gives you all cards of that type from their hand.
- If a player has none, they give nothing.
- The FAQ says opponents do not have to reveal their whole hand for verification; the game assumes honesty.

### Victory Point cards
- Stay hidden until revealed.
- They are only revealed on your own turn when they give you the points needed to win.
- The FAQ clarifies that you may reveal multiple VP cards at once to win.

## Longest Road
Rules from the rulebook and FAQ together:
- The special card is worth 2 victory points.
- To claim it, a player must have a **continuous road of at least 5 segments**.
- If another player builds a strictly longer qualifying road, they take the card.
- If another player merely ties the current owner, the current owner keeps it.
- Forks do not all count; count only one continuous branch.
- A loop is allowed; a circular road still counts by total connected segments.
- Your own settlements/cities do not break your road.
- An opponent's settlement/city placed on an intersection along your road **does** break it.
- If the previous owner's road is broken and nobody uniquely has the longest qualifying road, the card may be set aside until one player uniquely qualifies again.

This is one of the trickiest graph computations in the simulator.

## Largest Army
- Worth 2 victory points.
- First player to play 3 Knight cards gets it.
- If another player later has more played Knights than the current owner, that player takes it.
- Only **played** Knight cards count, not unplayed ones in hand.

## Victory points summary
Public VP sources:
- Settlement: 1
- City: 2
- Longest Road: 2
- Largest Army: 2

Hidden VP sources:
- Each Victory Point development card: 1

Win rule:
- A player wins immediately upon having 10+ victory points **on their own turn**.
- If a player technically reaches 10 during another player's turn, they must wait until their own turn to claim victory.
- The FAQ also confirms you do **not** have to roll if you begin your turn already at the winning threshold.

## Edge cases from the official FAQ that matter for a simulator
### Roads and connectivity
- You cannot build a road “through” another player's settlement/city.
- After setup, roads and settlements must connect to your existing network.
- A settlement may be built on an intersection of your own road; your own settlement does not interrupt your road.
- You may build a settlement adjacent to an interrupted road, as long as you have your own adjacent road.

### Robber and blocking
- The robber blocks only resource production from that hex.
- The robber does **not** stop building on adjacent intersections/edges.
- The robber does **not** block harbor access.
- You may move the robber twice in one turn: once due to a rolled 7, and again by playing a Knight.

### Trading restrictions
- No trade before resolving a rolled 7.
- No giving away cards for services or favors.
- No on-credit trades.
- No secret trades.
- No triangular trades.

### Development timing
- You may play a development card before rolling.
- If you do, you still may not play a second development card later that turn.
- You may buy as many development cards as you can afford on your turn.

### Resource handling
- Players may not voluntarily take fewer produced resources.
- If you must discard after a 7 and still have more than 7 cards afterward, you do **not** discard again.

## Minimal simulator state for RL
A correct base-game simulator needs at least:
- Board topology:
  - hex list with terrain types and current number tokens
  - intersection graph
  - edge graph
  - harbor assignments
- Per-player public state:
  - roads, settlements, cities
  - played Knights
  - Longest Road ownership
  - Largest Army ownership
  - public victory points
- Hidden/private state:
  - resource hand
  - development card hand
- Shared stochastic state:
  - robber location
  - remaining resource bank counts
  - remaining development deck order
  - dice outcome process
- Turn controller:
  - active player
  - subphase / action context
  - whether a development card has already been played this turn
  - whether a 7 or Knight is awaiting robber move / steal resolution
  - pending trade offers if modeling explicit domestic negotiation

## Action-space implications for RL
Catan is much harder than chess-like self-play because it is:
- **multiplayer** rather than 2-player,
- **general-sum** rather than zero-sum,
- **stochastic** because of dice and deck order,
- **imperfect-information** because hands and dev cards are hidden,
- **combinatorial** because trading and building create large legal-action sets.

That means a direct AlphaZero clone is usually a poor first approach.

## Recommended first approach for an RL project
If the goal is to build a strong learning agent without getting stuck for months in environment complexity, I would do this in stages.

### Stage 1: Build a faithful simulator first
Before training any agent:
- implement the complete base-game rules above,
- expose an action-mask API for legal moves,
- add deterministic seeds,
- write exhaustive tests for setup legality, robber resolution, Longest Road, Largest Army, and harbor trading.

This is the highest-leverage step. In practice, most Catan RL difficulty is environment correctness, not the first policy network.

### Stage 2: Start with simplified opponents and reduced mechanics
For the first trainable version, I would strongly consider temporarily simplifying one or both of:
- domestic player-to-player trading,
- full natural-language or negotiation-style trade offers.

A very practical progression is:
1. No domestic trading, only bank/harbor trading.
2. Then add a small discrete trade-offer action set.
3. Only later add richer negotiation.

This mirrors what several student and hobby Catan AI projects do because trading causes the action space to explode.

### Stage 3: Use self-play, but not pure adversarial AlphaZero-style self-play
Because base Catan is multiplayer and imperfect-information, better starting points are:
- PPO or A2C with action masking in self-play against a population of scripted and learned agents,
- NFSP-style ideas for imperfect information,
- Deep CFR style ideas if you want a game-theoretic route,
- search-augmented imperfect-information methods inspired by ReBeL if you later want a more advanced system.

My recommendation for a first serious build:
- **Environment**: custom simulator with legal action masks.
- **Training setup**: population-based self-play, not just latest-vs-latest.
- **Algorithm**: masked PPO as a baseline.
- **Evaluation**: Elo / match win rate / average VP against fixed baselines.

Why this is the best starting point:
- It is much easier to implement than Deep CFR or ReBeL.
- It works naturally with stochastic multiplayer games.
- It lets you bootstrap from heuristic opponents.
- You can progressively widen the action space instead of solving everything at once.

## Suggested milestone plan for your project
1. Implement a rules-correct engine for the base game.
2. Add a few scripted bots:
   - random legal bot
   - greedy build bot
   - heuristic settlement-placement bot
3. Train a masked PPO agent in a simplified version without domestic trading.
4. Add domestic trading with a heavily discretized action space.
5. Move to population self-play and opponent pools.
6. Only after that consider search or game-theoretic algorithms.

## Concrete recommendation
If we were building this together from scratch, I would not begin with a giant end-to-end neural network trying to learn full base Catan immediately.

I would begin with:
- a **fast, tested simulator**,
- a **graph-based board representation**,
- **action masking** for every subdecision,
- **self-play PPO** against a mixture of scripted and previous-policy opponents,
- and initially **disable domestic trading** until the rest of the game is stable.

That is the most realistic path to getting a working learning system rather than a permanently unfinished research project.

## Notes for future implementation
When we convert this into code, the parts most likely to cause bugs are:
- Longest Road graph logic
- robber move / steal sequencing
- bank shortage handling
- distance-rule validation on coastal intersections
- harbor ownership and timing semantics
- hidden-information observations vs full simulator state
- encoding domestic trades in a finite RL action space

## Sources consulted
- CATAN official rules landing page: https://www.catan.com/understand-catan/game-rules
- CATAN official base rules PDF: https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf
- CATAN official base-game FAQ: https://www.catan.com/faq/basegame
- OpenSpiel docs: https://openspiel.readthedocs.io/en/stable/intro.html
- Heinrich and Silver, Neural Fictitious Self-Play: https://discovery.ucl.ac.uk/id/eprint/1523603/
- Brown et al., Deep CFR: https://proceedings.mlr.press/v97/brown19b.html
- Brown et al., ReBeL: https://papers.nips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
- Settlers-RL project writeup: https://settlers-rl.github.io/
