# V0 Architecture

A cheif source of inspiration for this project is OpenAi dota 5's architecture.

The main difference is that we are using a CNN to process the observations, but a custom encoder that processes the raw observations into a format that can be fed into the LSTM network.

The reason for this is we would allow the structure to be more modular so we can switch parts in needed.

The architecture is as follows:

## Observation Encoder
* Spatial encoder - Processes map features like energy, tile type, vision power, and units mask
* Unit set processor - Processes variable-sized sets of units using Process Set approach
* Node processor - Processes energy and relic nodes
* LuxAI observation encoder - Combines the above three layers
* Final layer - Maps the combined embedding to a fixed dimension

## Observation Processor

Action selection (using policy gradient)
Unit selection (using policy gradient)
Coordinate prediction (using MSE weighted by rewards)

## Reward Function

https://arxiv.org/abs/2104.13906

This is gonna be very tricky.

Overall we want to win but by tuning it to expicitly winning a single match is going to be sparse as it found every 100 steps.
If we think about winning 3/5 matches. We essentially get the best result in 300 steps or worst case in 500. The sparsity will be

Points

Let us set the following rewards and tune them later. The relative order is going to be the same in this case.
* Match Won
* Round Won
* Relic Points in a game
* Total Energy
* Unit killed
* Percent of relic tiles found
* Percent of area explored

Match Won = 5 points
Round Won = 1 point
Relic Points = 0.2 point
Total Energy as a fraction of max energy = 1 point

*Team Spirit:*

This is a team game so as they say:

> Needs of the many outweigh the needs of the few - Spock

Let us say variable $\mu$ represent how much a team shares the reward with others.

We can define the rewards as:
$$ \mu  $$

I am just a chill guy, i will not have enemies data in my reward function. Yes, v1 will be different.

![I am chill guy](https://i1.sndcdn.com/artworks-9nBKzBuDB1qvt2pp-tfzeGw-t500x500.png)

*Long term credit assignment*




## V1 things

* Add embedding layer in UnitSetProcessor to create specialization and difference. Think of scouting bots which explore and the main force goes to attack.
* Non-linear rewards $\frac{(x+1-(1-x)^{4})}{2}$
