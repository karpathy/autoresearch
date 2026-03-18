Storage — what one entry looks like
Normalize — std scaling before any similarity check
Retrieve — cosine similarity to find closest past experiments
Confidence — goes up on confirmation, down on contradiction
Update rule — high similarity + same verdict → update confidence, high similarity + opposite verdict → call LLM to resolve
Gate — before agent proposes next experiment, query memory first


FLOW

there'll be already present hyperparamters with some false values which later will be changed, when training starts, so we need to store the previous values and the new values AFTER NORMALIZING THE VALUE AS IT'LL CREATE PROBLEM AT THE TIME OF COSINE CALCULATION, and the confidence score of the new values, if the new values are better than the previous values then the confidence score will be increased, if the new values are worse than the previous values then the confidence score will be decreased, if the new values are same as the previous values then the confidence score will be same, if the new values are not same as the previous values then the confidence score will be decreased

So your architecture is actually:

High similarity + same verdict → just update confidence, no LLM needed
High similarity + opposite verdict → call LLM to resolve, that's your one expensive operation
