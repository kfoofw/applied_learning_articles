Paper:

"From predictive uplift modeling to prescriptive uplift analytics: A practical approach to treatment optimization while accounting for estimation risk"

[Link here](https://sci-hub.se/10.1057/jma.2015.5)

- Objectives:
    - Optimizing uplift treatment under multiple treatment situations with limit budget
    - __Budget optimization is a hard problem to optimize as it is akin to a 0-1 knapsack problem. Binary treatment (0 or 1) optimization belongs to the class of NP-complete problems, which means that it cannot be solved in polynomial time.__
    - Proposed solution involves clustering-based grouping, which transforms the problem from a binary decision variable to a continuous variable. 
        - This can then be solved with linear programming which allows for application of widely accessible optimization tools (such as Excel Solver)
    - Authors also recognize the issue of variability/uncertainty in cluster groups' average uplift values
        - They proposed using bootstrap sampling to determine the uncertainty/variance of the cluster groups uplift values
        - Incorporate penalty of uncertainty/variance in the optimization through subtraction of standard deviation from expected uplift values for each cluster
            - Similar to Upper Confidence Bounds (UCB) Bandits approach but in a reverse manner
- Proposed Heuristics
    - Assume we have multiple treatments (M) with 1 control
    - Separate data into training and holdout data
    - Use training data to create uplift models:
        - Separate models approach for separate treatments
        - 1 control model, and M treatment models
    - With trained uplift models, use it to obtain the uplift/incremental lift treatment predictions (M predictions) on holdout set.
        - To obtain uplift value, calculate the incremental value prediction between the Treatment model and Control model for each individual unit (or client) for each treatment type.
    - At this point, for the holdout data, for each row/client, we have M uplift values
    - Perform clustering analysis on the clients to obtain K clusters (where K is to be determined from unsupervised clustering using the data)
        - Can do K-means clustering to obtain K clusters
        - Each user is represented by a vector of M uplift values
    - Once K clusters are obtained, calculate the cluster grouping average uplift values for M treatments
    - Use linear programming to optimize budget for various treatment campaigns. I.e. for a given budget, we have M treatment types, for each cluster grouping, we can afford X_{k,m} treatments, where k represents the cluster and m represents the treatment.
    - Once we have those values, actual treatment assignment can be performed via the following two methods:
        - Random sampling for each client in each cluster
        - For each client, get their predict_probability for each of the M treatments. Do some further form of ranking optimization to prioritize the treatments since it is possible some users may have high predict_probabilities for multiple treatments but we can only show one treatment per user.