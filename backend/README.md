# HackMIT 2025 Research Paper Comparison Tool - Backend

We are currently competing in a hackathon, under the education track. We think that currently, for casual researchers, it is difficult to compare research papers. In order to determine state-of-the-art papers, or compare papers quantitatively, it is necessary to
- have at least some domain knowledge in order to identify relevant quantitative benchmarks
- read through hundreds of papers and find benchmarks that are shared between competing papers
- keep up to date with new papers
- build a mental graph relating the hundreds of papers based on said metrics
Whenever new papers come out, it is necessary to keep up to date with journals and re-compare all relevant benchmarks, which is a tedious and irritating process.

In order to solve this, we propose a software that given a user input, like "explain the state of the art in model predictive control", "compare model predictive control to reinforcement learning", or "explain the history of model predictive control", will do the following:

1. Depending on the user input, an AI agent (which we will call the Front Agent) will determine a query to feed to arxiv to find one or more "seed" papers. For example, if the user says "explain the state of the art in model predictive control", the agent should output a search-describing object which lists the latest papers with the least amount of citations, and the top item of the search query will be a single seed. If the user says, "compare model predictive control to reinforcement learning", the agent should output two search-describing objects, recent with a good amount of citations or another relevant metric, one for model predictive control and one for reinforcement learning, and determine two "seed" papers. For a user input of "explain the history of model predictive control", the agent should output a search-describing object that looks for an old paper with a high amount of citations, then use the top element as a seed paper. If the user instead inputs a link to a paper or multiple papers (a literature review), the agent will use those papers as the seed papers.

2. Depending on the quantity of seeds produced by the last step, the front agent will tell the backend to spin up one or multiple webcrawlers and task them with exploring papers cited by and citing the seed papers. If only one seed is used, the agent might decide a maximum radius of node edges to limit the crawlers to. However, if more than one seed is used, the agent will need to connect all seed nodes with citation links, and maybe use some kind of algorithm to determine how much branching occurs in non-connecting directions. (We need to refine this section further, but the general idea is there). Citation discovery must occur in this stage, which requires each document visited to be parsed, however it's unknown whether we will use further LLMs (dubbed the Back Agents) here, or wait until step 3.

3. Once a graph of nodes is discovered, the backend will spin up a massively parallel process, spawning a domain expert agent per paper (perhaps with a blackboard, though that does reduce parallelism if not implemented correctly) with the goal of taking all explored documents, vectorizing or otherwise creating a base of knowledge about them, and using that representation to search each paper for all relevant metrics in said paper. Metrics must be discovered and stored in a way such that they are identifiable and "unique-keyed" between documents even if the agent does not have a blackboard. For example, if paper A and paper Z both reference LMArena scores, or RMSE tracking of a leg, this should be keyed into a global schema such that paper A, Z, and a future paper X can all be compared under the same benchmark. A quick textual summary of each paper should also be produced.

4. Finally, once every paper has had metrics discovered, the backend will output the graph, with edges of citations and node weights being a vector of shared benchmark between all papers (If this isn't possible we will figure out something else). From here this data structure can be passed off to the frontend, which another teammate is working on.

## Model Management

This project uses several large ML model files that exceed GitHub's file size limits. To manage these files:

### Setup and Installation

After cloning the repository, extract the model files:

```bash
# From the backend directory
python scripts/model_management.py extract_archive
```

This extracts the model files from `models.zip` into the `models/` directory.

### Before Committing Changes

If you've added or modified model files, archive them before committing:

```bash
# Create/update the models.zip archive
python scripts/model_management.py create_archive

# Clean up large model files from the repo
python scripts/model_management.py clean
```

This prevents Git from tracking the large files while preserving them in the zip archive.

### Model Details

See `models/README.md` for more information about the specific model files and how to manage them.
