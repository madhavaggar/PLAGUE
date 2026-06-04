import json


class PlagueConfig:
    def __init__(
        self,
        target_model,
        attacker_model,
        evaluator_model,
        summariser_model,
        embedding_model,
        project,
        max_rounds,
        use_strategy_library=True,
        dataset="harmbench",
        verbose=False,
        max_workers=6,
        max_tries=1,
        primer_steps=2,
        finisher_type="normal",
        use_iterative_prompting: bool = True,
        use_planner: bool = True,
        max_search_results: int = 2,
        use_actor_plan: bool = False,
    ):
        self.target_model = target_model
        self.attacker_model = attacker_model
        self.evaluator_model = evaluator_model
        self.summariser_model = summariser_model
        self.embedding_model = embedding_model
        self.project = project
        self.max_rounds = max_rounds
        self.dataset = dataset
        self.verbose = verbose
        self.max_workers = max_workers
        self.max_tries = max_tries
        self.primer_steps = primer_steps
        self.use_strategy_library = use_strategy_library
        self.finisher_type = finisher_type
        self.use_iterative_prompting = use_iterative_prompting
        self.use_planner = use_planner
        self.max_search_results = max_search_results
        self.use_actor_plan = use_actor_plan

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
