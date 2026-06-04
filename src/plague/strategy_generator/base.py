from abc import ABC, abstractmethod


class AdversarialStrategy(ABC):
    def __init__(self, category: str, definition: str, examples=[], goals=[]):
        self.category = category
        self.definition = definition
        self.scores = []
        self.average_score = 0
        self.num_trials = 0
        self.feedbacks = []
        self.examples = examples
        self.strategy_embedding = []
        self.retrieval_embedding = []
        self.goals = goals

    def update_score(self, score: float):
        self.average_score = (self.average_score * self.num_trials + score) / (
            self.num_trials + 1
        )
        self.num_trials += 1

    def update_feedback(self, feedback: str):
        self.feedbacks.append(feedback)

    def update_examples(self, examples):
        self.examples.append(examples)

    def print(self):
        return f"Strategy is {self.category}, definition is {self.definition}, examples are {self.examples}"


class AdversarialStrategyGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_strategies(self, goal: str, **kwargs):
        pass
