"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    - Math tag validation
    - Math tag presence
    """
    
    def __init__(self):
        self.num_reward_functions = 9

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML tags."""
        try:
            content = text.split(f"<{tag}>")[-1]
            content = content.split(f"</{tag}>")[0]
            return content.strip()
        except:
            return ""
    
    def _extract_xml_content_with_parent(self, text: str, parenttag: str, tag: str) -> str:
        """Extract content from XML tags."""
        try:
            parent_content = text.split(f"<{parenttag}>")[-1]
            parent_content = parent_content.split(f"</{parenttag}>")[0]
            return self._extract_xml_content(parent_content, tag)
        except:
            return ""

    def _extract_solution_answer(self, text: str) -> str:
        """Extract answer from solution part."""
        return self._extract_xml_content_with_parent(text, "solution", "answer")

    def _extract_student_answer(self, text: str) -> str:
        """Extract answer from student part."""
        return self._extract_xml_content_with_parent(text, "student", "answer")

    # === rewards ===
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correct answer in solution part."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_solution_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def _incorrectness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for incorrect answer in student part."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_student_answer(r) for r in responses]
        return [0.0 if r == a else 1.0 for r, a in zip(extracted, answer)]

    def _solutions_differ_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for solutions differing in student part."""
        responses = [completion[0]['content'] for completion in completions]
        extracted_solution = [self._extract_solution_answer(r) for r in responses]
        extracted_student = [self._extract_student_answer(r) for r in responses]
        return [0.0 if r == a else 1.0 for r, a in zip(extracted_solution, extracted_student)]

    def _int_format_reward_solution_answer(self, completions) -> List[float]:
        """Reward for integer format in solution answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_solution_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    def _int_format_reward_student_answer(self, completions) -> List[float]:
        """Reward for integer format in student answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_student_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    # make sure there are only one pair of answer tags in the solution and student parts
    def _answer_tags_one_per_part_reward(self, completions) -> List[float]:
        """Reward for having exactly one pair of answer tags in each part."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            # Count answer tags in solution part
            solution_part = self._extract_xml_content(response, "solution")
            solution_answer_tags = len(re.findall(r"<answer>.*?</answer>", solution_part))
            
            # Count answer tags in student part
            student_part = self._extract_xml_content(response, "student")
            student_answer_tags = len(re.findall(r"<answer>.*?</answer>", student_part))
            
            # Reward if exactly one pair in each part
            reward = 1.0 if solution_answer_tags == 1 and student_answer_tags == 1 else 0.0
            rewards.append(reward)
            
        return rewards

    def _strict_tag_ordering_reward(self, completions) -> List[float]:
        """Strict reward for correct tag ordering and structure."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            # Check for exact pattern: <solution>...<answer>...</answer>...</solution><student>...<answer>...</answer>...</student>
            pattern = r"^\s*<solution>\s*<answer>.*?</answer>.*?</solution>\s*<student>\s*<answer>.*?</answer>.*?</student>\s*$"
            reward = 3.0 if re.match(pattern, response, re.DOTALL) else 0.0
            rewards.append(reward)
            
        return rewards

    def _soft_tag_ordering_reward(self, completions) -> List[float]:
        """Softer reward for tag ordering that allows for more flexibility."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            # Check if solution comes before student
            solution_pos = response.find("<solution>")
            student_pos = response.find("<student>")
            
            if solution_pos == -1 or student_pos == -1 or solution_pos > student_pos:
                rewards.append(0.0)
                continue
                
            # Check if each part has its answer tag
            solution_part = response[solution_pos:student_pos]
            student_part = response[student_pos:]
            
            has_solution_answer = "<answer>" in solution_part and "</answer>" in solution_part
            has_student_answer = "<answer>" in student_part and "</answer>" in student_part
            
            reward = 1.0 if has_solution_answer and has_student_answer else 0.0
            rewards.append(reward)
            
        return rewards

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._incorrectness_reward(prompts, completions, answer),
            self._solutions_differ_reward(prompts, completions, answer),
            self._int_format_reward_solution_answer(completions),
            self._int_format_reward_student_answer(completions),
            self._answer_tags_one_per_part_reward(completions),
            self._strict_tag_ordering_reward(completions),
            self._soft_tag_ordering_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/incorrectness_reward_func": reward_per_func[1].item(),
            "rewards/solutions_differ_reward_func": reward_per_func[2].item(),
            "rewards/int_format_solution_reward_func": reward_per_func[3].item(),
            "rewards/int_format_student_reward_func": reward_per_func[4].item(),
            "rewards/answer_tags_one_per_part_reward_func": reward_per_func[5].item(),
            "rewards/strict_tag_ordering_reward_func": reward_per_func[6].item(),
            "rewards/soft_tag_ordering_reward_func": reward_per_func[7].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'incorrectness': reward_scores[1].item(),
            'solutions_differ': reward_scores[2].item(),
            'int_format_solution': reward_scores[3].item(),
            'int_format_student': reward_scores[4].item(),
            'answer_tags_one_per_part': reward_scores[5].item(),
            'strict_tag_ordering': reward_scores[6].item(),
            'soft_tag_ordering': reward_scores[7].item()
        }
