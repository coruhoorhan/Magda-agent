import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from magda_agent.llm_client import LLMClient


class MagenticOneWorkerV3:
    """
    Represents a specialized worker subagent in the Magentic-One V3 orchestration pattern.
    """

    def __init__(self, name: str, description: str, llm: LLMClient) -> None:
        """
        Initializes the MagenticOneWorkerV3.

        Args:
            name (str): The name of the specialized worker.
            description (str): A description of the worker's capabilities.
            llm (LLMClient): The language model client.
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.state: Dict[str, Any] = {}

    async def execute_subtask(self, subtask: str, context: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Executes a specific subtask and returns both string outcome and a structured state update.

        Args:
            subtask (str): The subtask to be executed.
            context (List[str]): Context strings for the current run.

        Returns:
            Tuple[str, Dict[str, Any]]: The output of the task execution, and any state variables it updated.
        """
        prompt = (
            f"You are the specialized agent {self.name} ({self.description}).\n"
            f"Execute this task: {subtask}\n"
            f"Current context: {context}\n"
            "Return the outcome text, followed by 'STATE_UPDATE:' and a JSON dictionary of updated state variables."
        )
        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])

            outcome = response
            state_update = {}
            if "STATE_UPDATE:" in response:
                parts = response.split("STATE_UPDATE:", 1)
                outcome = parts[0].strip()
                try:
                    clean_json = parts[1].strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:-3].strip()
                    elif clean_json.startswith("```"):
                        clean_json = clean_json[3:-3].strip()
                    state_update = json.loads(clean_json)
                except json.JSONDecodeError:
                    logging.warning(f"Worker {self.name} returned invalid state update JSON.")

            self.state.update(state_update)
            return outcome, state_update

        except Exception as e:
            logging.error(f"Worker {self.name} failed: {e}")
            return f"Worker {self.name} encountered an error: {e}", {}


class MagenticOneOrchestratorV3:
    """
    Implements a multi-agent orchestration pattern inspired by Microsoft's Magentic-One.

    V3 efficiently routes tasks to parallel subagents and merges state natively across workers.
    """

    def __init__(self, llm: LLMClient, base_workers: Optional[List[MagenticOneWorkerV3]] = None) -> None:
        """
        Initializes the MagenticOneOrchestratorV3.

        Args:
            llm (LLMClient): The language model client to be used.
            base_workers (Optional[List[MagenticOneWorkerV3]]): Pre-defined permanent workers.
        """
        self.llm = llm
        if base_workers is None:
            self.base_workers = [
                MagenticOneWorkerV3("WebSurfer", "Specialized in web browsing, search, and navigation.", llm),
                MagenticOneWorkerV3("FileSurfer", "Specialized in reading, writing, and navigating the filesystem.", llm),
                MagenticOneWorkerV3("Coder", "Specialized in writing code and logic scripts.", llm),
                MagenticOneWorkerV3("Executor", "Specialized in compiling, executing, and testing code.", llm)
            ]
        else:
            self.base_workers = base_workers
        self.active_workers: List[MagenticOneWorkerV3] = []
        self.global_state: Dict[str, Any] = {}

    def _evaluate_difficulty(self, task: str) -> int:
        """
        Evaluates the difficulty of the task based on heuristics to dynamically scale the team size.

        Args:
            task (str): The task to evaluate.

        Returns:
            int: An integer between 1 and 10 representing difficulty.
        """
        length = len(task)
        if length < 20:
            return 2
        elif length < 50:
            return 5
        elif length < 100:
            return 8
        else:
            return 10

    def _calculate_team_size(self, difficulty: int) -> int:
        """
        Calculates the appropriate team size based on task difficulty.

        Args:
            difficulty (int): The evaluated task difficulty (1-10).

        Returns:
            int: The calculated team size (1-5).
        """
        if difficulty <= 2:
            return 1
        elif difficulty <= 4:
            return 2
        elif difficulty <= 6:
            return 3
        elif difficulty <= 8:
            return 4
        else:
            return 5

    def _spawn_workers(self, team_size: int) -> None:
        """
        Spawns workers up to the required team size.

        Args:
            team_size (int): The number of active workers required for the task.
        """
        self.active_workers = []
        for i in range(team_size):
            if i < len(self.base_workers):
                self.active_workers.append(self.base_workers[i])
            else:
                self.active_workers.append(MagenticOneWorkerV3(
                    f"DynamicWorker_{i}",
                    "Specialized dynamically spawned worker for extra parallel tasks.",
                    self.llm
                ))
        logging.info(f"Spawned {len(self.active_workers)} active workers for the task.")

    def _kill_workers(self) -> None:
        """
        Kills dynamic workers and resets active workers to release resources.
        """
        count = len(self.active_workers)
        self.active_workers = []
        logging.info(f"Killed active workers, released {count} resources.")

    async def _plan(self, task: str, context: List[str], team_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Creates a plan by generating subtasks based on the main task, current context, global state, and team size.

        Args:
            task (str): The task for which a plan should be created.
            context (List[str]): Context representing execution history.
            team_size (Optional[int]): Number of parallel agents needed.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the tasks.
        """
        size_prompt = f" Generate EXACTLY {team_size} subtasks." if team_size else ""
        state_str = json.dumps(self.global_state) if self.global_state else "{}"
        prompt = (
            f"Plan task: {task}. Context: {context}. Global State: {state_str}.{size_prompt} "
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'id' and 'description' keys."
        )
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            parsed_plan = json.loads(clean_response)
            if isinstance(parsed_plan, list) and len(parsed_plan) > 0 and 'id' in parsed_plan[0] and 'description' in parsed_plan[0]:
                return parsed_plan[:team_size] if team_size else parsed_plan
        except Exception as e:
            logging.error(f"Failed to parse plan JSON: {e}. Raw response: {response}")

        num_tasks = team_size if team_size else 1
        return [{"id": f"fallback_subtask_{i}", "description": f"Attempt subtask {i} for {task}"} for i in range(1, num_tasks + 1)]

    async def _execute_step(self, step: Dict[str, Any], context: List[str], step_index: int) -> Tuple[str, Dict[str, Any]]:
        """
        Executes a single step in the plan using an active worker if available.

        Args:
            step (Dict[str, Any]): The step object to execute.
            context (List[str]): Ongoing context strings.
            step_index (int): Execution index to map steps to workers in a round-robin style.

        Returns:
            Tuple[str, Dict[str, Any]]: Resulting output string and the updated state dictionary.
        """
        description = step.get('description', '')

        if self.active_workers:
            worker = self.active_workers[step_index % len(self.active_workers)]
            return await worker.execute_subtask(description, context)

        # Fallback if no active workers
        prompt = f"Execute subtask: {description}"
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        return response, {}

    def _merge_state(self, new_states: List[Dict[str, Any]]) -> None:
        """
        Merges individual subagent states into the orchestrator's global state natively.

        Args:
            new_states (List[Dict[str, Any]]): A list of state update dictionaries.
        """
        for state in new_states:
            if isinstance(state, dict):
                # Basic shallow merge for native state merging
                for k, v in state.items():
                    if k in self.global_state and isinstance(self.global_state[k], list) and isinstance(v, list):
                        self.global_state[k].extend(v)
                    elif k in self.global_state and isinstance(self.global_state[k], dict) and isinstance(v, dict):
                        self.global_state[k].update(v)
                    else:
                        self.global_state[k] = v

    async def _execute_plan(self, plan: List[Dict[str, Any]], context: Optional[List[str]] = None) -> List[str]:
        """
        Executes a plan by delegating subtasks to active workers concurrently. State updates are merged natively.

        Args:
            plan (List[Dict[str, Any]]): The list of task objects generated by _plan.
            context (Optional[List[str]]): List of context strings.

        Returns:
            List[str]: Combined string results list from subagents.
        """
        if context is None:
            context = []

        tasks = []
        for i, step in enumerate(plan):
            tasks.append(self._execute_step(step, context, i))

        results = await asyncio.gather(*tasks)

        string_results = []
        state_updates = []

        for res, state in results:
            string_results.append(res)
            state_updates.append(state)

        self._merge_state(state_updates)

        return string_results

    async def _review(self, task: str, context: List[str]) -> Tuple[bool, str]:
        """
        Reviews the current context and global state to determine if the main task is complete.

        Args:
            task (str): Main orchestrator task string.
            context (List[str]): List of gathered execution string contexts.

        Returns:
            Tuple[bool, str]: Success boolean flag, and string response payload.
        """
        state_str = json.dumps(self.global_state)
        prompt = f"Review task: {task}. Context: {context}. Global State: {state_str}. Is complete? Return YES or NO, then the result."
        res = await self.llm.chat_completion([{"role": "user", "content": prompt}])

        is_complete = "YES" in res.upper()
        return is_complete, res

    async def orchestrate(self, task: str, max_iterations: int = 3) -> str:
        """
        Orchestrates the execution of a complex task by planning, parallel subagent delegation, native state merging, and reviewing.

        Args:
            task (str): The main task to accomplish.
            max_iterations (int): Maximum number of plan-execute-review loops.

        Returns:
            str: The final result of the orchestration process.
        """
        context: List[str] = []
        self.global_state = {}

        difficulty = self._evaluate_difficulty(task)
        team_size = self._calculate_team_size(difficulty)

        self._spawn_workers(team_size)

        try:
            for _ in range(max_iterations):
                # Step 1: Planning
                plan = await self._plan(task, context, team_size)

                # Step 2: Delegation and Execution with Native State Merging
                execution_results = await self._execute_plan(plan, context)
                context.extend(execution_results)

                # Step 3: Review
                is_complete, final_result = await self._review(task, context)
                if is_complete:
                    return final_result

            return f"Task incomplete after {max_iterations} iterations. Last context: {context}. Final State: {self.global_state}"
        finally:
            self._kill_workers()
