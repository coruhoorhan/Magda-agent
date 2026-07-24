import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import httpx
from magda_agent.integration.a2a_security import A2ASecurityContext


@dataclass
class AgentCardV5:
    """
    Represents the capabilities and identity of an agent in the network, version 5.
    Inspired by trend: A2A (Agent-to-Agent Protocol).
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    protocol_version: str = "v5"

    def to_json(self) -> str:
        """
        Serializes the AgentCardV5 to a JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCardV5":
        """
        Deserializes an AgentCardV5 from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)


class A2ADiscoveryServiceV5:
    """
    A registry to manage external agent discovery using Agent Cards (V5).
    """

    def __init__(self, security_context: Optional[A2ASecurityContext] = None) -> None:
        """
        Initializes the A2ADiscoveryServiceV5.
        """
        self._registry: Dict[str, AgentCardV5] = {}
        self.security_context = security_context or A2ASecurityContext()

    def register_agent(self, card: AgentCardV5) -> None:
        """
        Registers an AgentCardV5 in the registry.

        Args:
            card (AgentCardV5): The agent card to register.
        """
        self._registry[card.agent_id] = card
        logging.info(f"Registered AgentCardV5 for agent_id: {card.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregisters an agent from the registry by its ID.

        Args:
            agent_id (str): The ID of the agent to unregister.
        """
        if agent_id in self._registry:
            del self._registry[agent_id]
            logging.info(f"Unregistered agent_id: {agent_id}")
        else:
            logging.warning(f"Attempted to unregister non-existent agent_id: {agent_id}")

    def parse_and_register_cards(self, raw_cards: List[str]) -> List[AgentCardV5]:
        """
        Parses a list of JSON string representations of Agent Cards
        and registers them in the internal store.

        Args:
            raw_cards (List[str]): A list of JSON strings representing AgentCardV5s.

        Returns:
            List[AgentCardV5]: A list of successfully parsed and registered AgentCardV5 objects.
        """
        successfully_parsed: List[AgentCardV5] = []
        for card_json in raw_cards:
            try:
                card = AgentCardV5.from_json(card_json)
                self.register_agent(card)
                successfully_parsed.append(card)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logging.error(f"Failed to parse AgentCardV5. Error: {str(e)}, Raw Data: {card_json}")
        return successfully_parsed

    def get_agent_card(self, agent_id: str) -> Optional[AgentCardV5]:
        """
        Retrieves a registered AgentCardV5 by its agent_id.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            Optional[AgentCardV5]: The AgentCardV5 if found, otherwise None.
        """
        return self._registry.get(agent_id)

    def get_all_agents(self) -> List[AgentCardV5]:
        """
        Returns a list of all discovered agent cards.

        Returns:
            List[AgentCardV5]: A list of all AgentCardV5 objects in the registry.
        """
        return list(self._registry.values())

    async def discover_from_network(self, network_endpoint: str, auth_token: Optional[str] = None) -> List[AgentCardV5]:
        """
        Discovers peers from a network endpoint and parses Agent Cards, maintaining an internal registry of discovered agents.

        Args:
            network_endpoint (str): The endpoint to discover agents from.
            auth_token (Optional[str]): An optional authentication token for secure discovery.

        Returns:
            List[AgentCardV5]: A list of newly discovered AgentCardV5 objects.
        """
        if auth_token and not self.security_context.validate_token(auth_token):
            logging.error("Invalid auth token for discover_from_network")
            raise ValueError("Invalid authentication token")

        self.security_context.trace_action("discover_from_network_v5", {"endpoint": network_endpoint})
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{network_endpoint}", headers=headers)
                response.raise_for_status()
                cards_data = response.json()

                # Assume cards_data is a list of json strings or dicts
                raw_cards = []
                for item in cards_data:
                    if isinstance(item, str):
                        raw_cards.append(item)
                    else:
                        raw_cards.append(json.dumps(item))

                return self.parse_and_register_cards(raw_cards)
        except Exception as e:
            logging.error(f"Failed to discover from network {network_endpoint}: {e}")
            return []
