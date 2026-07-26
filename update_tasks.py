import json

tasks_file = 'agent_tasks.json'
with open(tasks_file, 'r') as f:
    data = json.load(f)

# Update current task
for task in data['tasks']:
    if task['id'] == 'openclaw-online-rl-dialogue-v6-e9301cfc':
        task['status'] = 'done'

# Add 3 new tasks
new_tasks = [
    {
        "id": "claude-agent-teams-subagent-spawner-v2",
        "status": "todo",
        "area": "agents",
        "risk": "medium",
        "title": "Claude Agent Teams Subagent Spawner V2",
        "description": "Inspired by Claude Agent Teams: Implement a robust subagent spawner for parallel tasks.",
        "allowed_paths": [
            "magda_agent/agents/subagent_spawner_v2.py",
            "tests/test_subagent_spawner_v2.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "Spawner can spawn independent agents",
            "Tests mock worktree provisioning and agent execution"
        ]
    },
    {
        "id": "openclaw-context-engine-hooks-v6",
        "status": "todo",
        "area": "memory",
        "risk": "medium",
        "title": "OpenClaw Context Engine Hooks V6",
        "description": "Inspired by OpenClaw trends: Implement advanced Context Engine hooks to manage token truncation dynamically.",
        "allowed_paths": [
            "magda_agent/memory/advanced_context_hooks_v6.py",
            "tests/test_advanced_context_hooks_v6.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "Context engine lifecycle correctly prioritizes high-salience tokens during context truncation.",
            "Tests verify truncation skips prioritized context items."
        ]
    },
    {
        "id": "mcp-dynamic-capability-negotiation-v7",
        "status": "todo",
        "area": "integration",
        "risk": "medium",
        "title": "MCP Dynamic Capability Negotiation V7",
        "description": "Inspired by MCP trends: Implement robust capability negotiation logic for tool handshakes.",
        "allowed_paths": [
            "magda_agent/integration/mcp_negotiation_v7.py",
            "tests/test_mcp_negotiation_v7.py",
            "agent_tasks.json"
        ],
        "acceptance": [
            "MCP negotiation logic handles capabilities handshake.",
            "Tests mock negotiation and verify fallback behaviors."
        ]
    }
]

data['tasks'].extend(new_tasks)

with open(tasks_file, 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
