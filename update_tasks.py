import json

with open('agent_tasks.json', 'r') as f:
    data = json.load(f)

for task in data['tasks']:
    if task['id'] == 'mcp-action-tool-concurrency-v2-d30c8711':
        task['status'] = 'done'

new_tasks = [
    {
      "id": "mcp-dynamic-tool-permissions-v1-unique",
      "title": "MCP Dynamic Tool Permissions v1",
      "description": "Inspired by MCP and A2A trends: Implement a dynamic tool permission system where specific Magda skills can be selectively exported or granted to remote A2A agents based on role-based access control.",
      "status": "todo",
      "area": "safety",
      "risk": "high",
      "allowed_paths": [
        "magda_agent/safety/mcp_permissions_v1.py",
        "tests/test_mcp_permissions_v1.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "Role-based access control implemented for MCP tools.",
        "Tests verify tool execution blocking for unauthorized agents."
      ]
    },
    {
      "id": "openclaw-rl-engagement-feedback-v1-unique",
      "title": "OpenClaw RL Engagement Feedback v1",
      "description": "Inspired by OpenClaw RL online learning trend: Expand implicit feedback extraction to monitor user engagement metrics (like response delay or interaction length) to tune conversational brevity weights.",
      "status": "todo",
      "area": "learning",
      "risk": "medium",
      "allowed_paths": [
        "magda_agent/learning/engagement_feedback_v1.py",
        "tests/test_engagement_feedback_v1.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "Engagement metrics extracted from user interactions.",
        "RL weights adjusted dynamically based on engagement trends.",
        "Tests verify habit weight shifts."
      ]
    },
    {
      "id": "claude-agent-teams-context-distillation-v1-unique",
      "title": "Claude Agent Teams Context Distillation v1",
      "description": "Inspired by Claude Agent Teams trend: Implement a context distillation mechanism where subagents compress their final outputs into dense semantic representations before returning them to the parent agent, minimizing context window explosion.",
      "status": "todo",
      "area": "agents",
      "risk": "medium",
      "allowed_paths": [
        "magda_agent/agents/context_distillation_v1.py",
        "tests/test_context_distillation_v1.py",
        "agent_tasks.json"
      ],
      "acceptance": [
        "Subagent output is distilled before returning.",
        "Parent agent receives and parses compressed context successfully.",
        "Tests verify compression and parsing functionality."
      ]
    }
]

data['tasks'].extend(new_tasks)

with open('agent_tasks.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

print("Updated agent_tasks.json")
