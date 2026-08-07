with open("magda_agent/safety/resource_sandbox_v1.py", "r") as f:
    content = f.read()

new_content = content.replace("raise TimeoutError(\"Sandbox execution exceeded CPU time limit or hung.\")", "raise TimeoutError(\"Sandbox execution exceeded CPU time limit.\")")

with open("magda_agent/safety/resource_sandbox_v1.py", "w") as f:
    f.write(new_content)
