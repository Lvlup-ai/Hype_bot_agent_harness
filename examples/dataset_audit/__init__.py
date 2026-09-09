"""Dataset quality audit: an end-to-end run of the harness in a neutral domain.

Three scripted agents (proposer, auditor, reviewer) look for rules that flag
the anomalous rows of a small table. No model is called: each agent is a
Python function that plays its role, and says where a model call would go.
"""
