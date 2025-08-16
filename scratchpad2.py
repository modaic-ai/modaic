from modaic import AutoAgent, AutoConfig, AutoIndexer

cfg = AutoConfig.from_precompiled("modaic-ai/hub_test")
agent = AutoAgent.from_precompiled("modaic-ai/hub_test")
indexer = AutoIndexer.from_precompiled("modaic-ai/hub_test")

print(cfg.hello)
print(cfg.bye)

print(agent.forward())
print(indexer.retrieve())
print(indexer.ingest())
