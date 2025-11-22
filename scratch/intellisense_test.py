import modaic

agent = modaic.AutoAgent.from_precompiled("fadeleke/prompt-to-signature")
result = agent("Summarize a document and extract key entities")
print(result)
