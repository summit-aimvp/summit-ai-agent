from langchain.tools import DuckDuckGoSearchRun

def search_web(query):
    search = DuckDuckGoSearchRun()
    return search.run(query)
