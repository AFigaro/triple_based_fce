from langchain.llms import OpenLLM

llm = OpenLLM(server_url='http://http://131.155.68.20:3000', server_type='grpc')
llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")