from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(location:str)->str:
    """Get weather of a specific location"""
    return "It's always sunny in california"


if __name__== "__main__":
    mcp.run(transport="streamable-http")