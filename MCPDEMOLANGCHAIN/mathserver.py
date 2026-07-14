import math
from mcp.server.fastmcp import FastMCP

#math is a server name
mcp = FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """Add two numbers"""
    return a+b

@mcp.tool()
def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b

#The transport="stdio" argument tells the server to:

#Use standard input/output (stdin and stdout) to receive and respond to tool function calls.
# stdio is hepful if you want to test the server locally and test it with the client
if __name__ == "__main__":
    mcp.run(transport="stdio")