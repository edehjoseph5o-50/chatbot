
from datetime import datetime, timezone as tz_module
from zoneinfo import ZoneInfo

from fastmcp import FastMCP

mcp = FastMCP("Time Server")


@mcp.tool()
def get_current_time(timezone: str = "Africa/Lagos") -> str:
    """
    Returns the current date and time for a given IANA timezone.
    Examples: Africa/Lagos, Europe/London, America/New_York.
    """

    try:
        now = datetime.now(ZoneInfo(timezone))

        return now.strftime(
            "%A, %Y-%m-%d %H:%M:%S %Z"
        )

    except Exception:

        now = datetime.now(tz_module.utc)

        return now.strftime(
            "%A, %Y-%m-%d %H:%M:%S UTC (invalid timezone, fell back to UTC)"
        )


if __name__ == "__main__":
    mcp.run()
