from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from .router_executor import RouterAgentExecutor
from .react_router_executor import ReactRouterExecutor


class DispatchingExecutor(AgentExecutor):
    """
    An executor that dispatches requests to other executors based on
    the 'orchestrator_mode' specified in the message metadata.
    """

    def __init__(self):
        self._simple_executor = RouterAgentExecutor()
        self._react_executor = ReactRouterExecutor()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Examines the message and delegates to the appropriate executor.
        """
        # Default to 'simple' mode if not specified
        mode = "simple"
        if context.message and context.message.metadata:
            mode = context.message.metadata.get("orchestrator_mode", "simple")

        if mode == "react":
            await self._react_executor.execute(context, event_queue)
        else:
            await self._simple_executor.execute(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        # The cancellation logic could also be dispatched, but for now, we can
        # assume it's the same for both or delegate to the simple one.
        return await self._simple_executor.cancel(context, event_queue)
